# sensenova-loop

A command-line tool, written in Rust, that runs
[SenseNova-Loop-9B-Preview](https://huggingface.co/sensenova/SenseNova-Loop-9B-Preview)
locally on your GPU and serves it through an **OpenAI-compatible API**, so any
tool that talks to OpenAI (Open WebUI, Continue, Cline, LibreChat, LangChain,
the `openai` SDKs, ...) can use it.

* **Native inference engine** — the model is implemented from scratch in Rust
  with hand-written WGSL compute kernels. No Python, PyTorch, CUDA toolkit or
  llama.cpp fork needed.
* **Any GPU, any desktop OS** — runs on [wgpu](https://wgpu.rs): Vulkan on
  Windows/Linux (NVIDIA, AMD, Intel), Metal on macOS (Apple Silicon and AMD),
  DirectX 12 as a fallback. So far tested on Windows 11 with an NVIDIA RTX 5080
  (Vulkan and DX12) and an AMD Radeon integrated GPU (Vulkan); Linux and macOS
  builds use the same portable code but have not been run yet.
* **Faithful to the architecture** — Gated DeltaNet linear attention, gated
  full attention with partial RoPE, the looped middle block (layers 4–27 run
  twice with shared weights) and the 2-stream mHC hyper-connections with
  Sinkhorn routing. Logits were checked against a PyTorch FP32 reference.
* **OpenAI API** — `/v1/chat/completions` (streaming, tool calling,
  `reasoning_content`), `/v1/completions`, `/v1/models`.
* **Prefix caching** — multi-turn chats and agent loops reuse the already
  processed conversation instead of re-reading it every turn.

## Quick start

```bash
cargo build --release
```

```bash
./target/release/sensenova-loop pull
```

`pull` downloads the official weights (~38 GB) and quantizes them into a
single file (Q8_0 by default, ~10 GB). Use `--quant q4_0` for a ~7 GB file that
fits smaller GPUs, and `--delete-source` to remove the 38 GB download afterwards.
Set `HF_TOKEN` if you need authenticated downloads. Interrupted downloads resume.

Chat in the terminal:

```bash
./target/release/sensenova-loop run
```

Or answer a single prompt:

```bash
./target/release/sensenova-loop run -p "Explain looped transformers in two sentences." --no-think
```

## OpenAI-compatible server

```bash
./target/release/sensenova-loop serve --port 8080
```

Point any OpenAI client at `http://127.0.0.1:8080/v1` (any API key works unless
you pass `--api-key`). The model id is `sensenova_loop`, but any model name is
accepted.

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="none")
reply = client.chat.completions.create(
    model="sensenova_loop",
    messages=[{"role": "user", "content": "Write a haiku about the ocean at night."}],
)
print(reply.choices[0].message.reasoning_content)  # the model's thinking
print(reply.choices[0].message.content)
```

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d '{"messages":[{"role":"user","content":"Hello!"}],"stream":true}'
```

Supported request fields: `messages`, `stream`, `stream_options.include_usage`,
`max_tokens` / `max_completion_tokens`, `temperature`, `top_p`, `top_k`,
`min_p`, `presence_penalty`, `frequency_penalty`, `repetition_penalty`,
`seed`, `stop`, `tools`, `tool_choice` (`"none"` disables tools).

**Thinking mode** is on by default; the reasoning is returned separately in
`reasoning_content`. Turn it off per request with
`"chat_template_kwargs": {"enable_thinking": false}` (vLLM convention) or
`"reasoning_effort": "none"`, or for the whole server with `--no-think`.
Clients that expect `<think>` tags inline in the content can use
`--inline-reasoning`.

**Tool calling** follows the OpenAI format: the model's `<tool_call>` output is
parsed into `tool_calls` with JSON `arguments`, and `tool` role messages are
fed back through the model's chat template.

**Network access:** the server binds to `127.0.0.1` by default. Use
`--host 0.0.0.0` together with `--api-key` to expose it on your network.

## Commands

| Command | Purpose |
| --- | --- |
| `pull [REPO] [--quant q8_0\|q4_0] [--delete-source]` | Download and quantize a model |
| `convert <DIR> [--quant ...]` | Quantize an already-downloaded Hugging Face checkpoint |
| `run [-p PROMPT] [--system TEXT] [--hide-reasoning] [--stats]` | Terminal chat (`/think`, `/nothink`, `/reset`, `/system`, `/exit`) |
| `serve [--host] [--port] [--api-key] [--served-model-name] [--inline-reasoning]` | OpenAI-compatible server |
| `devices` | List usable GPUs |
| `bench` | Measure prompt processing and generation speed |

Options shared by `run`, `serve` and `bench`:

* `--model <file.snl or dir>` — model to load (defaults to the pulled model).
* `--quant q8_0|q4_0` — choose between converted variants.
* `--ctx <tokens>` — context size; the KV cache is allocated up front (default 32768).
* `--backend vulkan|dx12|metal|gl` and `--device <index>` — pick the GPU (see `devices`).
* Sampling defaults: `--temperature 1.0 --top-p 0.95 --top-k 20` (from the
  model's generation config); `--max-tokens 8192`.

Models are stored in your local data directory
(`%LOCALAPPDATA%\sensenova-loop` on Windows, `~/.local/share/sensenova-loop`
on Linux, `~/Library/Application Support/sensenova-loop` on macOS), or wherever
`--models-dir` / `SENSENOVA_HOME` points.

## Performance

Measured on an RTX 5080 (16 GB, Vulkan, Windows 11):

| Weights | File | Prompt processing | Generation |
| --- | --- | --- | --- |
| Q8_0 | 10.1 GB | ~380 tok/s | ~36 tok/s |
| Q4_0 | 7.2 GB | ~420 tok/s | ~42 tok/s |

Because the looped block runs twice, every token executes 56 decoder layers,
so the model costs about as much per token as a ~14B dense model.

GPU memory needed is roughly the weight file plus ~60 KB per context token
(about 11.4 GB for Q8_0 at a 32K context). If generation is unexpectedly slow,
other applications may be holding GPU memory and forcing the driver to page:
lower `--ctx` or use `--quant q4_0`. On Windows, prefer the Vulkan backend
(the default); DX12 works but compiles shaders slowly and runs slower.

## Limitations

* Text only for now: the vision encoder is not implemented, and requests with
  image or video parts are rejected with a 400 error.
* One request is processed at a time; concurrent requests are queued.
* Multi-token prediction (MTP) speculative decoding is not used.

## How it works

`src/engine.rs` records the whole forward pass as a fixed list of GPU
dispatches in execution-plan order. Shared weights are bound to both visits of
the looped block, while each visit has its own KV cache or recurrent state.
Kernels live in `src/shaders/`: quantized matrix-vector and tiled matrix
multiplication (`matmul.wgsl`), Gated DeltaNet (`linear.wgsl`), GQA attention
with split-K decoding (`attention.wgsl`), and norms / mHC routing / SwiGLU
(`layers.wgsl`). The chat template is rendered with MiniJinja, exactly as
shipped with the model.
