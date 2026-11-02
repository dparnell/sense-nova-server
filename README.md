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
* **Images and video** — the model's vision encoder runs on the GPU too, with
  the same preprocessing and 3D (M-RoPE) positions as the reference
  implementation.
* **OpenAI API** — `/v1/chat/completions` (streaming, tool calling,
  `reasoning_content`, image and video inputs), `/v1/completions`, `/v1/models`.
* **Prefix caching** — multi-turn chats and agent loops reuse the already
  processed conversation instead of re-reading it every turn.
* **Speculative decoding** — the model's own multi-token-prediction (MTP)
  head drafts the next token and the main model verifies it in the same pass,
  for about 1.5x faster generation with exactly the same output.

## Quick start

```bash
cargo build --release
```

```bash
./target/release/sensenova-loop pull
```

`pull` downloads the official weights (~38 GB) and quantizes them (Q8_0 by
default, ~10 GB, plus a 0.5 GB `vision.snl` for the image encoder and a
0.3 GB `mtp.snl` for speculative decoding). If you pulled with an older
version, run `pull` again: it skips what is already downloaded and converted
and only adds what is missing. Use `--quant q4_0` for a ~7 GB file that
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

Ask about an image or a video:

```bash
./target/release/sensenova-loop run --image photo.jpg -p "What is in this picture?"
```

In the interactive chat, `/image <path or URL>` and `/video <path or URL>`
attach media to your next message.

## OpenAI-compatible server

```bash
./target/release/sensenova-loop serve --port 8080
```

In a terminal, `serve` shows a live dashboard: what the model is doing right
now (reading the prompt or generating, with progress and context use), charts
of generation and prompt-processing speed, GPU load and VRAM use over time,
GPU temperature and power, the process's CPU and RAM, request totals and a
table of recent requests with their speed and outcome. Press `q` (or Ctrl+C)
to stop the server. GPU load, VRAM, temperature and power come from NVIDIA's
NVML; on other GPUs the chart shows how busy the engine kept the GPU instead.
Use `--no-tui` (or redirect the output) for plain one-line-per-request logs.

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

**Images and video** use the usual OpenAI content parts. Media can be a
`data:` URL (base64) or an `http(s)` URL:

```python
reply = client.chat.completions.create(
    model="sensenova_loop",
    messages=[{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
        {"type": "text", "text": "What is in this picture?"},
    ]}],
)
```

Videos use `{"type": "video_url", "video_url": {"url": "..."}}`, or
`{"type": "video", "video": [frame URLs...], "fps": 2}` for a list of frames.
Animated GIF and WebP are decoded natively; other video formats (MP4, WebM,
MOV, ...) need `ffmpeg` and `ffprobe` on your PATH. As in the reference
processor, videos are sampled at 2 frames per second and each pair of frames
becomes one block of vision tokens labelled with its timestamp. Follow-up
turns reuse the already processed image instead of encoding it again.

Each 32x32 pixel area of an image becomes one prompt token, so image size
drives cost. Images are resized to at most `--max-image-pixels` (default
1 MP, about 1000 tokens); videos share `--max-video-pixels` (default 8 MP)
across at most `--max-video-frames` (default 64) frames. Raise these for fine
detail; the model itself accepts up to 16 MP per image. Clients cannot read
files on the server unless you pass `--allow-local-media`.

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
| `serve [--host] [--port] [--api-key] [--served-model-name] [--inline-reasoning] [--no-tui]` | OpenAI-compatible server with a live dashboard |
| `devices` | List usable GPUs |
| `bench` | Measure prompt processing and generation speed |

Options shared by `run`, `serve` and `bench`:

* `--model <file.snl or dir>` — model to load (defaults to the pulled model).
* `--quant q8_0|q4_0` — choose between converted variants.
* `--ctx <tokens>` — context size; the KV cache is allocated up front (default 32768).
* `--backend vulkan|dx12|metal|gl` and `--device <index>` — pick the GPU (see `devices`).
* `--no-vision` — skip loading the vision encoder (saves ~0.5 GB of GPU memory).
* `--no-mtp` — generate one token per step without speculative decoding
  (saves ~0.5 GB of GPU memory).
* `--max-image-pixels`, `--max-video-pixels`, `--max-video-frames` — media size limits.
* Sampling defaults: `--temperature 1.0 --top-p 0.95 --top-k 20` (from the
  model's generation config); `--max-tokens 8192`.

Models are stored in your local data directory
(`%LOCALAPPDATA%\sensenova-loop` on Windows, `~/.local/share/sensenova-loop`
on Linux, `~/Library/Application Support/sensenova-loop` on macOS), or wherever
`--models-dir` / `SENSENOVA_HOME` points.

## Performance

Measured on an RTX 5080 (16 GB, Vulkan, Windows 11):

| Weights | File | Prompt processing | Generation | Generation with MTP |
| --- | --- | --- | --- | --- |
| Q8_0 | 10.1 GB | ~380 tok/s | ~38 tok/s | ~55-60 tok/s |
| Q4_0 | 7.2 GB | ~420 tok/s | ~45 tok/s | ~64 tok/s |

With speculative decoding, the MTP head guesses the token after next; the main
model checks the guess while producing the next token, in one pass that costs
little more than a normal step. 75-95% of guesses are accepted in typical chat
and code. Accepted guesses are kept with exactly the probability the model
assigns them, so the output is distributed exactly as without MTP (and is
identical with `temperature` 0). If drafting stops paying off, for example
because GPU memory is full and the driver pages, it pauses itself.

The vision encoder takes about 0.4 s for an 800x600 image (475 tokens) and
about 1 s at the default 1 MP limit; reading the resulting image tokens into
the language model then runs at the prompt-processing speed above.

Because the looped block runs twice, every token executes 56 decoder layers,
so the model costs about as much per token as a ~14B dense model.

GPU memory needed is roughly the weight file plus ~60 KB per context token,
plus ~0.5 GB each for the vision encoder and the MTP head (about 12.5 GB for
Q8_0 at a 32K context). If generation is unexpectedly slow,
other applications may be holding GPU memory and forcing the driver to page
(the dashboard's VRAM bar turns red when the card is full):
lower `--ctx` or use `--quant q4_0`. On Windows, prefer the Vulkan backend
(the default); DX12 works but compiles shaders slowly and runs slower.

## Limitations

* Video files other than animated GIF/WebP need `ffmpeg` installed.
* One request is processed at a time; concurrent requests are queued.

## How it works

`src/engine.rs` records the whole forward pass as a fixed list of GPU
dispatches in execution-plan order. Shared weights are bound to both visits of
the looped block, while each visit has its own KV cache or recurrent state.
Kernels live in `src/shaders/`: quantized matrix-vector and tiled matrix
multiplication (`matmul.wgsl`), Gated DeltaNet (`linear.wgsl`), GQA attention
with split-K decoding (`attention.wgsl`), and norms / mHC routing / SwiGLU
(`layers.wgsl`). The vision tower (`src/vision.rs`, `vision.wgsl`) reuses the
quantized matmuls and adds LayerNorm, 2D RoPE and bidirectional attention
kernels; its output features replace the image placeholder tokens' embeddings,
and `src/mm.rs` handles preprocessing and the M-RoPE positions. The MTP head
is one more decoder layer with its own KV cache, fed the main model's final
hidden state and the next token's embedding. Verification runs two tokens
through the main model; the linear-attention layers checkpoint their recurrent
state after the first one so a rejected guess can be undone. The chat
template is rendered with MiniJinja, exactly as shipped with the model.
