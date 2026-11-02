mod chat;
mod config;
mod convert;
mod download;
mod engine;
mod gpu;
mod repl;
mod sampler;
mod server;
mod weights;
mod worker;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};

use crate::chat::{ChatTemplate, Tokenizer};
use crate::engine::Engine;
use crate::gpu::Gpu;
use crate::sampler::SamplingParams;
use crate::weights::DType;

const DEFAULT_REPO: &str = "sensenova/SenseNova-Loop-9B-Preview";

/// Run SenseNova Loop models locally on any GPU (Vulkan, DirectX 12 or Metal).
#[derive(Parser)]
#[command(version, about)]
struct Cli {
    /// Where models are stored [default: <local data dir>/sensenova-loop/models]
    #[arg(long, global = true, env = "SENSENOVA_HOME")]
    models_dir: Option<PathBuf>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Download a model from Hugging Face and quantize it for local use.
    Pull {
        #[arg(default_value = DEFAULT_REPO)]
        repo: String,
        /// Weight quantization: q8_0 (best quality, ~10 GB) or q4_0 (~6 GB).
        #[arg(long, default_value = "q8_0")]
        quant: String,
        #[arg(long, default_value = "main")]
        revision: String,
        /// Delete the original safetensors (~38 GB) after converting.
        #[arg(long)]
        delete_source: bool,
    },
    /// Quantize an already-downloaded Hugging Face checkpoint directory.
    Convert {
        /// Directory containing config.json, tokenizer.json and *.safetensors.
        source: PathBuf,
        #[arg(long, default_value = "q8_0")]
        quant: String,
        /// Output directory [default: the model directory under --models-dir].
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// List GPUs usable as --device.
    Devices {
        #[arg(long)]
        backend: Option<String>,
    },
    /// Chat in the terminal, or answer a single prompt with -p.
    Run {
        #[command(flatten)]
        model: ModelArgs,
        #[command(flatten)]
        sampling: GenArgs,
        /// Answer this prompt and exit.
        #[arg(short, long)]
        prompt: Option<String>,
        #[arg(long)]
        system: Option<String>,
        /// Hide the model's reasoning while it thinks.
        #[arg(long)]
        hide_reasoning: bool,
        /// Print token counts and speed after each answer.
        #[arg(long)]
        stats: bool,
    },
    /// Serve an OpenAI-compatible API (/v1/chat/completions, /v1/completions, /v1/models).
    Serve {
        #[command(flatten)]
        model: ModelArgs,
        #[command(flatten)]
        sampling: GenArgs,
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        #[arg(long, default_value_t = 8080)]
        port: u16,
        /// Require `Authorization: Bearer <key>` on API requests.
        #[arg(long, env = "SENSENOVA_API_KEY")]
        api_key: Option<String>,
        /// Model id reported by /v1/models (requests may use any model name).
        #[arg(long, default_value = "sensenova_loop")]
        served_model_name: String,
        /// Return reasoning inline as <think>...</think> in `content` instead of `reasoning_content`.
        #[arg(long)]
        inline_reasoning: bool,
    },
    /// Measure prompt processing and generation speed.
    Bench {
        #[command(flatten)]
        model: ModelArgs,
        #[arg(long, default_value_t = 512)]
        prompt_tokens: usize,
        #[arg(long, default_value_t = 128)]
        gen_tokens: usize,
    },
    /// Print the top next-token logits for a text (for checking against a reference implementation).
    #[command(hide = true)]
    Logits {
        #[command(flatten)]
        model: ModelArgs,
        text: String,
        /// Feed the prompt one token at a time instead of as a batch.
        #[arg(long)]
        stepwise: bool,
        #[arg(long, default_value_t = 10)]
        top: usize,
    },
}

#[derive(Args)]
struct ModelArgs {
    /// A .snl weight file or model directory [default: the pulled SenseNova-Loop-9B-Preview].
    #[arg(long, short)]
    model: Option<PathBuf>,
    /// Preferred quantization when several are present.
    #[arg(long)]
    quant: Option<String>,
    /// Context size in tokens (KV cache is allocated up front).
    #[arg(long, default_value_t = 32768)]
    ctx: usize,
    /// GPU backend: vulkan, dx12, metal or gl [default: best available].
    #[arg(long, env = "SENSENOVA_BACKEND")]
    backend: Option<String>,
    /// GPU index from `devices`.
    #[arg(long, env = "SENSENOVA_DEVICE")]
    device: Option<usize>,
}

#[derive(Args)]
struct GenArgs {
    #[arg(long, default_value_t = 1.0)]
    temperature: f32,
    #[arg(long, default_value_t = 0.95)]
    top_p: f32,
    #[arg(long, default_value_t = 20)]
    top_k: usize,
    #[arg(long, default_value_t = 0.0)]
    min_p: f32,
    #[arg(long, default_value_t = 0.0)]
    presence_penalty: f32,
    #[arg(long, default_value_t = 1.0)]
    repetition_penalty: f32,
    #[arg(long)]
    seed: Option<u64>,
    /// Default maximum tokens to generate per answer.
    #[arg(long, default_value_t = 8192)]
    max_tokens: usize,
    /// Disable thinking mode by default (clients can still enable it per request).
    #[arg(long)]
    no_think: bool,
}

impl GenArgs {
    fn params(&self) -> SamplingParams {
        SamplingParams {
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            min_p: self.min_p,
            presence_penalty: self.presence_penalty,
            frequency_penalty: 0.0,
            repetition_penalty: self.repetition_penalty,
            seed: self.seed,
        }
    }
}

fn models_dir(cli: &Cli) -> PathBuf {
    cli.models_dir
        .clone()
        .unwrap_or_else(|| dirs::data_local_dir().unwrap_or_else(|| PathBuf::from(".")).join("sensenova-loop").join("models"))
}

fn repo_dir(root: &Path, repo: &str) -> PathBuf {
    repo.split('/').fold(root.to_path_buf(), |p, s| p.join(s))
}

fn snl_name(quant: DType) -> String {
    format!("model-{}.snl", quant.name())
}

/// Copy tokenizer/template/config next to the converted weights so the source can be deleted.
fn convert_into(source: &Path, out_dir: &Path, quant: DType, name: &str) -> Result<PathBuf> {
    std::fs::create_dir_all(out_dir)?;
    for f in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "config.json", "generation_config.json"] {
        let src = source.join(f);
        if src.exists() {
            std::fs::copy(&src, out_dir.join(f))?;
        }
    }
    let dst = out_dir.join(snl_name(quant));
    println!("Converting to {} ...", dst.display());
    convert::convert(source, &dst, quant, name)?;
    Ok(dst)
}

struct Loaded {
    engine: Engine,
    tok: Arc<Tokenizer>,
    template: ChatTemplate,
}

fn load(cli: &Cli, m: &ModelArgs) -> Result<Loaded> {
    let path = match &m.model {
        Some(p) => p.clone(),
        None => repo_dir(&models_dir(cli), DEFAULT_REPO),
    };
    let file = if path.is_dir() {
        let mut order = vec![DType::Q8_0, DType::Q4_0];
        if let Some(q) = &m.quant {
            order.insert(0, DType::parse(q)?);
        }
        order
            .iter()
            .map(|q| path.join(snl_name(*q)))
            .find(|p| p.exists())
            .with_context(|| format!("no converted model in {}; run `sensenova-loop pull` first", path.display()))?
    } else if path.exists() {
        path.clone()
    } else {
        bail!("model not found: {} (run `sensenova-loop pull` first)", path.display());
    };
    let dir = file.parent().unwrap_or(Path::new("."));
    let tok = Arc::new(Tokenizer::load(&dir.join("tokenizer.json"))?);
    let template = ChatTemplate::load(&dir.join("chat_template.jinja"))?;

    let gpu = Gpu::new(m.backend.as_deref(), m.device)?;
    eprintln!("GPU: {} ({:?}, {:?})", gpu.info.name, gpu.info.backend, gpu.info.device_type);
    let t = Instant::now();
    let engine = Engine::load(gpu, &file, m.ctx)?;
    eprintln!(
        "Loaded {} [{}] in {:.1}s, context {} tokens, {:.1} GB of GPU memory",
        file.display(),
        engine.quant,
        t.elapsed().as_secs_f64(),
        engine.ctx,
        engine.gpu.allocated.get() as f64 / 1e9
    );
    Ok(Loaded { engine, tok, template })
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match &cli.command {
        Command::Pull { repo, quant, revision, delete_source } => {
            let quant = DType::parse(quant)?;
            let dir = repo_dir(&models_dir(&cli), repo);
            let source = dir.join("source");
            download::download(repo, revision, &source)?;
            let dst = convert_into(&source, &dir, quant, repo)?;
            if *delete_source {
                for e in std::fs::read_dir(&source)? {
                    let p = e?.path();
                    if p.extension().is_some_and(|x| x == "safetensors") {
                        std::fs::remove_file(p)?;
                    }
                }
            }
            println!("Ready: {}", dst.display());
        }
        Command::Convert { source, quant, out } => {
            let quant = DType::parse(quant)?;
            let out = out.clone().unwrap_or_else(|| repo_dir(&models_dir(&cli), DEFAULT_REPO));
            let name = source.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
            let dst = convert_into(source, &out, quant, &name)?;
            println!("Ready: {}", dst.display());
        }
        Command::Devices { backend } => {
            for (i, a) in gpu::list_adapters(backend.as_deref())?.iter().enumerate() {
                println!("[{i}] {} ({:?}, {:?}) driver: {} {}", a.name, a.backend, a.device_type, a.driver, a.driver_info);
            }
        }
        Command::Run { model, sampling, prompt, system, hide_reasoning, stats } => {
            let l = load(&cli, model)?;
            let tx = worker::spawn(l.engine, l.tok.clone());
            repl::run(
                tx,
                l.tok,
                &l.template,
                repl::ChatOptions {
                    system: system.clone(),
                    thinking: !sampling.no_think,
                    show_reasoning: !hide_reasoning,
                    params: sampling.params(),
                    max_tokens: sampling.max_tokens,
                    stats: *stats,
                },
                prompt.clone(),
            )?;
        }
        Command::Serve { model, sampling, host, port, api_key, served_model_name, inline_reasoning } => {
            let l = load(&cli, model)?;
            let ctx = l.engine.ctx;
            let im_start = l.tok.token_id("<|im_start|>");
            let tx = worker::spawn(l.engine, l.tok.clone());
            let state = Arc::new(server::AppState {
                cfg: server::ServerConfig {
                    model_name: served_model_name.clone(),
                    api_key: api_key.clone(),
                    defaults: sampling.params(),
                    max_tokens: sampling.max_tokens,
                    thinking: !sampling.no_think,
                    separate_reasoning: !inline_reasoning,
                },
                tx: std::sync::Mutex::new(tx),
                tok: l.tok,
                template: l.template,
                im_start,
                ctx,
            });
            let addr = format!("{host}:{port}");
            tokio::runtime::Runtime::new()?.block_on(async move {
                let listener = tokio::net::TcpListener::bind(&addr).await.with_context(|| format!("binding {addr}"))?;
                eprintln!("OpenAI-compatible API on http://{addr}/v1 (model id `{served_model_name}`)");
                axum::serve(listener, server::router(state))
                    .with_graceful_shutdown(async {
                        let _ = tokio::signal::ctrl_c().await;
                    })
                    .await?;
                anyhow::Ok(())
            })?;
        }
        Command::Logits { model, text, stepwise, top } => {
            let mut l = load(&cli, model)?;
            let ids = l.tok.encode(text)?;
            println!("tokens: {ids:?}");
            let logits = if *stepwise {
                let mut last = None;
                for &t in &ids {
                    last = l.engine.forward(&[t], true)?;
                }
                last.unwrap()
            } else {
                l.engine.forward(&ids, true)?.unwrap()
            };
            let mut idx: Vec<usize> = (0..logits.len()).collect();
            idx.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
            for &i in idx.iter().take(*top) {
                println!("{i:>8} {:>10.4} {:?}", logits[i], l.tok.decode(&[i as u32]));
            }
        }
        Command::Bench { model, prompt_tokens, gen_tokens } => {
            let mut l = load(&cli, model)?;
            let e = &mut l.engine;
            let prompt: Vec<u32> = (0..*prompt_tokens).map(|i| 1000 + (i as u32 * 7919) % 50000).collect();
            let _ = e.forward(&prompt[..prompt.len().min(16)], true)?; // warm up pipelines
            e.reset();
            let t = Instant::now();
            let mut logits = e.forward(&prompt, true)?.unwrap();
            let pp = t.elapsed().as_secs_f64();
            let t = Instant::now();
            for _ in 0..*gen_tokens {
                let tok = sampler::argmax(&logits);
                logits = e.forward(&[tok], true)?.unwrap();
            }
            let tg = t.elapsed().as_secs_f64();
            println!("prompt processing: {:.1} tok/s ({} tokens in {:.2}s)", *prompt_tokens as f64 / pp, prompt_tokens, pp);
            println!("generation:        {:.1} tok/s ({} tokens in {:.2}s)", *gen_tokens as f64 / tg, gen_tokens, tg);
            e.print_profile();
        }
    }
    Ok(())
}
