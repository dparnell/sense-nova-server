//! Interactive terminal chat.

use std::io::{BufRead, Write};
use std::sync::Arc;

use anyhow::{Result, bail};
use serde_json::{Value, json};
use tokio::sync::mpsc::unbounded_channel;

use crate::chat::{ChatTemplate, OutputParser, Piece, Tokenizer};
use crate::config::VisionConfig;
use crate::mm::{self, Limits, Media, MediaRef, Prompt};
use crate::sampler::SamplingParams;
use crate::worker::{GenEvent, GenRequest};

pub struct ChatOptions {
    pub system: Option<String>,
    pub thinking: bool,
    pub show_reasoning: bool,
    pub params: SamplingParams,
    pub max_tokens: usize,
    pub stats: bool,
    pub vision: Option<VisionConfig>,
    pub limits: Limits,
}

const DIM: &str = "\x1b[2m";
const RESET: &str = "\x1b[0m";

pub fn run(
    tx: std::sync::mpsc::Sender<GenRequest>,
    tok: Arc<Tokenizer>,
    template: &ChatTemplate,
    mut opts: ChatOptions,
    prompt: Option<String>,
    attach: Vec<MediaRef>,
) -> Result<()> {
    let im_start = tok.token_id("<|im_start|>");
    let mut history: Vec<Value> = Vec::new();
    // Media of every message in `history`, in order, and media waiting for the next message.
    let mut history_media: Vec<Arc<Media>> = Vec::new();
    let mut pending_media: Vec<Arc<Media>> = Vec::new();
    let attach_media = |refs: &[MediaRef], pending: &mut Vec<Arc<Media>>| -> Result<()> {
        if opts.vision.is_none() {
            bail!("image and video inputs need the vision encoder (run `sensenova-loop pull` to add it, and do not pass --no-vision)");
        }
        for m in mm::load_media(refs, &opts.limits)? {
            eprintln!(
                "(attached {}: {} tokens)",
                if m.kind == mm::MediaKind::Image { "image".to_string() } else { format!("video, {} frame pairs", m.frames) },
                m.tokens()
            );
            pending.push(m);
        }
        Ok(())
    };
    attach_media(&attach, &mut pending_media)?;
    let one_shot = prompt.is_some();
    if !one_shot {
        eprintln!("Type a message. Commands: /image <path or URL>, /video <path or URL>, /think, /nothink, /reset, /system <text>, /exit");
    }
    let stdin = std::io::stdin();
    let mut lines = stdin.lock().lines();
    let mut pending = prompt;
    loop {
        let input = match pending.take() {
            Some(p) => p,
            None => {
                if one_shot {
                    return Ok(());
                }
                print!("\n> ");
                std::io::stdout().flush()?;
                match lines.next() {
                    Some(l) => l?,
                    None => return Ok(()),
                }
            }
        };
        let input = input.trim().to_string();
        if input.is_empty() {
            continue;
        }
        match input.split_once(' ').map_or((input.as_str(), ""), |(a, b)| (a, b)) {
            ("/exit" | "/quit", _) => return Ok(()),
            ("/image" | "/video", "") => {
                eprintln!("usage: {} <path or URL>", input);
                continue;
            }
            (cmd @ ("/image" | "/video"), src) => {
                let r = if cmd == "/image" { MediaRef::Image(src.trim().into()) } else { MediaRef::Video(src.trim().into()) };
                if let Err(e) = attach_media(&[r], &mut pending_media) {
                    eprintln!("{e:#}");
                }
                continue;
            }
            ("/reset", _) => {
                history.clear();
                history_media.clear();
                pending_media.clear();
                eprintln!("(conversation cleared)");
                continue;
            }
            ("/think", _) => {
                opts.thinking = true;
                eprintln!("(thinking on)");
                continue;
            }
            ("/nothink", _) => {
                opts.thinking = false;
                eprintln!("(thinking off)");
                continue;
            }
            ("/system", text) => {
                opts.system = (!text.is_empty()).then(|| text.to_string());
                history.clear();
                history_media.clear();
                eprintln!("(system prompt set, conversation cleared)");
                continue;
            }
            (cmd, _) if cmd.starts_with('/') => {
                eprintln!("unknown command {cmd}");
                continue;
            }
            _ => {}
        }

        if pending_media.is_empty() {
            history.push(json!({"role": "user", "content": input}));
        } else {
            let mut parts: Vec<Value> = pending_media
                .iter()
                .map(|m| json!({"type": if m.kind == mm::MediaKind::Image { "image" } else { "video" }}))
                .collect();
            parts.push(json!({"type": "text", "text": input}));
            history.push(json!({"role": "user", "content": parts}));
            history_media.append(&mut pending_media);
        }
        let mut messages = Vec::new();
        if let Some(s) = &opts.system {
            messages.push(json!({"role": "system", "content": s}));
        }
        messages.extend(history.iter().cloned());
        let text = template.render(&messages, None, opts.thinking)?;
        let prompt = match &opts.vision {
            Some(vc) if !history_media.is_empty() => mm::build_prompt(&tok, vc, &text, history_media.clone())?,
            _ => Prompt::text(tok.encode(&text)?),
        };
        let (etx, mut erx) = unbounded_channel();
        tx.send(GenRequest {
            snapshot_at: im_start.and_then(|id| prompt.tokens.iter().rposition(|&t| t == id)),
            prompt,
            params: opts.params.clone(),
            max_tokens: opts.max_tokens,
            stop: Vec::new(),
            events: etx,
        })
        .map_err(|_| anyhow::anyhow!("generation thread stopped"))?;

        let mut parser = OutputParser::new(opts.thinking, true, false);
        let mut answer = String::new();
        let mut in_reasoning = false;
        let mut out = std::io::stdout();
        let mut print = |pieces: Vec<Piece>, answer: &mut String| -> Result<()> {
            for p in pieces {
                match p {
                    Piece::Reasoning(s) => {
                        if !opts.show_reasoning {
                            continue;
                        }
                        if !in_reasoning {
                            write!(out, "{DIM}")?;
                            in_reasoning = true;
                        }
                        write!(out, "{s}")?;
                    }
                    Piece::Content(s) => {
                        if in_reasoning {
                            write!(out, "{RESET}\n\n")?;
                            in_reasoning = false;
                        }
                        write!(out, "{s}")?;
                        answer.push_str(&s);
                    }
                }
            }
            out.flush()?;
            Ok(())
        };
        loop {
            match erx.blocking_recv() {
                Some(GenEvent::Text(t)) => print(parser.push(&t), &mut answer)?,
                Some(GenEvent::Error(e)) => bail!("{e}"),
                Some(GenEvent::Done(f)) => {
                    print(parser.finish(None).0, &mut answer)?;
                    if in_reasoning {
                        print!("{RESET}");
                    }
                    println!();
                    if opts.stats {
                        let spec = if f.drafted > 0 {
                            format!(", {:.0}% of drafts accepted", f.accepted as f64 / f.drafted as f64 * 100.0)
                        } else {
                            String::new()
                        };
                        eprintln!(
                            "{DIM}[prompt {} tok, {} cached, {:.0} tok/s | {} tok at {:.1} tok/s{spec}]{RESET}",
                            f.prompt_tokens,
                            f.cached_tokens,
                            (f.prompt_tokens - f.cached_tokens) as f64 / f.prefill_secs.max(1e-9),
                            f.completion_tokens,
                            f.completion_tokens as f64 / f.decode_secs.max(1e-9)
                        );
                    }
                    break;
                }
                None => bail!("generation thread stopped"),
            }
        }
        history.push(json!({"role": "assistant", "content": answer}));
    }
}
