//! Interactive terminal chat.

use std::io::{BufRead, Write};
use std::sync::Arc;

use anyhow::{Result, bail};
use serde_json::{Value, json};
use tokio::sync::mpsc::unbounded_channel;

use crate::chat::{ChatTemplate, OutputParser, Piece, Tokenizer};
use crate::sampler::SamplingParams;
use crate::worker::{GenEvent, GenRequest};

pub struct ChatOptions {
    pub system: Option<String>,
    pub thinking: bool,
    pub show_reasoning: bool,
    pub params: SamplingParams,
    pub max_tokens: usize,
    pub stats: bool,
}

const DIM: &str = "\x1b[2m";
const RESET: &str = "\x1b[0m";

pub fn run(
    tx: std::sync::mpsc::Sender<GenRequest>,
    tok: Arc<Tokenizer>,
    template: &ChatTemplate,
    mut opts: ChatOptions,
    prompt: Option<String>,
) -> Result<()> {
    let im_start = tok.token_id("<|im_start|>");
    let mut history: Vec<Value> = Vec::new();
    let one_shot = prompt.is_some();
    if !one_shot {
        eprintln!("Type a message. Commands: /think, /nothink, /reset, /system <text>, /exit");
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
            ("/reset", _) => {
                history.clear();
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
                eprintln!("(system prompt set, conversation cleared)");
                continue;
            }
            (cmd, _) if cmd.starts_with('/') => {
                eprintln!("unknown command {cmd}");
                continue;
            }
            _ => {}
        }

        history.push(json!({"role": "user", "content": input}));
        let mut messages = Vec::new();
        if let Some(s) = &opts.system {
            messages.push(json!({"role": "system", "content": s}));
        }
        messages.extend(history.iter().cloned());
        let text = template.render(&messages, None, opts.thinking)?;
        let tokens = tok.encode(&text)?;
        let (etx, mut erx) = unbounded_channel();
        tx.send(GenRequest {
            snapshot_at: im_start.and_then(|id| tokens.iter().rposition(|&t| t == id)),
            prompt: tokens,
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
                        eprintln!(
                            "{DIM}[prompt {} tok, {} cached, {:.0} tok/s | {} tok at {:.1} tok/s]{RESET}",
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
