//! Resumable download of a Hugging Face model repository.

use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::Path;

use anyhow::{Context, Result, bail};
use indicatif::{ProgressBar, ProgressStyle};
use serde_json::Value;

const ENDPOINT: &str = "https://huggingface.co";

fn wanted(path: &str) -> bool {
    matches!(
        path,
        "config.json"
            | "generation_config.json"
            | "tokenizer.json"
            | "tokenizer_config.json"
            | "chat_template.jinja"
            | "model.safetensors.index.json"
    ) || (path.starts_with("model") && path.ends_with(".safetensors") && !path.contains('/'))
}

fn client() -> Result<reqwest::blocking::Client> {
    let mut headers = reqwest::header::HeaderMap::new();
    if let Ok(token) = std::env::var("HF_TOKEN") {
        headers.insert(reqwest::header::AUTHORIZATION, format!("Bearer {token}").parse()?);
    }
    Ok(reqwest::blocking::Client::builder()
        .default_headers(headers)
        .user_agent(concat!("sensenova-loop/", env!("CARGO_PKG_VERSION")))
        .timeout(None)
        .build()?)
}

/// Download the files needed for conversion from `repo` into `dir`, skipping complete files.
pub fn download(repo: &str, revision: &str, dir: &Path) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let http = client()?;
    let tree: Vec<Value> = http
        .get(format!("{ENDPOINT}/api/models/{repo}/tree/{revision}"))
        .send()?
        .error_for_status()
        .with_context(|| format!("listing files of {repo}"))?
        .json()?;
    let files: Vec<(String, u64)> = tree
        .iter()
        .filter(|e| e["type"] == "file")
        .filter_map(|e| Some((e["path"].as_str()?.to_string(), e["lfs"]["size"].as_u64().or(e["size"].as_u64())?)))
        .filter(|(p, _)| wanted(p))
        .collect();
    if !files.iter().any(|(p, _)| p.ends_with(".safetensors")) {
        bail!("{repo} has no safetensors weights");
    }
    let total: u64 = files.iter().map(|(_, s)| s).sum();
    println!("Downloading {repo} ({:.1} GB) to {}", total as f64 / 1e9, dir.display());
    for (path, size) in files {
        let dst = dir.join(&path);
        if dst.metadata().ok().map(|m| m.len()) == Some(size) {
            continue;
        }
        fetch(&http, &format!("{ENDPOINT}/{repo}/resolve/{revision}/{path}"), &dst, size)
            .with_context(|| format!("downloading {path}"))?;
    }
    Ok(())
}

fn fetch(http: &reqwest::blocking::Client, url: &str, dst: &Path, size: u64) -> Result<()> {
    let part = dst.with_extension(format!(
        "{}.part",
        dst.extension().and_then(|e| e.to_str()).unwrap_or("")
    ));
    let mut have = part.metadata().map(|m| m.len()).unwrap_or(0);
    if have > size {
        std::fs::remove_file(&part)?;
        have = 0;
    }
    let bar = ProgressBar::new(size);
    bar.set_style(
        ProgressStyle::with_template("{msg:32!} {bar:30} {bytes}/{total_bytes} {bytes_per_sec} eta {eta}").unwrap(),
    );
    bar.set_message(dst.file_name().unwrap().to_string_lossy().to_string());
    bar.set_position(have);

    let mut req = http.get(url);
    if have > 0 {
        req = req.header(reqwest::header::RANGE, format!("bytes={have}-"));
    }
    let mut resp = req.send()?.error_for_status()?;
    let mut out: File = if have > 0 && resp.status() == reqwest::StatusCode::PARTIAL_CONTENT {
        OpenOptions::new().append(true).open(&part)?
    } else {
        bar.set_position(0);
        File::create(&part)?
    };
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = resp.read(&mut buf)?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n])?;
        bar.inc(n as u64);
    }
    out.flush()?;
    drop(out);
    let got = part.metadata()?.len();
    if got != size {
        bail!("incomplete download ({got} of {size} bytes); run the command again to resume");
    }
    std::fs::rename(&part, dst)?;
    bar.finish();
    Ok(())
}
