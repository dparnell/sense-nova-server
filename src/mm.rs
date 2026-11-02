//! Image and video inputs: fetching, Qwen3-VL preprocessing (resize, normalize,
//! patchify), and turning a rendered prompt with media placeholders into
//! language-model tokens, prefix-cache keys and M-RoPE positions.

use std::hash::{Hash, Hasher};
use std::io::{Cursor, Read};
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use base64::Engine as _;
use image::{AnimationDecoder, DynamicImage, ImageFormat, RgbImage, RgbaImage};
use serde_json::Value;

use crate::chat::Tokenizer;
use crate::config::VisionConfig;

// The vision tower is validated to use these (see `VisionConfig` checks).
const PATCH: usize = 16;
const MERGE: usize = 2;
const TEMPORAL: usize = 2;
const FACTOR: usize = PATCH * MERGE;
/// Processor limits from the model's preprocessor configs.
const IMAGE_MIN_PIXELS: usize = 65_536;
const IMAGE_MAX_PIXELS: usize = 16_777_216;
const VIDEO_MIN_PIXELS: usize = 4_096;
const VIDEO_MAX_PIXELS: usize = 25_165_824;
const VIDEO_FPS: f64 = 2.0;
const VIDEO_MIN_FRAMES: usize = 4;
const VIDEO_MAX_FRAMES: usize = 768;
/// Largest media download accepted.
const MAX_FETCH_BYTES: u64 = 256 << 20;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MediaKind {
    Image,
    Video,
}

/// A preprocessed image or video, ready for the vision tower.
pub struct Media {
    pub kind: MediaKind,
    /// Temporal patches (1 for an image; frame pairs for a video).
    pub frames: usize,
    /// Grid size in patches.
    pub h: usize,
    pub w: usize,
    /// frames * h * w patch vectors of 3 * 2 * 16 * 16 values, in merge order.
    pub patches: Vec<f32>,
    /// Seconds at the middle of each temporal patch (videos).
    pub timestamps: Vec<f64>,
    /// Content hash, used to tell different media apart in the prefix cache.
    pub hash: u64,
}

impl Media {
    pub fn tokens_per_frame(&self) -> usize {
        self.h * self.w / (MERGE * MERGE)
    }
    pub fn tokens(&self) -> usize {
        self.frames * self.tokens_per_frame()
    }
}

/// Caps on how many pixels media may be resized to; they bound vision and prompt cost.
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    pub max_image_pixels: usize,
    /// Total over all sampled frames of a video.
    pub max_video_pixels: usize,
    pub max_frames: usize,
    /// Accept file paths and file:// URLs (never for network clients).
    pub allow_local_files: bool,
}

/// Python's `round` (ties to even).
fn round_py(x: f64) -> f64 {
    x.round_ties_even()
}

/// Qwen2-VL image `smart_resize`: dims divisible by 32, pixel count within bounds.
pub fn smart_resize_image(h: usize, w: usize, min_px: usize, max_px: usize) -> Result<(usize, usize)> {
    let (hf, wf) = (h as f64, w as f64);
    if hf.max(wf) / hf.min(wf) > 200.0 {
        bail!("image aspect ratio is more extreme than 200:1");
    }
    let f = FACTOR as f64;
    let mut hb = round_py(hf / f) * f;
    let mut wb = round_py(wf / f) * f;
    if hb * wb > max_px as f64 {
        let beta = (hf * wf / max_px as f64).sqrt();
        hb = f.max((hf / beta / f).floor() * f);
        wb = f.max((wf / beta / f).floor() * f);
    } else if hb * wb < min_px as f64 {
        let beta = (min_px as f64 / (hf * wf)).sqrt();
        hb = (hf * beta / f).ceil() * f;
        wb = (wf * beta / f).ceil() * f;
    }
    Ok((hb as usize, wb as usize))
}

/// Qwen3-VL video `smart_resize`; the pixel budget covers all frames.
pub fn smart_resize_video(frames: usize, h: usize, w: usize, min_px: usize, max_px: usize) -> Result<(usize, usize)> {
    let (hf, wf, tf) = (h.max(FACTOR) as f64, w.max(FACTOR) as f64, frames as f64);
    if hf.max(wf) / hf.min(wf) > 200.0 {
        bail!("video aspect ratio is more extreme than 200:1");
    }
    let f = FACTOR as f64;
    let mut hb = round_py(hf / f) * f;
    let mut wb = round_py(wf / f) * f;
    let tb = (tf / TEMPORAL as f64).ceil() * TEMPORAL as f64;
    if tb * hb * wb > max_px as f64 {
        let beta = (tf * hf * wf / max_px as f64).sqrt();
        hb = f.max((hf / beta / f).floor() * f);
        wb = f.max((wf / beta / f).floor() * f);
    } else if tb * hb * wb < min_px as f64 {
        let beta = (min_px as f64 / (tf * hf * wf)).sqrt();
        hb = (hf * beta / f).ceil() * f;
        wb = (wf * beta / f).ceil() * f;
    }
    Ok((hb as usize, wb as usize))
}

/// Flatten alpha onto white, as the HF processor's `convert_rgb` does.
fn to_rgb(img: DynamicImage) -> RgbImage {
    if !img.color().has_alpha() {
        return img.into_rgb8();
    }
    let rgba: RgbaImage = img.into_rgba8();
    let mut out = RgbImage::new(rgba.width(), rgba.height());
    for (o, p) in out.pixels_mut().zip(rgba.pixels()) {
        let a = p[3] as f32 / 255.0;
        for c in 0..3 {
            o[c] = (p[c] as f32 * a + 255.0 * (1.0 - a)).round() as u8;
        }
    }
    out
}

fn resize(img: &RgbImage, w: usize, h: usize) -> RgbImage {
    if img.width() as usize == w && img.height() as usize == h {
        return img.clone();
    }
    image::imageops::resize(img, w as u32, h as u32, image::imageops::FilterType::CatmullRom)
}

/// Patch vectors (c, t, y, x) in merge order: 2x2 blocks of patches, row-major over blocks.
fn patchify(frames: &[RgbImage]) -> Vec<f32> {
    let (wpx, hpx) = (frames[0].width() as usize, frames[0].height() as usize);
    let (gh, gw) = (hpx / PATCH, wpx / PATCH);
    let mut out = Vec::with_capacity(frames.len() * hpx * wpx * 3);
    for pair in frames.chunks_exact(TEMPORAL) {
        for br in 0..gh / MERGE {
            for bc in 0..gw / MERGE {
                for i in 0..MERGE {
                    for j in 0..MERGE {
                        let (row, col) = (br * MERGE + i, bc * MERGE + j);
                        for c in 0..3 {
                            for f in pair {
                                for y in 0..PATCH {
                                    for x in 0..PATCH {
                                        let v = f.get_pixel((col * PATCH + x) as u32, (row * PATCH + y) as u32)[c];
                                        // rescale 1/255 then normalize with mean 0.5, std 0.5.
                                        out.push(v as f32 / 127.5 - 1.0);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

fn content_hash(parts: &[&[u8]], dims: &[usize]) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    for p in parts {
        p.hash(&mut h);
    }
    dims.hash(&mut h);
    h.finish()
}

pub fn image_from_bytes(bytes: &[u8], limits: &Limits) -> Result<Media> {
    let img = image::load_from_memory(bytes).context("decoding image")?;
    let rgb = to_rgb(img);
    let max_px = limits.max_image_pixels.clamp(IMAGE_MIN_PIXELS, IMAGE_MAX_PIXELS);
    let (h, w) = smart_resize_image(rgb.height() as usize, rgb.width() as usize, IMAGE_MIN_PIXELS, max_px)?;
    let r = resize(&rgb, w, h);
    // A still image fills both frames of the temporal patch.
    let patches = patchify(&[r.clone(), r]);
    Ok(Media {
        kind: MediaKind::Image,
        frames: 1,
        h: h / PATCH,
        w: w / PATCH,
        patches,
        timestamps: Vec::new(),
        hash: content_hash(&[bytes], &[h, w]),
    })
}

/// Frame indices sampled at 2 fps, as the Qwen3-VL video processor does.
fn sample_indices(total: usize, fps: f64, max_frames: usize) -> Vec<usize> {
    let want = (total as f64 / fps * VIDEO_FPS) as usize;
    let n = want.max(VIDEO_MIN_FRAMES).min(max_frames.clamp(1, VIDEO_MAX_FRAMES)).min(total).max(1);
    // np.linspace(0, total - 1, n).round()
    (0..n)
        .map(|i| if n == 1 { 0 } else { round_py((total - 1) as f64 * i as f64 / (n - 1) as f64) as usize })
        .collect()
}

/// Build a video from already sampled frames at original indices `indices` of a `fps` stream.
fn video_from_samples(frames: Vec<RgbImage>, mut indices: Vec<usize>, fps: f64, limits: &Limits, hash: u64) -> Result<Media> {
    if frames.is_empty() {
        bail!("video has no frames");
    }
    let n = frames.len();
    let max_px = limits.max_video_pixels.clamp(VIDEO_MIN_PIXELS, VIDEO_MAX_PIXELS);
    let (h, w) = smart_resize_video(n, frames[0].height() as usize, frames[0].width() as usize, VIDEO_MIN_PIXELS, max_px)?;
    let mut resized: Vec<RgbImage> = frames.iter().map(|f| resize(f, w, h)).collect();
    while resized.len() % TEMPORAL != 0 {
        resized.push(resized.last().unwrap().clone());
    }
    while indices.len() % TEMPORAL != 0 {
        indices.push(*indices.last().unwrap());
    }
    let timestamps = indices.chunks(TEMPORAL).map(|p| (p[0] as f64 + p[TEMPORAL - 1] as f64) / fps / 2.0).collect();
    Ok(Media {
        kind: MediaKind::Video,
        frames: resized.len() / TEMPORAL,
        h: h / PATCH,
        w: w / PATCH,
        patches: patchify(&resized),
        timestamps,
        hash: content_hash(&[&hash.to_le_bytes()], &[h, w, n]),
    })
}

/// A video from a list of still frames taken `fps` times per second (all frames are used).
pub fn video_from_frame_images(images: &[Vec<u8>], fps: f64, limits: &Limits) -> Result<Media> {
    let mut frames = Vec::new();
    for b in images {
        frames.push(to_rgb(image::load_from_memory(b).context("decoding video frame")?));
    }
    // Uniformly thin out very long frame lists.
    let keep = sample_indices(frames.len(), VIDEO_FPS, limits.max_frames.min(frames.len()));
    let keep = if frames.len() <= limits.max_frames { (0..frames.len()).collect() } else { keep };
    let frames = keep.iter().map(|&i| frames[i].clone()).collect();
    let parts: Vec<&[u8]> = images.iter().map(|b| b.as_slice()).collect();
    video_from_samples(frames, keep, fps, limits, content_hash(&parts, &[]))
}

/// Decode a video file: animated GIF/WebP natively, anything else through `ffmpeg`.
pub fn video_from_bytes(bytes: &[u8], limits: &Limits) -> Result<Media> {
    let anim = match image::guess_format(bytes) {
        Ok(ImageFormat::Gif) => Some(
            image::codecs::gif::GifDecoder::new(Cursor::new(bytes))?.into_frames().collect_frames().context("decoding GIF")?,
        ),
        Ok(ImageFormat::WebP) => {
            let d = image::codecs::webp::WebPDecoder::new(Cursor::new(bytes))?;
            if d.has_animation() { Some(d.into_frames().collect_frames().context("decoding WebP")?) } else { None }
        }
        _ => None,
    };
    if let Some(frames) = anim {
        let secs: f64 = frames.iter().map(|f| std::time::Duration::from(f.delay()).as_secs_f64()).sum();
        let fps = if secs > 0.0 { frames.len() as f64 / secs } else { 10.0 };
        let idx = sample_indices(frames.len(), fps, limits.max_frames);
        let sampled = idx.iter().map(|&i| to_rgb(DynamicImage::ImageRgba8(frames[i].buffer().clone()))).collect();
        return video_from_samples(sampled, idx, fps, limits, content_hash(&[bytes], &[]));
    }
    ffmpeg_video(bytes, limits)
}

fn ffmpeg_video(bytes: &[u8], limits: &Limits) -> Result<Media> {
    use std::process::{Command, Stdio};
    let tmp = std::env::temp_dir().join(format!(
        "sensenova-video-{}-{:x}",
        std::process::id(),
        content_hash(&[bytes], &[])
    ));
    std::fs::write(&tmp, bytes)?;
    struct Cleanup(std::path::PathBuf);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }
    let _cleanup = Cleanup(tmp.clone());

    let probe = Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", "v:0", "-show_entries"])
        .arg("stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration")
        .args(["-of", "json"])
        .arg(&tmp)
        .output()
        .map_err(|_| anyhow::anyhow!("video input needs ffmpeg and ffprobe on PATH (GIF and WebP work without them)"))?;
    if !probe.status.success() {
        bail!("ffprobe could not read the video: {}", String::from_utf8_lossy(&probe.stderr).trim());
    }
    let info: serde_json::Value = serde_json::from_slice(&probe.stdout)?;
    let s = &info["streams"][0];
    let num = |k: &str| s[k].as_str().and_then(|v| v.parse::<f64>().ok()).or_else(|| s[k].as_f64());
    let rate = |k: &str| {
        s[k].as_str().and_then(|r| {
            let (a, b) = r.split_once('/')?;
            let (a, b) = (a.parse::<f64>().ok()?, b.parse::<f64>().ok()?);
            (a > 0.0 && b > 0.0).then_some(a / b)
        })
    };
    let (w, h) = (s["width"].as_u64().unwrap_or(0) as usize, s["height"].as_u64().unwrap_or(0) as usize);
    if w == 0 || h == 0 {
        bail!("ffprobe found no video stream");
    }
    let fps = rate("avg_frame_rate").or_else(|| rate("r_frame_rate")).unwrap_or(24.0);
    let total = num("nb_frames").map(|n| n as usize).or_else(|| num("duration").map(|d| (d * fps) as usize)).unwrap_or(0);
    if total == 0 {
        bail!("could not determine the video's length");
    }
    let indices = sample_indices(total, fps, limits.max_frames);
    let max_px = limits.max_video_pixels.clamp(VIDEO_MIN_PIXELS, VIDEO_MAX_PIXELS);
    let (rh, rw) = smart_resize_video(indices.len(), h, w, VIDEO_MIN_PIXELS, max_px)?;
    let select: Vec<String> = indices.iter().map(|i| format!("eq(n\\,{i})")).collect();
    let filter = format!("select='{}',scale={rw}:{rh}:flags=bicubic", select.join("+"));
    let mut child = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(&tmp)
        .args(["-vf", &filter, "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("starting ffmpeg")?;
    let mut raw = Vec::new();
    child.stdout.take().unwrap().read_to_end(&mut raw)?;
    let out = child.wait_with_output()?;
    if !out.status.success() {
        bail!("ffmpeg failed: {}", String::from_utf8_lossy(&out.stderr).trim());
    }
    let frame_bytes = rw * rh * 3;
    let frames: Vec<RgbImage> = raw
        .chunks_exact(frame_bytes)
        .map(|c| RgbImage::from_raw(rw as u32, rh as u32, c.to_vec()).unwrap())
        .collect();
    let got = frames.len().min(indices.len());
    video_from_samples(frames[..got].to_vec(), indices[..got].to_vec(), fps, limits, content_hash(&[bytes], &[]))
}

/// Read media from a data: URL, http(s) URL, or (when allowed) a local path / file:// URL.
pub fn fetch(url: &str, limits: &Limits) -> Result<Vec<u8>> {
    if let Some(rest) = url.strip_prefix("data:") {
        let (meta, data) = rest.split_once(',').context("malformed data: URL")?;
        if !meta.ends_with(";base64") {
            bail!("only base64 data: URLs are supported");
        }
        return base64::engine::general_purpose::STANDARD
            .decode(data.trim())
            .or_else(|_| base64::engine::general_purpose::URL_SAFE.decode(data.trim()))
            .context("invalid base64 in data: URL");
    }
    if url.starts_with("http://") || url.starts_with("https://") {
        let client = reqwest::blocking::Client::builder().timeout(std::time::Duration::from_secs(60)).build()?;
        let resp = client.get(url).send().with_context(|| format!("fetching {url}"))?.error_for_status()?;
        if resp.content_length().is_some_and(|n| n > MAX_FETCH_BYTES) {
            bail!("{url} is larger than {} MB", MAX_FETCH_BYTES >> 20);
        }
        let mut buf = Vec::new();
        resp.take(MAX_FETCH_BYTES + 1).read_to_end(&mut buf)?;
        if buf.len() as u64 > MAX_FETCH_BYTES {
            bail!("{url} is larger than {} MB", MAX_FETCH_BYTES >> 20);
        }
        return Ok(buf);
    }
    if !limits.allow_local_files {
        bail!("media must be a data: or http(s) URL");
    }
    let path = url.strip_prefix("file://").unwrap_or(url);
    std::fs::read(path).with_context(|| format!("reading {path}"))
}

/// A tokenized prompt with media spliced in.
pub struct Prompt {
    pub tokens: Vec<u32>,
    /// Prefix-cache keys: the token id for text, a content-derived key for vision tokens.
    pub keys: Vec<u64>,
    /// M-RoPE (t, h, w) position of every token.
    pub pos: Vec<[u32; 3]>,
    /// Runs of vision tokens: (first token index, media index, frame index).
    pub runs: Vec<(usize, usize, usize)>,
    pub media: Vec<Arc<Media>>,
}

impl Prompt {
    /// Rotary position of the token following the prompt.
    pub fn next_pos(&self) -> u32 {
        self.pos.iter().flat_map(|p| p.iter().copied()).max().map_or(0, |m| m + 1)
    }

    /// A text-only prompt.
    pub fn text(tokens: Vec<u32>) -> Self {
        Prompt {
            keys: tokens.iter().map(|&t| t as u64).collect(),
            pos: (0..tokens.len() as u32).map(|i| [i; 3]).collect(),
            tokens,
            runs: Vec::new(),
            media: Vec::new(),
        }
    }
}

const IMAGE_PH: &str = "<|vision_start|><|image_pad|><|vision_end|>";
const VIDEO_PH: &str = "<|vision_start|><|video_pad|><|vision_end|>";

/// Expand the chat template's media placeholders (images and videos in prompt
/// order) to one pad token per vision feature, then tokenize and assign positions.
pub fn build_prompt(tok: &Tokenizer, vc: &VisionConfig, text: &str, media: Vec<Arc<Media>>) -> Result<Prompt> {
    let mut expanded = String::with_capacity(text.len());
    let mut rest = text;
    let mut order = Vec::new(); // media index for each placeholder
    let (mut next_img, mut next_vid) = (0usize, 0usize);
    let nth = |kind: MediaKind, n: usize| media.iter().enumerate().filter(|(_, m)| m.kind == kind).nth(n).map(|(i, _)| i);
    loop {
        let img = rest.find(IMAGE_PH);
        let vid = rest.find(VIDEO_PH);
        let (at, kind) = match (img, vid) {
            (Some(a), Some(b)) if b < a => (b, MediaKind::Video),
            (Some(a), _) => (a, MediaKind::Image),
            (None, Some(b)) => (b, MediaKind::Video),
            (None, None) => break,
        };
        expanded.push_str(&rest[..at]);
        let counter = if kind == MediaKind::Image { &mut next_img } else { &mut next_vid };
        let idx = nth(kind, *counter).context("prompt has more media placeholders than attached media")?;
        *counter += 1;
        let m = &media[idx];
        let pads = if kind == MediaKind::Image { "<|image_pad|>" } else { "<|video_pad|>" }.repeat(m.tokens_per_frame());
        for f in 0..m.frames {
            if kind == MediaKind::Video {
                expanded.push_str(&format!("<{:.1} seconds>", m.timestamps[f]));
            }
            expanded.push_str("<|vision_start|>");
            expanded.push_str(&pads);
            expanded.push_str("<|vision_end|>");
        }
        order.push(idx);
        rest = &rest[at + if kind == MediaKind::Image { IMAGE_PH.len() } else { VIDEO_PH.len() }..];
    }
    expanded.push_str(rest);
    if order.len() != media.len() {
        bail!("{} media attached but the prompt has {} placeholders", media.len(), order.len());
    }

    let tokens = tok.encode(&expanded)?;
    let mut keys = Vec::with_capacity(tokens.len());
    let mut pos = Vec::with_capacity(tokens.len());
    let mut runs = Vec::new();
    // Each run of pad tokens is one frame of the next media item, in placeholder order.
    let mut frames = order.iter().flat_map(|&mi| (0..media[mi].frames).map(move |f| (mi, f)));
    let mut cur = 0u32;
    let mut i = 0;
    while i < tokens.len() {
        let t = tokens[i];
        if t != vc.image_token && t != vc.video_token {
            keys.push(t as u64);
            pos.push([cur; 3]);
            cur += 1;
            i += 1;
            continue;
        }
        let (mi, f) = frames.next().context("more vision tokens than media")?;
        let m = &media[mi];
        let (gh, gw) = (m.h / MERGE, m.w / MERGE);
        let n = gh * gw;
        if i + n > tokens.len() || tokens[i..i + n].iter().any(|&x| x != t) {
            bail!("vision placeholder tokens do not match the media size");
        }
        runs.push((i, mi, f));
        for k in 0..n {
            keys.push((1 << 63) | (m.hash ^ ((f as u64) << 32 | k as u64)).wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 1);
            pos.push([cur, cur + (k / gw) as u32, cur + (k % gw) as u32]);
        }
        cur += gh.max(gw) as u32;
        i += n;
    }
    if frames.next().is_some() {
        bail!("fewer vision tokens than media frames");
    }
    Ok(Prompt { tokens, keys, pos, runs, media })
}

/// Where one image or video in a request comes from.
pub enum MediaRef {
    Image(String),
    Video(String),
    /// A video given as a list of frame images taken `fps` times per second.
    Frames(Vec<String>, f64),
}

fn url_of(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.clone()),
        Value::Object(o) => o.get("url").and_then(Value::as_str).map(String::from),
        _ => None,
    }
}

/// Pull images and videos out of OpenAI-style message content, in order,
/// replacing each with the bare `{"type": "image"}` / `{"type": "video"}`
/// part the chat template expects.
pub fn extract_media(messages: &[Value]) -> Result<(Vec<Value>, Vec<MediaRef>)> {
    let mut refs = Vec::new();
    let mut out = Vec::with_capacity(messages.len());
    for m in messages {
        let Some(parts) = m.get("content").and_then(Value::as_array) else {
            out.push(m.clone());
            continue;
        };
        let mut new_parts = Vec::with_capacity(parts.len());
        for p in parts {
            let kind = p.get("type").and_then(Value::as_str).unwrap_or("text");
            let missing = || anyhow::anyhow!("`{kind}` content part has no URL");
            match kind {
                "text" => new_parts.push(p.clone()),
                "input_text" => new_parts.push(serde_json::json!({"type": "text", "text": p["text"]})),
                "image_url" | "input_image" | "image" => {
                    let url = url_of(&p["image_url"]).or_else(|| url_of(&p["image"])).ok_or_else(missing)?;
                    refs.push(MediaRef::Image(url));
                    new_parts.push(serde_json::json!({"type": "image"}));
                }
                "video_url" | "video" => {
                    match (&p["video"], url_of(&p["video_url"])) {
                        (Value::Array(frames), _) => {
                            let frames = frames.iter().map(url_of).collect::<Option<Vec<_>>>().ok_or_else(missing)?;
                            refs.push(MediaRef::Frames(frames, p.get("fps").and_then(Value::as_f64).unwrap_or(VIDEO_FPS)));
                        }
                        (v, url) => refs.push(MediaRef::Video(url.or_else(|| url_of(v)).ok_or_else(missing)?)),
                    }
                    new_parts.push(serde_json::json!({"type": "video"}));
                }
                other => bail!("unsupported content part type `{other}`"),
            }
        }
        let mut m = m.clone();
        m["content"] = Value::Array(new_parts);
        out.push(m);
    }
    Ok((out, refs))
}

/// Fetch and preprocess every referenced image and video (blocking).
pub fn load_media(refs: &[MediaRef], limits: &Limits) -> Result<Vec<Arc<Media>>> {
    refs.iter()
        .map(|r| {
            Ok(Arc::new(match r {
                MediaRef::Image(url) => image_from_bytes(&fetch(url, limits)?, limits)?,
                MediaRef::Video(url) => video_from_bytes(&fetch(url, limits)?, limits)?,
                MediaRef::Frames(urls, fps) => {
                    let frames = urls.iter().map(|u| fetch(u, limits)).collect::<Result<Vec<_>>>()?;
                    video_from_frame_images(&frames, *fps, limits)?
                }
            }))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smart_resize_matches_hf_processor() {
        // Values produced by transformers' Qwen2VLImageProcessor / Qwen3VLVideoProcessor.
        assert_eq!(smart_resize_image(600, 800, IMAGE_MIN_PIXELS, 1 << 20).unwrap(), (608, 800));
        assert_eq!(smart_resize_image(600, 800, IMAGE_MIN_PIXELS, 65_536).unwrap(), (192, 288));
        // Small images are scaled up to the minimum pixel count.
        assert_eq!(smart_resize_image(100, 100, IMAGE_MIN_PIXELS, IMAGE_MAX_PIXELS).unwrap(), (256, 256));
        assert_eq!(smart_resize_video(6, 240, 320, VIDEO_MIN_PIXELS, 8 << 20).unwrap(), (256, 320));
        assert!(smart_resize_image(10, 3000, IMAGE_MIN_PIXELS, IMAGE_MAX_PIXELS).is_err());
    }

    #[test]
    fn sampling_and_timestamps() {
        // 12 frames at 4 fps -> 3 s -> 6 frames at 2 fps.
        assert_eq!(sample_indices(12, 4.0, 768), vec![0, 2, 4, 7, 9, 11]);
        // At least 4 frames, never more than exist.
        assert_eq!(sample_indices(3, 30.0, 768), vec![0, 1, 2]);
        assert_eq!(sample_indices(300, 30.0, 768).len(), 20);
    }

    #[test]
    fn patch_order_is_merge_blocks() {
        // A 64x32 image: 4x2 patches -> two 2x2 merge blocks side by side.
        let mut img = RgbImage::new(64, 32);
        for (x, _, p) in img.enumerate_pixels_mut() {
            *p = image::Rgb([(x / 16) as u8 * 50, 0, 0]);
        }
        let v = patchify(&[img.clone(), img]);
        let first_of = |patch: usize| v[patch * 1536];
        let col = |c: u8| c as f32 * 50.0 / 127.5 - 1.0;
        // Block 0 holds patch columns 0,1 (rows 0,1); block 1 holds columns 2,3.
        assert_eq!([first_of(0), first_of(1), first_of(2), first_of(3)], [col(0), col(1), col(0), col(1)]);
        assert_eq!([first_of(4), first_of(5)], [col(2), col(3)]);
    }
}
