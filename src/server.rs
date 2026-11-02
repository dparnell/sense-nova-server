//! OpenAI-compatible HTTP API: /v1/models, /v1/chat/completions, /v1/completions.

use std::convert::Infallible;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::{Request, State};
use axum::http::{HeaderMap, StatusCode};
use axum::middleware::{self, Next};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures::Stream;
use serde_json::{Value, json};
use tokio::sync::mpsc::{UnboundedReceiver, unbounded_channel};

use crate::chat::{ChatTemplate, OutputParser, Piece, Tokenizer, ToolCall};
use crate::config::VisionConfig;
use crate::mm::{self, Limits, Prompt};
use crate::sampler::SamplingParams;
use crate::stats::Stats;
use crate::worker::{Finish, FinishReason, GenEvent, GenRequest};

pub struct ServerConfig {
    pub model_name: String,
    pub api_key: Option<String>,
    pub defaults: SamplingParams,
    pub max_tokens: usize,
    pub thinking: bool,
    /// Return reasoning in `reasoning_content` (true) or inline in `content` with <think> tags.
    pub separate_reasoning: bool,
}

pub struct AppState {
    pub cfg: ServerConfig,
    pub tx: std::sync::Mutex<std::sync::mpsc::Sender<GenRequest>>,
    pub tok: Arc<Tokenizer>,
    pub template: ChatTemplate,
    pub im_start: Option<u32>,
    pub ctx: usize,
    pub stats: Arc<Stats>,
    /// Present when the vision encoder is loaded.
    pub vision: Option<VisionConfig>,
    pub limits: Limits,
}

type Shared = Arc<AppState>;

pub fn router(state: Shared) -> Router {
    let api = Router::new()
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route_layer(middleware::from_fn_with_state(state.clone(), auth));
    Router::new()
        .route("/health", get(|| async { Json(json!({"status": "ok"})) }))
        .route("/", get(|| async { "SenseNova Loop OpenAI-compatible server. Try GET /v1/models\n" }))
        .merge(api)
        .layer(tower_http::cors::CorsLayer::permissive())
        .with_state(state)
}

async fn auth(State(st): State<Shared>, headers: HeaderMap, req: Request, next: Next) -> Response {
    if let Some(key) = &st.cfg.api_key {
        let ok = headers
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "))
            .is_some_and(|v| v == key);
        if !ok {
            return error(StatusCode::UNAUTHORIZED, "invalid or missing API key", "authentication_error");
        }
    }
    next.run(req).await
}

fn error(status: StatusCode, msg: &str, kind: &str) -> Response {
    (status, Json(json!({"error": {"message": msg, "type": kind, "code": status.as_u16()}}))).into_response()
}

fn bad_request(msg: impl AsRef<str>) -> Response {
    error(StatusCode::BAD_REQUEST, msg.as_ref(), "invalid_request_error")
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

fn new_id(prefix: &str) -> String {
    static N: AtomicU64 = AtomicU64::new(0);
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    format!("{prefix}-{:x}{:04x}", nanos, N.fetch_add(1, Ordering::Relaxed) & 0xffff)
}

async fn models(State(st): State<Shared>) -> Json<Value> {
    Json(json!({
        "object": "list",
        "data": [{
            "id": st.cfg.model_name,
            "object": "model",
            "created": now(),
            "owned_by": "sensenova",
            "max_model_len": st.ctx,
        }]
    }))
}

struct Common {
    params: SamplingParams,
    max_tokens: usize,
    stop: Vec<String>,
    stream: bool,
    include_usage: bool,
}

fn common(st: &AppState, b: &Value) -> Common {
    let d = &st.cfg.defaults;
    let f = |k: &str, def: f32| b.get(k).and_then(Value::as_f64).map_or(def, |v| v as f32);
    let params = SamplingParams {
        temperature: f("temperature", d.temperature),
        top_p: f("top_p", d.top_p),
        top_k: b.get("top_k").and_then(Value::as_i64).map_or(d.top_k, |v| v.max(0) as usize),
        min_p: f("min_p", d.min_p),
        presence_penalty: f("presence_penalty", d.presence_penalty),
        frequency_penalty: f("frequency_penalty", d.frequency_penalty),
        repetition_penalty: f("repetition_penalty", d.repetition_penalty),
        seed: b.get("seed").and_then(Value::as_u64).or(d.seed),
    };
    let max_tokens = b
        .get("max_completion_tokens")
        .or_else(|| b.get("max_tokens"))
        .and_then(Value::as_u64)
        .map_or(st.cfg.max_tokens, |v| v as usize);
    let stop = match b.get("stop") {
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(a)) => a.iter().filter_map(|v| v.as_str().map(String::from)).collect(),
        _ => Vec::new(),
    };
    Common {
        params,
        max_tokens,
        stop,
        stream: b.get("stream").and_then(Value::as_bool).unwrap_or(false),
        include_usage: b["stream_options"]["include_usage"].as_bool().unwrap_or(false),
    }
}

fn usage(f: &Finish) -> Value {
    json!({
        "prompt_tokens": f.prompt_tokens,
        "completion_tokens": f.completion_tokens,
        "total_tokens": f.prompt_tokens + f.completion_tokens,
        "prompt_tokens_details": {"cached_tokens": f.cached_tokens},
    })
}

fn finish_reason(f: &Finish, tool_calls: bool) -> &'static str {
    if tool_calls {
        "tool_calls"
    } else if f.reason == FinishReason::Length {
        "length"
    } else {
        "stop"
    }
}

fn submit(st: &AppState, prompt: Prompt, snapshot_at: Option<usize>, c: &Common) -> UnboundedReceiver<GenEvent> {
    let (tx, rx) = unbounded_channel();
    let req = GenRequest {
        prompt,
        snapshot_at,
        params: c.params.clone(),
        max_tokens: c.max_tokens,
        stop: c.stop.clone(),
        events: tx.clone(),
    };
    st.stats.queued.fetch_add(1, Ordering::Relaxed);
    if st.tx.lock().unwrap().send(req).is_err() {
        st.stats.queued.fetch_sub(1, Ordering::Relaxed);
        let _ = tx.send(GenEvent::Error("generation thread has stopped".into()));
    }
    rx
}

fn thinking_enabled(st: &AppState, b: &Value) -> bool {
    if let Some(v) = b["chat_template_kwargs"]["enable_thinking"].as_bool() {
        return v;
    }
    if let Some(v) = b["enable_thinking"].as_bool() {
        return v;
    }
    if let Some(e) = b["reasoning_effort"].as_str() {
        return !matches!(e, "none" | "minimal");
    }
    st.cfg.thinking
}

fn tool_calls_json(calls: &[ToolCall], with_index: bool) -> Value {
    Value::Array(
        calls
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let mut v = json!({
                    "id": new_id("call"),
                    "type": "function",
                    "function": {"name": c.name, "arguments": c.arguments},
                });
                if with_index {
                    v["index"] = json!(i);
                }
                v
            })
            .collect(),
    )
}

async fn chat_completions(State(st): State<Shared>, Json(b): Json<Value>) -> Response {
    let Some(messages) = b.get("messages").and_then(Value::as_array) else {
        return bad_request("`messages` is required");
    };
    let (messages, refs) = match mm::extract_media(messages) {
        Ok(x) => x,
        Err(e) => return bad_request(format!("{e:#}")),
    };
    let media = if refs.is_empty() {
        Vec::new()
    } else {
        if st.vision.is_none() {
            return bad_request("image and video inputs need the vision encoder, which is not loaded (run `sensenova-loop pull` to add it, and do not pass --no-vision)");
        }
        let limits = st.limits;
        match tokio::task::spawn_blocking(move || mm::load_media(&refs, &limits)).await {
            Ok(Ok(m)) => m,
            Ok(Err(e)) => return bad_request(format!("{e:#}")),
            Err(e) => return error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string(), "server_error"),
        }
    };
    let tools = match b.get("tools") {
        Some(Value::Array(t)) if !t.is_empty() && b["tool_choice"].as_str() != Some("none") => Some(Value::Array(t.clone())),
        _ => None,
    };
    let thinking = thinking_enabled(&st, &b);
    let text = match st.template.render(&messages, tools.as_ref(), thinking) {
        Ok(p) => p,
        Err(e) => return bad_request(format!("{e:#}")),
    };
    let prompt = match (&st.vision, media.is_empty()) {
        (Some(vc), false) => mm::build_prompt(&st.tok, vc, &text, media),
        _ => st.tok.encode(&text).map(Prompt::text),
    };
    let prompt = match prompt {
        Ok(p) => p,
        Err(e) => return bad_request(format!("{e:#}")),
    };
    if prompt.tokens.len() >= st.ctx {
        return bad_request(format!("prompt is {} tokens; the server context size is {}", prompt.tokens.len(), st.ctx));
    }
    // Snapshot right before the final assistant header so the next turn can reuse the history.
    let snapshot_at = st.im_start.and_then(|id| prompt.tokens.iter().rposition(|&t| t == id));
    let c = common(&st, &b);
    let rx = submit(&st, prompt, snapshot_at, &c);
    let parser = OutputParser::new(thinking, st.cfg.separate_reasoning, tools.is_some());
    let id = new_id("chatcmpl");
    if c.stream {
        let stream = chat_stream(st.clone(), rx, parser, tools, id, c.include_usage);
        Sse::new(stream).keep_alive(KeepAlive::default()).into_response()
    } else {
        chat_once(&st, rx, parser, tools.as_ref(), id).await
    }
}

async fn chat_once(
    st: &AppState,
    mut rx: UnboundedReceiver<GenEvent>,
    mut parser: OutputParser,
    tools: Option<&Value>,
    id: String,
) -> Response {
    let mut content = String::new();
    let mut reasoning = String::new();
    let mut add = |pieces: Vec<Piece>| {
        for p in pieces {
            match p {
                Piece::Content(s) => content.push_str(&s),
                Piece::Reasoning(s) => reasoning.push_str(&s),
            }
        }
    };
    while let Some(ev) = rx.recv().await {
        match ev {
            GenEvent::Text(t) => add(parser.push(&t)),
            GenEvent::Error(e) => return error(StatusCode::INTERNAL_SERVER_ERROR, &e, "server_error"),
            GenEvent::Done(f) => {
                let (rest, calls) = parser.finish(tools);
                add(rest);
                let mut message = json!({"role": "assistant", "content": content});
                if !reasoning.is_empty() {
                    message["reasoning_content"] = json!(reasoning.trim());
                }
                if !calls.is_empty() {
                    message["tool_calls"] = tool_calls_json(&calls, false);
                    if content.is_empty() {
                        message["content"] = Value::Null;
                    }
                }
                return Json(json!({
                    "id": id,
                    "object": "chat.completion",
                    "created": now(),
                    "model": st.cfg.model_name,
                    "choices": [{"index": 0, "message": message, "finish_reason": finish_reason(&f, !calls.is_empty())}],
                    "usage": usage(&f),
                }))
                .into_response();
            }
        }
    }
    error(StatusCode::INTERNAL_SERVER_ERROR, "generation ended unexpectedly", "server_error")
}

fn chat_stream(
    st: Shared,
    mut rx: UnboundedReceiver<GenEvent>,
    mut parser: OutputParser,
    tools: Option<Value>,
    id: String,
    include_usage: bool,
) -> impl Stream<Item = Result<Event, Infallible>> {
    let model = st.cfg.model_name.clone();
    let created = now();
    let usage_id = id.clone();
    let chunk = move |delta: Value, finish: Option<&str>| {
        Event::default().data(
            json!({
                "id": id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            })
            .to_string(),
        )
    };
    let piece_delta = |p: Piece| match p {
        Piece::Content(s) => json!({"content": s}),
        Piece::Reasoning(s) => json!({"reasoning_content": s}),
    };
    async_stream::stream! {
        yield Ok(chunk(json!({"role": "assistant", "content": ""}), None));
        while let Some(ev) = rx.recv().await {
            match ev {
                GenEvent::Text(t) => {
                    for p in parser.push(&t) {
                        yield Ok(chunk(piece_delta(p), None));
                    }
                }
                GenEvent::Error(e) => {
                    yield Ok(Event::default().data(json!({"error": {"message": e, "type": "server_error"}}).to_string()));
                    break;
                }
                GenEvent::Done(f) => {
                    let (rest, calls) = parser.finish(tools.as_ref());
                    for p in rest {
                        yield Ok(chunk(piece_delta(p), None));
                    }
                    if !calls.is_empty() {
                        yield Ok(chunk(json!({"tool_calls": tool_calls_json(&calls, true)}), None));
                    }
                    yield Ok(chunk(json!({}), Some(finish_reason(&f, !calls.is_empty()))));
                    if include_usage {
                        let u = json!({
                            "id": usage_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": st.cfg.model_name,
                            "choices": [],
                            "usage": usage(&f),
                        });
                        yield Ok(Event::default().data(u.to_string()));
                    }
                    break;
                }
            }
        }
        yield Ok(Event::default().data("[DONE]"));
    }
}

async fn completions(State(st): State<Shared>, Json(b): Json<Value>) -> Response {
    let tokens = match b.get("prompt") {
        Some(Value::String(s)) => st.tok.encode(s),
        Some(Value::Array(a)) if a.first().is_some_and(Value::is_string) => st.tok.encode(a[0].as_str().unwrap()),
        Some(Value::Array(a)) if a.iter().all(Value::is_u64) => Ok(a.iter().map(|v| v.as_u64().unwrap() as u32).collect()),
        _ => return bad_request("`prompt` must be a string or an array of token ids"),
    };
    let tokens = match tokens {
        Ok(t) if !t.is_empty() => t,
        Ok(_) => return bad_request("empty prompt"),
        Err(e) => return bad_request(format!("{e:#}")),
    };
    if tokens.len() >= st.ctx {
        return bad_request(format!("prompt is {} tokens; the server context size is {}", tokens.len(), st.ctx));
    }
    let mut c = common(&st, &b);
    if b.get("max_tokens").is_none() {
        c.max_tokens = 16.max(c.max_tokens.min(256));
    }
    let mut rx = submit(&st, Prompt::text(tokens), None, &c);
    let id = new_id("cmpl");
    let model = st.cfg.model_name.clone();
    if c.stream {
        let include_usage = c.include_usage;
        let stream = async_stream::stream! {
            let created = now();
            let chunk = |text: &str, finish: Option<&str>| json!({
                "id": id, "object": "text_completion", "created": created, "model": model,
                "choices": [{"index": 0, "text": text, "finish_reason": finish, "logprobs": null}],
            });
            while let Some(ev) = rx.recv().await {
                match ev {
                    GenEvent::Text(t) => yield Ok::<_, Infallible>(Event::default().data(chunk(&t, None).to_string())),
                    GenEvent::Error(e) => {
                        yield Ok(Event::default().data(json!({"error": {"message": e}}).to_string()));
                        break;
                    }
                    GenEvent::Done(f) => {
                        let mut last = chunk("", Some(finish_reason(&f, false)));
                        if include_usage {
                            last["usage"] = usage(&f);
                        }
                        yield Ok(Event::default().data(last.to_string()));
                        break;
                    }
                }
            }
            yield Ok(Event::default().data("[DONE]"));
        };
        return Sse::new(stream).keep_alive(KeepAlive::default()).into_response();
    }
    let mut text = String::new();
    while let Some(ev) = rx.recv().await {
        match ev {
            GenEvent::Text(t) => text.push_str(&t),
            GenEvent::Error(e) => return error(StatusCode::INTERNAL_SERVER_ERROR, &e, "server_error"),
            GenEvent::Done(f) => {
                return Json(json!({
                    "id": id,
                    "object": "text_completion",
                    "created": now(),
                    "model": model,
                    "choices": [{"index": 0, "text": text, "finish_reason": finish_reason(&f, false), "logprobs": null}],
                    "usage": usage(&f),
                }))
                .into_response();
            }
        }
    }
    error(StatusCode::INTERNAL_SERVER_ERROR, "generation ended unexpectedly", "server_error")
}
