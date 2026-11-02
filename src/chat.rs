//! Tokenizer, chat template rendering, and parsing of model output into
//! reasoning, content and tool calls.

use std::path::Path;

use anyhow::{Context, Result, anyhow};
use minijinja::{Environment, Error, ErrorKind, Value as JValue};
use serde_json::{Map, Value, json};

pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
}

impl Tokenizer {
    pub fn load(path: &Path) -> Result<Self> {
        let inner = tokenizers::Tokenizer::from_file(path).map_err(|e| anyhow!("loading tokenizer: {e}"))?;
        Ok(Tokenizer { inner })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self.inner.encode(text, false).map_err(|e| anyhow!("tokenizing: {e}"))?.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        self.inner.decode(ids, false).unwrap_or_default()
    }

    pub fn token_id(&self, s: &str) -> Option<u32> {
        self.inner.token_to_id(s)
    }
}

/// Incremental detokenizer that only emits complete UTF-8 text.
#[derive(Default)]
pub struct Detokenizer {
    ids: Vec<u32>,
    prefix: usize,
    read: usize,
}

impl Detokenizer {
    pub fn push(&mut self, tok: &Tokenizer, id: u32) -> String {
        self.ids.push(id);
        let before = tok.decode(&self.ids[self.prefix..self.read]);
        let after = tok.decode(&self.ids[self.prefix..]);
        if after.len() > before.len() && !after.ends_with('\u{FFFD}') {
            let delta = after[before.len()..].to_string();
            self.prefix = self.read;
            self.read = self.ids.len();
            delta
        } else {
            String::new()
        }
    }
}

/// Python `json.dumps` style separators (", " and ": "), as Jinja's `tojson` produces in transformers.
struct PyFormatter;

impl serde_json::ser::Formatter for PyFormatter {
    fn begin_array_value<W: ?Sized + std::io::Write>(&mut self, w: &mut W, first: bool) -> std::io::Result<()> {
        if first { Ok(()) } else { w.write_all(b", ") }
    }
    fn begin_object_key<W: ?Sized + std::io::Write>(&mut self, w: &mut W, first: bool) -> std::io::Result<()> {
        if first { Ok(()) } else { w.write_all(b", ") }
    }
    fn begin_object_value<W: ?Sized + std::io::Write>(&mut self, w: &mut W) -> std::io::Result<()> {
        w.write_all(b": ")
    }
}

pub fn py_json(v: &impl serde::Serialize) -> String {
    let mut buf = Vec::new();
    let mut ser = serde_json::Serializer::with_formatter(&mut buf, PyFormatter);
    v.serialize(&mut ser).ok();
    String::from_utf8(buf).unwrap_or_default()
}

pub struct ChatTemplate {
    env: Environment<'static>,
}

impl ChatTemplate {
    pub fn load(path: &Path) -> Result<Self> {
        let src = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let mut env = Environment::new();
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function("raise_exception", |msg: String| -> Result<String, Error> {
            Err(Error::new(ErrorKind::InvalidOperation, msg))
        });
        env.add_filter("tojson", |v: JValue| -> String { py_json(&v) });
        // transformers renders chat templates with trim_blocks and lstrip_blocks.
        env.set_syntax(
            minijinja::syntax::SyntaxConfig::builder()
                .trim_blocks(true)
                .lstrip_blocks(true)
                .build()?,
        );
        env.add_template_owned("chat", src)?;
        Ok(ChatTemplate { env })
    }

    pub fn render(&self, messages: &[Value], tools: Option<&Value>, enable_thinking: bool) -> Result<String> {
        let messages: Vec<Value> = messages.iter().map(normalize_message).collect();
        let tmpl = self.env.get_template("chat")?;
        let ctx = json!({
            "messages": messages,
            "tools": tools.cloned().unwrap_or(Value::Null),
            "add_generation_prompt": true,
            "enable_thinking": enable_thinking,
        });
        tmpl.render(JValue::from(minijinja::value::Serde(&ctx))).map_err(|e| anyhow!("chat template: {e:#}"))
    }
}

/// Adapt OpenAI-style messages to what the template expects: tool call
/// arguments as objects rather than JSON strings, and no null content.
fn normalize_message(m: &Value) -> Value {
    let mut m = m.clone();
    if let Some(obj) = m.as_object_mut() {
        if obj.get("content").is_some_and(Value::is_null) {
            obj.insert("content".into(), Value::String(String::new()));
        }
        if let Some(Value::Array(calls)) = obj.get_mut("tool_calls") {
            for c in calls {
                if let Some(f) = c.get_mut("function").and_then(Value::as_object_mut) {
                    if let Some(Value::String(s)) = f.get("arguments") {
                        let parsed = serde_json::from_str::<Value>(s).unwrap_or_else(|_| json!({}));
                        f.insert("arguments".into(), parsed);
                    }
                }
            }
        }
    }
    m
}

#[derive(Clone, Debug, PartialEq)]
pub enum Piece {
    Reasoning(String),
    Content(String),
}

#[derive(Clone, Debug)]
pub struct ToolCall {
    pub name: String,
    pub arguments: String,
}

/// Splits streamed model text into reasoning (`... </think>`), content, and
/// a trailing tool-call section (`<tool_call> ...`).
pub struct OutputParser {
    in_reasoning: bool,
    separate_reasoning: bool,
    detect_tools: bool,
    pending: String,
    tool_buf: Option<String>,
    strip_leading: bool,
}

const THINK_END: &str = "</think>";
const TOOL_START: &str = "<tool_call>";

impl OutputParser {
    /// `thinking`: the prompt ended inside an open `<think>` block.
    pub fn new(thinking: bool, separate_reasoning: bool, detect_tools: bool) -> Self {
        OutputParser {
            in_reasoning: thinking,
            separate_reasoning,
            detect_tools,
            pending: String::new(),
            tool_buf: None,
            strip_leading: !thinking,
        }
    }

    pub fn push(&mut self, text: &str) -> Vec<Piece> {
        let mut out = Vec::new();
        if let Some(buf) = &mut self.tool_buf {
            buf.push_str(text);
            return out;
        }
        self.pending.push_str(text);
        loop {
            if self.in_reasoning {
                if let Some(i) = self.pending.find(THINK_END) {
                    let r = self.pending[..i].to_string();
                    self.pending = self.pending[i + THINK_END.len()..].to_string();
                    self.in_reasoning = false;
                    self.strip_leading = true;
                    if self.separate_reasoning {
                        push_piece(&mut out, Piece::Reasoning(r));
                    } else {
                        push_piece(&mut out, Piece::Content(r + THINK_END));
                    }
                    continue;
                }
                let keep = partial_suffix(&self.pending, THINK_END);
                let emit: String = self.pending.drain(..self.pending.len() - keep).collect();
                if self.separate_reasoning {
                    push_piece(&mut out, Piece::Reasoning(emit));
                } else {
                    push_piece(&mut out, Piece::Content(emit));
                }
                return out;
            }
            if self.strip_leading {
                let trimmed = self.pending.trim_start_matches(['\n', ' ']);
                if trimmed.is_empty() {
                    self.pending.clear();
                    return out;
                }
                self.pending = trimmed.to_string();
                self.strip_leading = false;
            }
            if self.detect_tools {
                if let Some(i) = self.pending.find(TOOL_START) {
                    let c = self.pending[..i].trim_end().to_string();
                    push_piece(&mut out, Piece::Content(c));
                    self.tool_buf = Some(self.pending[i..].to_string());
                    self.pending.clear();
                    return out;
                }
                let keep = partial_suffix(&self.pending, TOOL_START);
                let emit: String = self.pending.drain(..self.pending.len() - keep).collect();
                push_piece(&mut out, Piece::Content(emit));
            } else {
                let emit = std::mem::take(&mut self.pending);
                push_piece(&mut out, Piece::Content(emit));
            }
            return out;
        }
    }

    /// Flush buffered text and return parsed tool calls (if any).
    pub fn finish(&mut self, tools: Option<&Value>) -> (Vec<Piece>, Vec<ToolCall>) {
        let mut out = Vec::new();
        let rest = std::mem::take(&mut self.pending);
        if self.in_reasoning && self.separate_reasoning {
            push_piece(&mut out, Piece::Reasoning(rest));
        } else {
            push_piece(&mut out, Piece::Content(rest));
        }
        let calls = match self.tool_buf.take() {
            Some(buf) => {
                let calls = parse_tool_calls(&buf, tools);
                if calls.is_empty() {
                    push_piece(&mut out, Piece::Content(buf));
                }
                calls
            }
            None => Vec::new(),
        };
        (out, calls)
    }
}

fn push_piece(out: &mut Vec<Piece>, p: Piece) {
    match &p {
        Piece::Reasoning(s) | Piece::Content(s) if s.is_empty() => {}
        _ => out.push(p),
    }
}

/// Length of the longest suffix of `s` that is a proper prefix of `pat`.
pub fn partial_suffix(s: &str, pat: &str) -> usize {
    (1..pat.len().min(s.len() + 1))
        .rev()
        .find(|&n| s.is_char_boundary(s.len() - n) && pat.starts_with(&s[s.len() - n..]))
        .unwrap_or(0)
}

/// Parse `<tool_call><function=NAME><parameter=P>VALUE</parameter></function></tool_call>` blocks.
pub fn parse_tool_calls(text: &str, tools: Option<&Value>) -> Vec<ToolCall> {
    let mut calls = Vec::new();
    for block in text.split(TOOL_START).skip(1) {
        let block = block.split("</tool_call>").next().unwrap_or(block);
        let Some(fstart) = block.find("<function=") else { continue };
        let after = &block[fstart + "<function=".len()..];
        let Some(gt) = after.find('>') else { continue };
        let name = after[..gt].trim().to_string();
        let body = &after[gt + 1..];
        let body = body.split("</function>").next().unwrap_or(body);
        let props = tools
            .and_then(Value::as_array)
            .and_then(|ts| ts.iter().find(|t| t["function"]["name"].as_str() == Some(name.as_str())))
            .map(|t| t["function"]["parameters"]["properties"].clone());
        let mut args = Map::new();
        for p in body.split("<parameter=").skip(1) {
            let Some(gt) = p.find('>') else { continue };
            let pname = p[..gt].trim().to_string();
            let mut val = &p[gt + 1..];
            if let Some(end) = val.find("</parameter>") {
                val = &val[..end];
            }
            let val = val.strip_prefix('\n').unwrap_or(val);
            let val = val.strip_suffix('\n').unwrap_or(val);
            let declared = props.as_ref().and_then(|ps| ps[&pname]["type"].as_str().map(str::to_string));
            let v = match declared.as_deref() {
                Some("string") => Value::String(val.to_string()),
                _ => serde_json::from_str::<Value>(val.trim()).unwrap_or_else(|_| Value::String(val.to_string())),
            };
            args.insert(pname, v);
        }
        calls.push(ToolCall { name, arguments: Value::Object(args).to_string() });
    }
    calls
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_reasoning_and_content() {
        let mut p = OutputParser::new(true, true, false);
        let mut all = Vec::new();
        for chunk in ["Let me th", "ink.</th", "ink>\n\nHel", "lo"] {
            all.extend(p.push(chunk));
        }
        all.extend(p.finish(None).0);
        let reasoning: String = all.iter().filter_map(|x| if let Piece::Reasoning(s) = x { Some(s.as_str()) } else { None }).collect();
        let content: String = all.iter().filter_map(|x| if let Piece::Content(s) = x { Some(s.as_str()) } else { None }).collect();
        assert_eq!(reasoning, "Let me think.");
        assert_eq!(content, "Hello");
    }

    #[test]
    fn parses_tool_calls() {
        let tools = json!([{"type": "function", "function": {"name": "get_weather", "parameters": {"properties": {"city": {"type": "string"}, "days": {"type": "integer"}}}}}]);
        let mut p = OutputParser::new(false, true, true);
        let mut pieces = p.push("Checking.\n\n<tool_ca");
        pieces.extend(p.push("ll>\n<function=get_weather>\n<parameter=city>\n42\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>"));
        let (rest, calls) = p.finish(Some(&tools));
        pieces.extend(rest);
        let content: String = pieces.iter().map(|p| if let Piece::Content(s) = p { s.as_str() } else { "" }).collect();
        assert_eq!(content.trim(), "Checking.");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        let args: Value = serde_json::from_str(&calls[0].arguments).unwrap();
        assert_eq!(args, json!({"city": "42", "days": 3}));
    }

    #[test]
    fn python_json_separators() {
        assert_eq!(py_json(&json!({"a": [1, 2], "b": "x"})), r#"{"a": [1, 2], "b": "x"}"#);
    }
}
