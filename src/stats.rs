//! Live server statistics: written by the generation thread, read by the dashboard.

use std::collections::VecDeque;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use crate::worker::{Finish, FinishReason};

/// Seconds of per-second throughput history kept for the charts.
pub const HISTORY_SECS: u64 = 600;
const RECENT: usize = 50;
const LOGS: usize = 50;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Output {
    /// Interactive chat: stay silent.
    Quiet,
    /// Log one line per request to stderr.
    Print,
    /// The dashboard shows everything; never write to the terminal.
    Tui,
}

#[derive(Clone, Copy)]
pub enum Phase {
    /// Running the vision encoder over attached images and videos.
    Encoding { done: usize, total: usize },
    Prefill { done: usize, total: usize },
    Decode { generated: usize, max: usize },
}

#[derive(Clone)]
pub struct Current {
    pub id: u64,
    pub prompt: usize,
    pub cached: usize,
    pub phase: Phase,
    pub started: Instant,
    /// Smoothed decode speed of this request.
    pub rate: f64,
}

/// Work done during one wall-clock second.
#[derive(Clone, Copy, Default)]
pub struct Bucket {
    pub prefill_tokens: usize,
    pub prefill_secs: f64,
    pub decode_tokens: usize,
    pub decode_secs: f64,
}

#[derive(Clone)]
pub struct Record {
    pub id: u64,
    /// Seconds since server start when the request finished.
    pub at: f64,
    pub prompt: usize,
    pub cached: usize,
    pub generated: usize,
    pub prefill_secs: f64,
    pub decode_secs: f64,
    pub outcome: &'static str,
}

#[derive(Clone, Copy, Default)]
pub struct Totals {
    pub requests: u64,
    pub errors: u64,
    pub prompt_tokens: u64,
    pub cached_tokens: u64,
    pub generated_tokens: u64,
    pub busy_secs: f64,
}

#[derive(Clone, Default)]
pub struct Snapshot {
    pub current: Option<Current>,
    /// (second since start, work in that second), oldest first.
    pub buckets: VecDeque<(u64, Bucket)>,
    pub recent: VecDeque<Record>,
    pub totals: Totals,
    pub context_used: usize,
    pub logs: VecDeque<(f64, String)>,
}

pub struct Stats {
    pub start: Instant,
    pub output: Output,
    /// Requests submitted but not yet picked up by the generation thread.
    pub queued: AtomicUsize,
    inner: Mutex<Snapshot>,
}

impl Stats {
    pub fn new(output: Output) -> Self {
        Stats { start: Instant::now(), output, queued: AtomicUsize::new(0), inner: Mutex::default() }
    }

    pub fn snapshot(&self) -> Snapshot {
        self.inner.lock().unwrap().clone()
    }

    pub fn log(&self, msg: impl Into<String>) {
        let msg = msg.into();
        match self.output {
            Output::Tui => {
                let mut s = self.inner.lock().unwrap();
                s.logs.push_back((self.start.elapsed().as_secs_f64(), msg));
                if s.logs.len() > LOGS {
                    s.logs.pop_front();
                }
            }
            Output::Print => eprintln!("{msg}"),
            Output::Quiet => {}
        }
    }

    pub fn begin(&self, prompt: usize, pos: usize) {
        self.queued.fetch_sub(1, Ordering::Relaxed);
        let mut s = self.inner.lock().unwrap();
        let id = s.totals.requests + s.totals.errors + 1;
        s.current = Some(Current {
            id,
            prompt,
            cached: 0,
            phase: Phase::Prefill { done: 0, total: prompt },
            started: Instant::now(),
            rate: 0.0,
        });
        s.context_used = pos;
    }

    /// The prefix cache lookup reused `cached` prompt tokens.
    pub fn begin_cached(&self, cached: usize, pos: usize) {
        let mut s = self.inner.lock().unwrap();
        s.context_used = pos;
        if let Some(c) = &mut s.current {
            c.cached = cached;
            c.phase = Phase::Prefill { done: 0, total: c.prompt - cached };
        }
    }

    pub fn encoding(&self, done: usize, total: usize) {
        if let Some(c) = &mut self.inner.lock().unwrap().current {
            c.phase = Phase::Encoding { done, total };
        }
    }

    pub fn prefilled(&self, tokens: usize, secs: f64, pos: usize) {
        let mut s = self.inner.lock().unwrap();
        self.bucket(&mut s, |b| {
            b.prefill_tokens += tokens;
            b.prefill_secs += secs;
        });
        s.context_used = pos;
        if let Some(c) = &mut s.current {
            match &mut c.phase {
                Phase::Prefill { done, .. } => *done += tokens,
                // The first prompt chunk after encoding media.
                Phase::Encoding { .. } => c.phase = Phase::Prefill { done: tokens, total: c.prompt - c.cached },
                Phase::Decode { .. } => {}
            }
        }
    }

    /// One decode step finished; `generated` counts sampled tokens so far.
    pub fn decoded(&self, generated: usize, max: usize, secs: f64, pos: usize) {
        let mut s = self.inner.lock().unwrap();
        self.bucket(&mut s, |b| {
            b.decode_tokens += 1;
            b.decode_secs += secs;
        });
        s.context_used = pos;
        if let Some(c) = &mut s.current {
            let r = 1.0 / secs.max(1e-6);
            c.rate = if c.rate == 0.0 { r } else { 0.9 * c.rate + 0.1 * r };
            c.phase = Phase::Decode { generated, max };
        }
    }

    pub fn finish(&self, f: &Finish) {
        let new = f.prompt_tokens - f.cached_tokens;
        let pp = new as f64 / f.prefill_secs.max(1e-9);
        let tg = f.completion_tokens as f64 / f.decode_secs.max(1e-9);
        if self.output == Output::Print {
            eprintln!(
                "request: prompt {} tok ({} cached, {pp:.0} tok/s), generated {} tok ({tg:.1} tok/s)",
                f.prompt_tokens, f.cached_tokens, f.completion_tokens,
            );
        }
        let mut s = self.inner.lock().unwrap();
        let id = s.current.take().map_or(0, |c| c.id);
        let t = &mut s.totals;
        t.requests += 1;
        t.prompt_tokens += f.prompt_tokens as u64;
        t.cached_tokens += f.cached_tokens as u64;
        t.generated_tokens += f.completion_tokens as u64;
        t.busy_secs += f.prefill_secs + f.decode_secs;
        let outcome = match (f.reason, f.cancelled) {
            (_, true) => "cancelled",
            (FinishReason::Stop, _) => "stop",
            (FinishReason::Length, _) => "length",
        };
        self.record(&mut s, Record {
            id,
            at: self.start.elapsed().as_secs_f64(),
            prompt: f.prompt_tokens,
            cached: f.cached_tokens,
            generated: f.completion_tokens,
            prefill_secs: f.prefill_secs,
            decode_secs: f.decode_secs,
            outcome,
        });
    }

    pub fn error(&self, msg: &str) {
        {
            let mut s = self.inner.lock().unwrap();
            s.totals.errors += 1;
            if let Some(c) = s.current.take() {
                let rec = Record {
                    id: c.id,
                    at: self.start.elapsed().as_secs_f64(),
                    prompt: c.prompt,
                    cached: c.cached,
                    generated: 0,
                    prefill_secs: 0.0,
                    decode_secs: 0.0,
                    outcome: "error",
                };
                self.record(&mut s, rec);
            }
        }
        self.log(format!("error: {msg}"));
    }

    fn record(&self, s: &mut Snapshot, r: Record) {
        s.recent.push_front(r);
        s.recent.truncate(RECENT);
    }

    fn bucket(&self, s: &mut Snapshot, f: impl FnOnce(&mut Bucket)) {
        if self.output != Output::Tui {
            return;
        }
        let now = self.start.elapsed().as_secs();
        if s.buckets.back().is_none_or(|(t, _)| *t != now) {
            s.buckets.push_back((now, Bucket::default()));
        }
        while s.buckets.front().is_some_and(|(t, _)| *t + HISTORY_SECS < now) {
            s.buckets.pop_front();
        }
        f(&mut s.buckets.back_mut().unwrap().1);
    }
}
