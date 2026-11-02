//! Live terminal dashboard for `serve`: throughput charts, GPU telemetry and recent requests.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use anyhow::Result;
use nvml_wrapper::Nvml;
use nvml_wrapper::enum_wrappers::device::TemperatureSensor;
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style, Stylize};
use ratatui::symbols::Marker;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Axis, Block, Cell, Chart, Dataset, Gauge, GraphType, Paragraph, Row, Table};
use ratatui::{DefaultTerminal, Frame};
use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate, System};

use crate::stats::{HISTORY_SECS, Phase, Snapshot, Stats};

const GEN: Color = Color::Cyan;
const PREFILL: Color = Color::Magenta;
const GPU_LOAD: Color = Color::Green;
const VRAM: Color = Color::Blue;
const DIM: Color = Color::DarkGray;

pub struct Info {
    pub url: String,
    pub model_id: String,
    pub quant: String,
    pub gpu: wgpu::AdapterInfo,
    pub ctx: usize,
    /// GPU memory allocated for weights and caches.
    pub model_bytes: u64,
}

/// Show the dashboard until the user quits (q, Esc or Ctrl+C).
pub fn run(info: Info, stats: Arc<Stats>) -> Result<()> {
    let mut term = ratatui::init();
    let res = App::new(info, stats).run(&mut term);
    ratatui::restore();
    res
}

#[derive(Clone, Copy, Default)]
struct HwSample {
    gpu_load: Option<u32>,
    vram_used: Option<u64>,
    vram_total: Option<u64>,
    temp: Option<u32>,
    power_w: Option<f64>,
    cpu: f32,
    rss: u64,
}

struct Hw {
    /// NVML handle and device index, for NVIDIA GPUs.
    nvml: Option<(Nvml, u32)>,
    sys: Option<System>,
    pid: Pid,
    cores: f32,
}

impl Hw {
    fn new(info: &wgpu::AdapterInfo) -> Self {
        Hw {
            nvml: find_nvml(info),
            sys: System::new().ok(),
            pid: sysinfo::get_current_pid().unwrap_or(Pid::from_u32(0)),
            cores: std::thread::available_parallelism().map_or(1, |n| n.get()) as f32,
        }
    }

    fn sample(&mut self) -> HwSample {
        let mut h = HwSample::default();
        if let Some(d) = self.nvml.as_ref().and_then(|(n, i)| n.device_by_index(*i).ok()) {
            h.gpu_load = d.utilization_rates().ok().map(|u| u.gpu);
            if let Ok(m) = d.memory_info() {
                h.vram_used = Some(m.used);
                h.vram_total = Some(m.total);
            }
            h.temp = d.temperature(TemperatureSensor::Gpu).ok();
            h.power_w = d.power_usage().ok().map(|mw| mw as f64 / 1000.0);
        }
        let Some(sys) = &mut self.sys else { return h };
        sys.refresh_processes_specifics(
            ProcessesToUpdate::Some(&[self.pid]),
            true,
            ProcessRefreshKind::nothing().with_memory().with_cpu(),
        );
        if let Some(p) = sys.process(self.pid) {
            h.cpu = p.cpu_usage() / self.cores;
            h.rss = p.memory();
        }
        h
    }
}

/// Find the NVML device behind the wgpu adapter: by PCI bus id, then by name.
fn find_nvml(info: &wgpu::AdapterInfo) -> Option<(Nvml, u32)> {
    if info.vendor != 0x10de {
        return None;
    }
    let nvml = Nvml::init().ok()?;
    let n = nvml.device_count().ok()?;
    // wgpu reports "dddd:bb:dd.f"; NVML pads the domain to 8 digits.
    let bus = info.device_pci_bus_id.to_lowercase();
    let bus = bus.split_once(':').map_or("", |(_, rest)| rest).to_string();
    let matches = |i: u32, f: &dyn Fn(&nvml_wrapper::Device) -> bool| nvml.device_by_index(i).is_ok_and(|d| f(&d));
    let idx = (0..n)
        .find(|&i| !bus.is_empty() && matches(i, &|d| d.pci_info().is_ok_and(|p| p.bus_id.to_lowercase().ends_with(&bus))))
        .or_else(|| (0..n).find(|&i| matches(i, &|d| d.name().is_ok_and(|name| name == info.name))))
        .or((n == 1).then_some(0))?;
    Some((nvml, idx))
}

struct App {
    info: Info,
    stats: Arc<Stats>,
    hw: Hw,
    /// (second since start, sample), oldest first.
    hw_hist: VecDeque<(u64, HwSample)>,
}

impl App {
    fn new(info: Info, stats: Arc<Stats>) -> Self {
        let hw = Hw::new(&info.gpu);
        App { info, stats, hw, hw_hist: VecDeque::new() }
    }

    fn run(&mut self, term: &mut DefaultTerminal) -> Result<()> {
        let mut last_sample: Option<Instant> = None;
        loop {
            if last_sample.is_none_or(|t| t.elapsed() >= Duration::from_secs(1)) {
                last_sample = Some(Instant::now());
                let now = self.stats.start.elapsed().as_secs();
                self.hw_hist.push_back((now, self.hw.sample()));
                while self.hw_hist.front().is_some_and(|(t, _)| t + HISTORY_SECS < now) {
                    self.hw_hist.pop_front();
                }
            }
            let snap = self.stats.snapshot();
            term.draw(|f| self.draw(f, &snap))?;
            if event::poll(Duration::from_millis(250))?
                && let Event::Key(k) = event::read()?
                && k.kind == KeyEventKind::Press
                && (matches!(k.code, KeyCode::Char('q') | KeyCode::Esc)
                    || (k.code == KeyCode::Char('c') && k.modifiers.contains(KeyModifiers::CONTROL)))
            {
                return Ok(());
            }
        }
    }

    fn draw(&self, f: &mut Frame, s: &Snapshot) {
        let [header, status, charts, system, table, footer] = Layout::vertical([
            Constraint::Length(1),
            Constraint::Length(5),
            Constraint::Min(8),
            Constraint::Length(7),
            Constraint::Min(5),
            Constraint::Length(1),
        ])
        .areas(f.area());
        self.draw_header(f, header);
        self.draw_status(f, status, s);
        let [gen_area, pp_area] = Layout::horizontal([Constraint::Percentage(50); 2]).areas(charts);
        self.draw_throughput(f, gen_area, s, true);
        self.draw_throughput(f, pp_area, s, false);
        let [gpu_area, sys_area] = Layout::horizontal([Constraint::Percentage(50); 2]).areas(system);
        self.draw_gpu_chart(f, gpu_area, s);
        self.draw_system(f, sys_area, s);
        self.draw_requests(f, table, s);
        self.draw_footer(f, footer, s);
    }

    fn draw_header(&self, f: &mut Frame, area: Rect) {
        let i = &self.info;
        let sep = Span::styled(" · ", Style::new().fg(DIM));
        let line = Line::from(vec![
            Span::styled(" SenseNova Loop ", Style::new().bold().fg(Color::Black).bg(GEN)),
            Span::raw(" "),
            Span::styled(i.url.clone(), Style::new().bold()),
            sep.clone(),
            Span::raw(format!("model {}", i.model_id)),
            sep.clone(),
            Span::raw(i.quant.clone()),
            sep.clone(),
            Span::raw(format!("{} ({:?})", i.gpu.name, i.gpu.backend)),
            sep.clone(),
            Span::raw(format!("up {}", fmt_dur(self.stats.start.elapsed().as_secs_f64()))),
        ]);
        f.render_widget(Paragraph::new(line), area);
    }

    fn draw_status(&self, f: &mut Frame, area: Rect, s: &Snapshot) {
        let block = Block::bordered().title(" Status ");
        let inner = block.inner(area);
        f.render_widget(block, area);
        let [line_area, progress_area, ctx_area] = Layout::vertical([Constraint::Length(1); 3]).areas(inner);

        let queued = self.stats.queued.load(Ordering::Relaxed);
        let mut spans = Vec::new();
        let (label, ratio, gauge_color) = match &s.current {
            None => {
                spans.push(Span::styled("● Idle", Style::new().fg(Color::Green).bold()));
                spans.push(Span::styled("  waiting for requests", Style::new().fg(DIM)));
                (format!("{} requests served", s.totals.requests), 0.0, DIM)
            }
            Some(c) => {
                let (state, color) = match c.phase {
                    Phase::Prefill { .. } => ("● Reading prompt", PREFILL),
                    Phase::Decode { .. } => ("● Generating", GEN),
                };
                spans.push(Span::styled(state, Style::new().fg(color).bold()));
                spans.push(Span::raw(format!(
                    "  request #{}  prompt {} tok ({} cached)  elapsed {}",
                    c.id,
                    c.prompt,
                    c.cached,
                    fmt_dur(c.started.elapsed().as_secs_f64())
                )));
                match c.phase {
                    Phase::Prefill { done, total } => (
                        format!("prompt {done} / {total} tok"),
                        done as f64 / total.max(1) as f64,
                        PREFILL,
                    ),
                    Phase::Decode { generated, max } => {
                        spans.push(Span::styled(format!("  {:.1} tok/s", c.rate), Style::new().fg(GEN).bold()));
                        (format!("generated {generated} / {max} tok"), generated as f64 / max.max(1) as f64, GEN)
                    }
                }
            }
        };
        if queued > 0 {
            spans.push(Span::styled(format!("  +{queued} queued"), Style::new().fg(Color::Yellow).bold()));
        }
        f.render_widget(Paragraph::new(Line::from(spans)), line_area);
        labeled_gauge(f, progress_area, "Progress", &label, ratio, gauge_color);
        let ctx = self.info.ctx;
        labeled_gauge(
            f,
            ctx_area,
            "Context",
            &format!("{} / {} tok", s.context_used, ctx),
            s.context_used as f64 / ctx as f64,
            if s.context_used * 10 > ctx * 9 { Color::Red } else { Color::Yellow },
        );
    }

    /// Seconds of history that fit the chart at one braille dot column per second.
    fn window(area: Rect) -> u64 {
        ((area.width.saturating_sub(10) as u64) * 2).clamp(60, HISTORY_SECS)
    }

    fn draw_throughput(&self, f: &mut Frame, area: Rect, s: &Snapshot, decode: bool) {
        let window = Self::window(area);
        let now = self.stats.start.elapsed().as_secs();
        let (mut tokens, mut secs) = (0usize, 0f64);
        let points: Vec<(u64, f64)> = s
            .buckets
            .iter()
            .filter(|(t, _)| t + window >= now)
            .filter_map(|(t, b)| {
                let (n, d) = if decode { (b.decode_tokens, b.decode_secs) } else { (b.prefill_tokens, b.prefill_secs) };
                tokens += n;
                secs += d;
                (n > 0 && d > 0.0).then(|| (*t, n as f64 / d))
            })
            .collect();
        let segments = segments(&points, now);
        let max = points.iter().map(|p| p.1).fold(0.0, f64::max);
        let (name, color) = if decode { ("Generation", GEN) } else { ("Prompt processing", PREFILL) };
        let now_rate = match (&s.current, decode) {
            (Some(c), true) if matches!(c.phase, Phase::Decode { .. }) => Some(c.rate),
            _ => points.last().filter(|(t, _)| t + 2 >= now).map(|p| p.1),
        };
        let mut title = vec![Span::styled(format!(" {name} tok/s "), Style::new().fg(color).bold())];
        if let Some(r) = now_rate {
            title.push(Span::styled(format!("now {r:.1} "), Style::new().bold()));
        }
        if secs > 0.0 {
            title.push(Span::styled(format!("avg {:.1} ", tokens as f64 / secs), Style::new().fg(DIM)));
        }
        let datasets = segments
            .iter()
            .map(|seg| Dataset::default().marker(Marker::Braille).graph_type(GraphType::Line).style(Style::new().fg(color)).data(seg))
            .collect();
        let chart = time_chart(datasets, window, nice_max(max), Line::from(title));
        f.render_widget(chart, area);
    }

    fn draw_gpu_chart(&self, f: &mut Frame, area: Rect, s: &Snapshot) {
        let window = Self::window(area);
        let now = self.stats.start.elapsed().as_secs();
        let recent = || self.hw_hist.iter().filter(|(t, _)| t + window >= now);
        let mut series: Vec<(&str, Color, Vec<(u64, f64)>)> = Vec::new();
        if self.hw.nvml.is_some() {
            let load = recent().filter_map(|(t, h)| h.gpu_load.map(|g| (*t, g as f64))).collect();
            let vram = recent()
                .filter_map(|(t, h)| Some((*t, h.vram_used? as f64 / h.vram_total?.max(1) as f64 * 100.0)))
                .collect();
            series.push(("GPU load %", GPU_LOAD, load));
            series.push(("VRAM %", VRAM, vram));
        } else {
            // Without device telemetry, show how busy the engine kept the GPU.
            let busy = s
                .buckets
                .iter()
                .filter(|(t, _)| t + window >= now)
                .map(|(t, b)| (*t, ((b.prefill_secs + b.decode_secs) * 100.0).min(100.0)))
                .collect();
            series.push(("engine busy %", GPU_LOAD, busy));
        }
        let segs: Vec<(usize, Vec<(f64, f64)>)> =
            series.iter().enumerate().flat_map(|(i, (_, _, pts))| segments(pts, now).into_iter().map(move |s| (i, s))).collect();
        let datasets = segs
            .iter()
            .map(|(i, seg)| {
                Dataset::default().marker(Marker::Braille).graph_type(GraphType::Line).style(Style::new().fg(series[*i].1)).data(seg)
            })
            .collect();
        let mut title = vec![Span::styled(" GPU ", Style::new().bold())];
        for (name, color, pts) in &series {
            let v = pts.last().map_or(String::from("-"), |p| format!("{:.0}", p.1));
            title.push(Span::styled(format!("{name} {v} "), Style::new().fg(*color).bold()));
        }
        f.render_widget(time_chart(datasets, window, 100.0, Line::from(title)), area);
    }

    fn draw_system(&self, f: &mut Frame, area: Rect, s: &Snapshot) {
        let block = Block::bordered().title(" System ");
        let inner = block.inner(area);
        f.render_widget(block, area);
        let rows: [Rect; 5] = Layout::vertical([Constraint::Length(1); 5]).areas(inner);
        let h = self.hw_hist.back().map(|(_, h)| *h).unwrap_or_default();
        let model = gib(self.info.model_bytes);

        match (h.vram_used, h.vram_total) {
            (Some(used), Some(total)) => {
                let full = total.saturating_sub(used) < 300 << 20;
                let mut label = format!("{:.1} / {:.1} GiB  (this model {model:.1})", gib(used), gib(total));
                if full {
                    label.push_str("  FULL: may be paging to system RAM");
                }
                labeled_gauge(f, rows[0], "VRAM", &label, used as f64 / total as f64, if full { Color::Red } else { VRAM });
            }
            _ => {
                let line = Line::from(vec![
                    Span::styled(format!("{:<9}", "VRAM"), Style::new().fg(DIM)),
                    Span::raw(format!("this model {model:.1} GiB")),
                    Span::styled("  (device telemetry needs an NVIDIA GPU)", Style::new().fg(DIM)),
                ]);
                f.render_widget(Paragraph::new(line), rows[0]);
            }
        }
        let mut gpu = vec![Span::styled(format!("{:<9}", "GPU"), Style::new().fg(DIM))];
        if let Some(l) = h.gpu_load {
            gpu.push(Span::styled(format!("{l}% load"), Style::new().fg(GPU_LOAD).bold()));
        }
        if let Some(t) = h.temp {
            let c = if t >= 85 { Color::Red } else if t >= 75 { Color::Yellow } else { Color::Reset };
            gpu.push(Span::styled(format!("   {t}°C"), Style::new().fg(c)));
        }
        if let Some(p) = h.power_w {
            gpu.push(Span::raw(format!("   {p:.0} W")));
        }
        if gpu.len() > 1 {
            f.render_widget(Paragraph::new(Line::from(gpu)), rows[1]);
        }
        let proc_line = Line::from(vec![
            Span::styled(format!("{:<9}", "Process"), Style::new().fg(DIM)),
            Span::raw(format!("CPU {:.0}%   RAM {:.2} GiB", h.cpu, gib(h.rss))),
        ]);
        f.render_widget(Paragraph::new(proc_line), rows[2]);

        let t = &s.totals;
        let cache_pct = if t.prompt_tokens > 0 { t.cached_tokens as f64 / t.prompt_tokens as f64 * 100.0 } else { 0.0 };
        let mut req = vec![
            Span::styled(format!("{:<9}", "Requests"), Style::new().fg(DIM)),
            Span::styled(t.requests.to_string(), Style::new().bold()),
            Span::raw(format!(" served   busy {}", fmt_dur(t.busy_secs))),
        ];
        if t.errors > 0 {
            req.push(Span::styled(format!("   {} failed", t.errors), Style::new().fg(Color::Red).bold()));
        }
        f.render_widget(Paragraph::new(Line::from(req)), rows[3]);
        let tokens = Line::from(vec![
            Span::styled(format!("{:<9}", "Tokens"), Style::new().fg(DIM)),
            Span::styled(fmt_count(t.prompt_tokens), Style::new().fg(PREFILL).bold()),
            Span::raw(format!(" in ({cache_pct:.0}% from cache)   ")),
            Span::styled(fmt_count(t.generated_tokens), Style::new().fg(GEN).bold()),
            Span::raw(" out"),
        ]);
        f.render_widget(Paragraph::new(tokens), rows[4]);
    }

    fn draw_requests(&self, f: &mut Frame, area: Rect, s: &Snapshot) {
        let now = self.stats.start.elapsed().as_secs_f64();
        let header = Row::new(["#", "finished", "prompt", "cached", "output", "prompt tok/s", "gen tok/s", "duration", "result"])
            .style(Style::new().fg(DIM).add_modifier(Modifier::BOLD));
        let rows: Vec<Row> = s
            .recent
            .iter()
            .map(|r| {
                let new = r.prompt - r.cached;
                let pp = if new > 0 && r.prefill_secs > 0.0 { format!("{:.0}", new as f64 / r.prefill_secs) } else { "-".into() };
                let tg = if r.generated > 0 && r.decode_secs > 0.0 {
                    format!("{:.1}", r.generated as f64 / r.decode_secs)
                } else {
                    "-".into()
                };
                let color = match r.outcome {
                    "error" => Color::Red,
                    "cancelled" | "length" => Color::Yellow,
                    _ => Color::Green,
                };
                Row::new([
                    Cell::from(r.id.to_string()),
                    Cell::from(format!("{} ago", fmt_dur(now - r.at))),
                    Cell::from(r.prompt.to_string()),
                    Cell::from(r.cached.to_string()),
                    Cell::from(r.generated.to_string()),
                    Cell::from(pp).fg(PREFILL),
                    Cell::from(tg).fg(GEN),
                    Cell::from(fmt_dur(r.prefill_secs + r.decode_secs)),
                    Cell::from(r.outcome).fg(color),
                ])
            })
            .collect();
        let widths = [
            Constraint::Length(5),
            Constraint::Length(10),
            Constraint::Length(8),
            Constraint::Length(8),
            Constraint::Length(8),
            Constraint::Length(13),
            Constraint::Length(10),
            Constraint::Length(9),
            Constraint::Min(9),
        ];
        let table = Table::new(rows, widths).header(header).block(Block::bordered().title(" Recent requests "));
        f.render_widget(table, area);
    }

    fn draw_footer(&self, f: &mut Frame, area: Rect, s: &Snapshot) {
        let mut spans = vec![Span::styled(" q ", Style::new().fg(Color::Black).bg(DIM)), Span::styled(" quit ", Style::new().fg(DIM))];
        if let Some((at, msg)) = s.logs.back() {
            let now = self.stats.start.elapsed().as_secs_f64();
            let color = if msg.starts_with("error") { Color::Red } else { Color::Yellow };
            spans.push(Span::styled(format!("  {} ago: {msg}", fmt_dur(now - at)), Style::new().fg(color)));
        }
        f.render_widget(Paragraph::new(Line::from(spans)), area);
    }
}

fn labeled_gauge(f: &mut Frame, area: Rect, name: &str, label: &str, ratio: f64, color: Color) {
    let [name_area, gauge_area] = Layout::horizontal([Constraint::Length(9), Constraint::Min(1)]).areas(area);
    f.render_widget(Paragraph::new(name).fg(DIM), name_area);
    let gauge = Gauge::default()
        .gauge_style(Style::new().fg(color).bg(Color::Indexed(236)))
        .ratio(ratio.clamp(0.0, 1.0))
        .label(Span::styled(label.to_string(), Style::new().fg(Color::White).bold()));
    f.render_widget(gauge, gauge_area);
}

/// Split per-second points into runs of consecutive seconds, as (seconds ago, value),
/// so idle gaps are not bridged by a line.
fn segments(points: &[(u64, f64)], now: u64) -> Vec<Vec<(f64, f64)>> {
    let mut out: Vec<Vec<(f64, f64)>> = Vec::new();
    let mut prev: Option<u64> = None;
    for &(t, v) in points {
        let x = t as f64 - now as f64;
        match out.last_mut() {
            Some(seg) if prev.is_some_and(|p| t <= p + 2) => seg.push((x, v)),
            _ => out.push(vec![(x, v)]),
        }
        prev = Some(t);
    }
    for seg in &mut out {
        if seg.len() == 1 {
            // A lone point would not draw as a line.
            let (x, v) = seg[0];
            seg.push((x + 0.6, v));
        }
    }
    out
}

fn time_chart<'a>(datasets: Vec<Dataset<'a>>, window: u64, ymax: f64, title: Line<'a>) -> Chart<'a> {
    let dim = Style::new().fg(DIM);
    Chart::new(datasets)
        .block(Block::bordered().title(title))
        .x_axis(
            Axis::default()
                .style(dim)
                .bounds([-(window as f64), 0.0])
                .labels([Span::raw(format!("-{}", fmt_dur(window as f64))), Span::raw("now")]),
        )
        .y_axis(Axis::default().style(dim).bounds([0.0, ymax]).labels([
            Span::raw("0"),
            Span::raw(fmt_axis(ymax / 2.0)),
            Span::raw(fmt_axis(ymax)),
        ]))
}

/// Round a chart maximum up to 1, 2 or 5 times a power of ten, with some headroom.
fn nice_max(v: f64) -> f64 {
    let v = (v * 1.1).max(1.0);
    let p = 10f64.powf(v.log10().floor());
    [1.0, 2.0, 5.0, 10.0].iter().map(|m| m * p).find(|&m| m >= v).unwrap_or(10.0 * p)
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1u64 << 30) as f64
}

fn fmt_axis(v: f64) -> String {
    if v >= 1000.0 { format!("{:.1}K", v / 1000.0) } else if v >= 10.0 { format!("{v:.0}") } else { format!("{v:.1}") }
}

fn fmt_count(n: u64) -> String {
    match n {
        0..1_000 => n.to_string(),
        1_000..1_000_000 => format!("{:.1}K", n as f64 / 1e3),
        _ => format!("{:.2}M", n as f64 / 1e6),
    }
}

fn fmt_dur(secs: f64) -> String {
    let s = secs.max(0.0);
    if s < 10.0 {
        format!("{s:.1}s")
    } else if s < 60.0 {
        format!("{s:.0}s")
    } else if s < 3600.0 {
        format!("{}m{:02}s", (s / 60.0) as u64, s as u64 % 60)
    } else {
        format!("{}h{:02}m", (s / 3600.0) as u64, (s as u64 % 3600) / 60)
    }
}
