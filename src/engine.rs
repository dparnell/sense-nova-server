//! The SenseNova Loop forward pass on the GPU.
//!
//! The physical decoder layers are recorded in execution-plan order (the looped
//! block appears `loop_count` times). Shared weights are bound to every visit,
//! while each visit owns its own KV cache / recurrent state slot.

use std::path::Path;
use std::sync::Arc;

use anyhow::{Result, bail};
use bytemuck::{Pod, Zeroable};

use crate::config::ModelConfig;
use crate::gpu::{Gpu, Kernel, Kernels, MatKind, Res, bind_group};
use crate::weights::{DType, WeightFile, dequant_q8_row};

/// Tokens processed per GPU submission during prefill.
pub const MAX_M: usize = 512;
/// Number of recurrent-state snapshots kept for prefix reuse across requests.
pub const SNAPSHOTS: usize = 4;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct StepU {
    m: u32,
    pos0: u32,
    parts: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MatP {
    n: u32,
    k: u32,
    x_off: u32,
    x_stride: u32,
    y_off: u32,
    y_stride: u32,
    n_off: u32,
    m_one: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LayerP {
    h: u32,
    mode: u32,
    last_row: u32,
    eps: f32,
    weight_one: f32,
    iterations: u32,
    inter: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AttnP {
    n_heads: u32,
    n_kv: u32,
    rot_dim: u32,
    proj_stride: u32,
    rope_theta: f32,
    eps: f32,
    scale: f32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LinP {
    conv_dim: u32,
    proj_stride: u32,
    n_k: u32,
    n_v: u32,
    eps: f32,
    _pad: [u32; 3],
}

/// A quantized matrix on the GPU, split into row chunks that fit one binding.
struct QWeight {
    dtype: DType,
    k: usize,
    chunks: Vec<(wgpu::Buffer, wgpu::Buffer, usize, usize)>, // (quants, scales, rows, first row)
}

#[derive(Clone, Copy, Debug)]
enum K {
    BranchNorm,
    MhcFinish,
    Merge,
    Swiglu,
    Combine,
    Conv,
    Delta,
}

enum Op {
    Mat { dtype: DType, groups: Vec<(wgpu::BindGroup, usize)>, m_one: bool },
    /// mHC router dot products: (9, M) workgroups.
    MhcDots { bg: wgpu::BindGroup },
    /// One workgroup per token (or a single one for the final norm).
    PerToken { kind: K, bg: wgpu::BindGroup, single: bool },
    /// One thread per element of an [M, width] array.
    Elementwise { kind: K, bg: wgpu::BindGroup, width: usize },
    /// Fixed grid, independent of M.
    Fixed { kind: K, bg: wgpu::BindGroup, x: u32 },
    AttnPrep { bg: wgpu::BindGroup, slots: u32 },
    Attend { bg: wgpu::BindGroup, heads: u32 },
}

pub struct Engine {
    pub gpu: Gpu,
    pub cfg: ModelConfig,
    pub ctx: usize,
    pub quant: String,
    kernels: Kernels,
    weights: Arc<WeightFile>,
    step: wgpu::Buffer,
    x: wgpu::Buffer,
    logits: wgpu::Buffer,
    staging: wgpu::Buffer,
    lin_state: wgpu::Buffer,
    conv_state: wgpu::Buffer,
    /// Saved recurrent states, one (linear, conv) pair per snapshot slot.
    snaps: Vec<(wgpu::Buffer, wgpu::Buffer)>,
    ops: Vec<Op>,
    final_ops: Vec<Op>,
    /// Number of tokens currently held in the caches.
    pub pos: usize,
    /// Tokens per prefill submission and ops per GPU submission, smaller on slow GPUs.
    pub prefill_chunk: usize,
    ops_per_submit: usize,
    slow_gpu: bool,
    profile: Option<std::cell::RefCell<std::collections::HashMap<String, (f64, usize)>>>,
    _keep: Vec<wgpu::Buffer>,
}

fn op_label(op: &Op) -> String {
    match op {
        Op::Mat { groups, m_one, .. } => {
            format!("mat n={}{}", groups.iter().map(|g| g.1).sum::<usize>(), if *m_one { " (head)" } else { "" })
        }
        Op::PerToken { kind, .. } | Op::Elementwise { kind, .. } | Op::Fixed { kind, .. } => format!("{kind:?}"),
        Op::AttnPrep { .. } => "AttnPrep".into(),
        Op::MhcDots { .. } => "MhcDots".into(),
        Op::Attend { .. } => "Attend".into(),
    }
}

fn align256(x: u64) -> u64 {
    x.div_ceil(256) * 256
}

impl Engine {
    pub fn load(gpu: Gpu, path: &Path, ctx: usize) -> Result<Self> {
        let wf = Arc::new(WeightFile::open(path)?);
        let cfg = wf.header.config.clone();
        let kernels = Kernels::new(&gpu)?;
        let ctx = ctx.min(cfg.max_position);
        let h = cfg.hidden;
        let mut keep: Vec<wgpu::Buffer> = Vec::new();

        // ---- weights ----
        let max_bind = gpu.max_binding();
        let upload_q = |name: &str| -> Result<QWeight> {
            let info = wf.info(name)?;
            let (q, s) = wf.bytes(name)?;
            let (rows, k) = (info.rows(), info.cols());
            let row_q = q.len() / rows;
            let row_s = s.len() / rows;
            let per = ((max_bind as usize / row_q.max(1)) / 64 * 64).max(64).min(rows);
            let mut chunks = Vec::new();
            let mut r0 = 0;
            while r0 < rows {
                let n = per.min(rows - r0);
                let qb = gpu.storage_init(name, &q[r0 * row_q..(r0 + n) * row_q]);
                let sb = gpu.storage_init(name, &s[r0 * row_s..(r0 + n) * row_s]);
                chunks.push((qb, sb, n, r0));
                r0 += n;
            }
            Ok(QWeight { dtype: info.dtype, k, chunks })
        };
        let upload_f = |name: &str| -> Result<wgpu::Buffer> {
            let (q, _) = wf.bytes(name)?;
            Ok(gpu.storage_init(name, q))
        };

        // ---- activations ----
        let proj_w = (2 * cfg.intermediate).max(cfg.lin_proj_rows()).max(cfg.attn_proj_rows());
        let act_w = cfg.intermediate.max(cfg.conv_dim());
        let attn_w = cfg.q_dim().max(cfg.lin_value_dim());
        let f = |n: usize| (n * 4) as u64;
        let step = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("step"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let x = gpu.storage("x", f(MAX_M * 2 * h));
        let xn = gpu.storage("xn", f(MAX_M * h));
        let proj = gpu.storage("proj", f(MAX_M * proj_w));
        let act = gpu.storage("act", f(MAX_M * act_w));
        let attn_out = gpu.storage("attn_out", f(MAX_M * attn_w));
        let branch = gpu.storage("branch", f(MAX_M * h));
        let qbuf = gpu.storage("qbuf", f(MAX_M * cfg.q_dim()));
        let coef = gpu.storage("coef", f(MAX_M * 8));
        let partial = gpu.storage("partial", f(cfg.n_heads * MAX_M * (cfg.head_dim + 2)));
        let logits = gpu.storage("logits", f(cfg.vocab));
        let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("logits staging"),
            size: f(cfg.vocab),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let dummy = || gpu.storage("dummy", 16);

        // ---- caches ----
        let plan = cfg.execution_plan();
        let n_full = plan.iter().filter(|&&l| cfg.full_attention[l]).count();
        let n_lin = plan.len() - n_full;
        let kv_bytes = (ctx * cfg.kv_dim() * 2) as u64;
        if kv_bytes > max_bind {
            bail!("context {ctx} is too large for this GPU's binding limit; lower --ctx");
        }
        let state_bytes = f(cfg.lin_v_heads * cfg.lin_k_dim * cfg.lin_v_dim) as u64;
        let conv_bytes = align256(f(3 * cfg.conv_dim()));
        let lin_state = gpu.storage("lin_state", state_bytes * n_lin as u64);
        let conv_state = gpu.storage("conv_state", conv_bytes * n_lin as u64);
        let snaps = (0..SNAPSHOTS)
            .map(|_| (gpu.storage("lin_snap", state_bytes * n_lin as u64), gpu.storage("conv_snap", conv_bytes * n_lin as u64)))
            .collect();

        // ---- per-layer weights ----
        struct Layer {
            in_ln: wgpu::Buffer,
            post_ln: wgpu::Buffer,
            mix: Option<(QWeight, QWeight)>, // (in_proj or qkv, out_proj or o_proj)
            small: wgpu::Buffer,             // qk_norm or lin params
            gate_up: QWeight,
            down: QWeight,
            mhc: Option<(wgpu::Buffer, wgpu::Buffer)>,
        }
        let mut layers = Vec::new();
        for i in 0..cfg.n_layers {
            let full = cfg.full_attention[i];
            let mix = if full {
                (upload_q(&format!("l{i}.qkv"))?, upload_q(&format!("l{i}.o_proj"))?)
            } else {
                (upload_q(&format!("l{i}.in_proj"))?, upload_q(&format!("l{i}.out_proj"))?)
            };
            layers.push(Layer {
                in_ln: upload_f(&format!("l{i}.in_ln"))?,
                post_ln: upload_f(&format!("l{i}.post_ln"))?,
                small: upload_f(&if full { format!("l{i}.qk_norm") } else { format!("l{i}.lin") })?,
                mix: Some(mix),
                gate_up: upload_q(&format!("l{i}.gate_up"))?,
                down: upload_q(&format!("l{i}.down"))?,
                mhc: if cfg.is_mhc(i) {
                    Some((upload_f(&format!("l{i}.mhc_attn"))?, upload_f(&format!("l{i}.mhc_ffn"))?))
                } else {
                    None
                },
            });
        }
        let lm_head = upload_q("lm_head")?;
        let final_norm = upload_f("norm")?;

        // ---- op graph ----
        let k = &kernels;
        let mat = |w: &QWeight, x: &wgpu::Buffer, x_stride: usize, y: &wgpu::Buffer, y_stride: usize, m_one: bool| -> Op {
            let groups = w
                .chunks
                .iter()
                .map(|(qb, sb, rows, r0)| {
                    let p = gpu.uniform(
                        "mat",
                        &MatP {
                            n: *rows as u32,
                            k: w.k as u32,
                            x_off: 0,
                            x_stride: x_stride as u32,
                            y_off: 0,
                            y_stride: y_stride as u32,
                            n_off: *r0 as u32,
                            m_one: m_one as u32,
                        },
                    );
                    let res = [Res::Whole(&step), Res::Whole(&p), Res::Whole(x), Res::Whole(qb), Res::Whole(sb), Res::Whole(y)];
                    (bind_group(&gpu, k.matmul(w.dtype, MatKind::Tiled), "mat", &res), *rows)
                })
                .collect();
            Op::Mat { dtype: w.dtype, groups, m_one }
        };
        let layer_p = |mode: u32, last_row: bool| LayerP {
            h: h as u32,
            mode,
            last_row: last_row as u32,
            eps: cfg.eps,
            weight_one: 1.0,
            iterations: cfg.mhc.as_ref().map_or(0, |m| m.iterations as u32),
            inter: cfg.intermediate as u32,
            _pad: 0,
        };
        // layers.wgsl bindings: x, coef, w, out
        let layer_bg = |kern: &Kernel, p: LayerP, xb: &wgpu::Buffer, w: &wgpu::Buffer, out: &wgpu::Buffer, coef_b: &wgpu::Buffer| {
            let pb = gpu.uniform("layer", &p);
            bind_group(&gpu, kern, "layer", &[Res::Whole(&step), Res::Whole(&pb), Res::Whole(xb), Res::Whole(coef_b), Res::Whole(w), Res::Whole(out)])
        };

        let mut ops = Vec::new();
        let (mut lin_slot, mut full_slot) = (0usize, 0usize);
        let mhc_start = cfg.mhc.as_ref().map(|m| m.start);
        let mhc_end = cfg.mhc.as_ref().map(|m| m.end);
        let d_w = dummy();
        let mhc_raw = gpu.storage("mhc_raw", f(MAX_M * 9));
        for &li in &plan {
            let l = &layers[li];
            let is_mhc = l.mhc.is_some();
            // Token mixer input.
            if is_mhc && Some(li) == mhc_start {
                ops.push(Op::Elementwise { kind: K::Merge, bg: layer_bg(&k.merge, layer_p(3, false), &x, &d_w, &branch, &coef), width: h });
            }
            let push_norm = |ops: &mut Vec<Op>, router: Option<&wgpu::Buffer>, norm_w: &wgpu::Buffer| {
                if let Some(rw) = router {
                    ops.push(Op::MhcDots { bg: layer_bg(&k.mhc_dots, layer_p(0, false), &x, rw, &mhc_raw, &coef) });
                    ops.push(Op::Elementwise { kind: K::MhcFinish, bg: layer_bg(&k.mhc_finish, layer_p(0, false), &x, rw, &mhc_raw, &coef), width: 1 });
                }
                let mode = router.is_some() as u32;
                ops.push(Op::PerToken { kind: K::BranchNorm, bg: layer_bg(&k.branch_norm, layer_p(mode, false), &x, norm_w, &xn, &coef), single: false });
            };
            push_norm(&mut ops, l.mhc.as_ref().map(|m| &m.0), &l.in_ln);

            let (inp, outp) = l.mix.as_ref().unwrap();
            if cfg.full_attention[li] {
                let stride = cfg.attn_proj_rows();
                ops.push(mat(inp, &xn, h, &proj, stride, false));
                let kc = gpu.storage("kcache", kv_bytes);
                let vc = gpu.storage("vcache", kv_bytes);
                let ap = gpu.uniform(
                    "attn",
                    &AttnP {
                        n_heads: cfg.n_heads as u32,
                        n_kv: cfg.n_kv_heads as u32,
                        rot_dim: cfg.rot_dim as u32,
                        proj_stride: stride as u32,
                        rope_theta: cfg.rope_theta,
                        eps: cfg.eps,
                        scale: 1.0 / (cfg.head_dim as f32).sqrt(),
                        _pad: 0,
                    },
                );
                let res = |kern: &Kernel| {
                    bind_group(&gpu, kern, "attn", &[
                        Res::Whole(&step), Res::Whole(&ap), Res::Whole(&proj), Res::Whole(&qbuf), Res::Whole(&kc),
                        Res::Whole(&vc), Res::Whole(&l.small), Res::Whole(&attn_out), Res::Whole(&partial),
                    ])
                };
                ops.push(Op::AttnPrep { bg: res(&k.attn_prep), slots: (cfg.n_heads + 2 * cfg.n_kv_heads) as u32 });
                ops.push(Op::Attend { bg: res(&k.attend), heads: cfg.n_kv_heads as u32 });
                // `combine` shares the bind group layout; reuse a fresh group for it.
                ops.push(Op::PerToken { kind: K::Combine, bg: res(&k.combine), single: false });
                keep.push(kc);
                keep.push(vc);
                keep.push(ap);
                ops.push(mat(outp, &attn_out, cfg.q_dim(), &branch, h, false));
                full_slot += 1;
            } else {
                let stride = cfg.lin_proj_rows();
                ops.push(mat(inp, &xn, h, &proj, stride, false));
                let lp = gpu.uniform(
                    "lin",
                    &LinP {
                        conv_dim: cfg.conv_dim() as u32,
                        proj_stride: stride as u32,
                        n_k: cfg.lin_k_heads as u32,
                        n_v: cfg.lin_v_heads as u32,
                        eps: cfg.eps,
                        _pad: [0; 3],
                    },
                );
                let res = |kern: &Kernel| {
                    bind_group(&gpu, kern, "lin", &[
                        Res::Whole(&step), Res::Whole(&lp), Res::Whole(&proj), Res::Whole(&act),
                        Res::Range(&conv_state, conv_bytes * lin_slot as u64, f(3 * cfg.conv_dim())),
                        Res::Range(&lin_state, state_bytes * lin_slot as u64, state_bytes),
                        Res::Whole(&l.small), Res::Whole(&attn_out),
                    ])
                };
                ops.push(Op::Fixed { kind: K::Conv, bg: res(&k.conv), x: cfg.conv_dim().div_ceil(256) as u32 });
                ops.push(Op::Fixed { kind: K::Delta, bg: res(&k.delta), x: cfg.lin_v_heads as u32 });
                keep.push(lp);
                ops.push(mat(outp, &attn_out, cfg.lin_value_dim(), &branch, h, false));
                lin_slot += 1;
            }
            let merge_mode = if is_mhc { 1 } else { 0 };
            ops.push(Op::Elementwise { kind: K::Merge, bg: layer_bg(&k.merge, layer_p(merge_mode, false), &x, &d_w, &branch, &coef), width: h });

            // MLP.
            push_norm(&mut ops, l.mhc.as_ref().map(|m| &m.1), &l.post_ln);
            ops.push(mat(&l.gate_up, &xn, h, &proj, 2 * cfg.intermediate, false));
            ops.push(Op::Elementwise { kind: K::Swiglu, bg: layer_bg(&k.swiglu, layer_p(0, false), &proj, &d_w, &act, &coef), width: cfg.intermediate });
            ops.push(mat(&l.down, &act, cfg.intermediate, &branch, h, false));
            let mode = if !is_mhc { 0 } else if Some(li + 1) == mhc_end { 2 } else { 1 };
            ops.push(Op::Elementwise { kind: K::Merge, bg: layer_bg(&k.merge, layer_p(mode, false), &x, &d_w, &branch, &coef), width: h });
        }
        let _ = full_slot;

        let mut final_p = layer_p(0, true);
        final_p.weight_one = 1.0;
        let final_ops = vec![
            Op::PerToken { kind: K::BranchNorm, bg: layer_bg(&k.branch_norm, final_p, &x, &final_norm, &xn, &coef), single: true },
            mat(&lm_head, &xn, h, &logits, cfg.vocab, true),
        ];

        // Keep every GPU resource referenced by the bind groups alive.
        keep.extend([xn, proj, act, attn_out, branch, qbuf, coef, partial, final_norm, d_w, mhc_raw]);
        for l in layers {
            keep.extend([l.in_ln, l.post_ln, l.small]);
            let (a, b) = l.mix.unwrap();
            for w in [a, b, l.gate_up, l.down] {
                keep.extend(w.chunks.into_iter().flat_map(|(q, s, _, _)| [q, s]));
            }
            if let Some((a, b)) = l.mhc {
                keep.extend([a, b]);
            }
        }
        keep.extend(lm_head.chunks.into_iter().flat_map(|(q, s, _, _)| [q, s]));

        let quant = wf.header.quant.clone();
        let slow = !matches!(gpu.info.device_type, wgpu::DeviceType::DiscreteGpu)
            || std::env::var_os("SENSENOVA_SMALL_SUBMITS").is_some();
        let (prefill_chunk, ops_per_submit) = if slow { (64, 8) } else { (MAX_M, 128) };
        let mut engine = Engine {
            gpu,
            cfg,
            ctx,
            quant,
            kernels,
            // Re-map the file so the pages read during upload leave this
            // process's resident memory; afterwards only embedding rows are touched.
            weights: {
                drop(wf);
                Arc::new(WeightFile::open(path)?)
            },
            step,
            x,
            logits,
            staging,
            lin_state,
            conv_state,
            snaps,
            ops,
            final_ops,
            pos: 0,
            prefill_chunk,
            ops_per_submit,
            slow_gpu: slow,
            profile: std::env::var_os("SENSENOVA_PROFILE").map(|_| Default::default()),
            _keep: keep,
        };
        engine.reset();
        Ok(engine)
    }

    /// Clear all recurrent state; the KV caches are simply overwritten from position 0.
    pub fn reset(&mut self) {
        let mut enc = self.gpu.device.create_command_encoder(&Default::default());
        enc.clear_buffer(&self.lin_state, 0, None);
        enc.clear_buffer(&self.conv_state, 0, None);
        self.gpu.queue.submit([enc.finish()]);
        self.pos = 0;
    }

    /// Save the recurrent state in `slot` so that `restore` can rewind to the current position.
    pub fn snapshot(&mut self, slot: usize) {
        let (lin, conv) = &self.snaps[slot];
        let mut enc = self.gpu.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&self.lin_state, 0, lin, 0, self.lin_state.size());
        enc.copy_buffer_to_buffer(&self.conv_state, 0, conv, 0, self.conv_state.size());
        self.gpu.queue.submit([enc.finish()]);
    }

    /// Rewind to the snapshot in `slot`, taken at `pos`. KV entries before
    /// `pos` are still valid because later tokens only write positions >= `pos`.
    pub fn restore(&mut self, slot: usize, pos: usize) {
        let (lin, conv) = &self.snaps[slot];
        let mut enc = self.gpu.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(lin, 0, &self.lin_state, 0, self.lin_state.size());
        enc.copy_buffer_to_buffer(conv, 0, &self.conv_state, 0, self.conv_state.size());
        self.gpu.queue.submit([enc.finish()]);
        self.pos = pos;
    }

    fn embed(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        let h = self.cfg.hidden;
        let (q, s) = self.weights.bytes("embed")?;
        let scales: &[f32] = bytemuck::cast_slice(s);
        let mut out = vec![0f32; tokens.len() * 2 * h];
        for (i, &t) in tokens.iter().enumerate() {
            let t = t as usize;
            if t >= self.cfg.vocab {
                bail!("token id {t} out of range");
            }
            dequant_q8_row(&q[t * h..(t + 1) * h], &scales[t * h / 32..(t + 1) * h / 32], &mut out[i * 2 * h..i * 2 * h + h]);
        }
        Ok(out)
    }

    /// Feed `tokens` at the current position. Returns logits for the last token when asked.
    pub fn forward(&mut self, tokens: &[u32], want_logits: bool) -> Result<Option<Vec<f32>>> {
        if tokens.is_empty() {
            bail!("forward called with no tokens");
        }
        if self.pos + tokens.len() > self.ctx {
            bail!("context length exceeded ({} > {})", self.pos + tokens.len(), self.ctx);
        }
        let n_chunks = tokens.len().div_ceil(self.prefill_chunk);
        for (ci, chunk) in tokens.chunks(self.prefill_chunk).enumerate() {
            let last = ci + 1 == n_chunks;
            let m = chunk.len();
            let total = self.pos + m;
            let parts = if m <= 4 { total.div_ceil(256).clamp(1, 64) } else { 1 };
            self.gpu.queue.write_buffer(&self.step, 0, bytemuck::bytes_of(&StepU {
                m: m as u32,
                pos0: self.pos as u32,
                parts: parts as u32,
                _pad: 0,
            }));
            let t_start = std::time::Instant::now();
            let emb = self.embed(chunk)?;
            self.gpu.queue.write_buffer(&self.x, 0, bytemuck::cast_slice(&emb));

            if let Some(prof) = &self.profile {
                // Profiling: run each op on its own and wait for it.
                let ops = self.ops.iter().chain(if last && want_logits { &self.final_ops[..] } else { &[] });
                // Time each op run once and 21 times; the difference cancels submit overhead.
                // Repeated ops corrupt the activations, so profiling output is for timing only.
                for op in ops {
                    let timed = |reps: usize| {
                        let t = std::time::Instant::now();
                        let mut enc = self.gpu.device.create_command_encoder(&Default::default());
                        {
                            let mut pass = enc.begin_compute_pass(&Default::default());
                            for _ in 0..reps {
                                self.record(&mut pass, op, m, parts);
                            }
                        }
                        self.gpu.queue.submit([enc.finish()]);
                        let _ = self.gpu.device.poll(wgpu::PollType::wait_indefinitely());
                        t.elapsed().as_secs_f64()
                    };
                    let one = timed(1);
                    let many = timed(21);
                    let mut p = prof.borrow_mut();
                    let e = p.entry(op_label(op)).or_insert((0.0, 0));
                    e.0 += ((many - one) / 20.0).max(0.0);
                    e.1 += 1;
                }
            }
            let mut enc = self.gpu.device.create_command_encoder(&Default::default());
            if self.profile.is_none() {
                // Split the pass into several submissions so no single one runs
                // long enough to trip the OS GPU watchdog on slow devices.
                // A single decode step is short on discrete GPUs, and extra submissions
                // leave the GPU idle between them, so only split prefill there.
                let per_submit = if m > 1 || self.slow_gpu { self.ops_per_submit } else { self.ops.len() };
                for (gi, group) in self.ops.chunks(per_submit).enumerate() {
                    if gi > 0 {
                        self.gpu.queue.submit([std::mem::replace(
                            &mut enc,
                            self.gpu.device.create_command_encoder(&Default::default()),
                        )
                        .finish()]);
                    }
                    let mut pass = enc.begin_compute_pass(&Default::default());
                    for op in group {
                        self.record(&mut pass, op, m, parts);
                    }
                }
                if last && want_logits {
                    let mut pass = enc.begin_compute_pass(&Default::default());
                    for op in &self.final_ops {
                        self.record(&mut pass, op, m, parts);
                    }
                }
            }
            if last && want_logits {
                enc.copy_buffer_to_buffer(&self.logits, 0, &self.staging, 0, self.staging.size());
            }
            let t_rec = std::time::Instant::now();
            self.gpu.queue.submit([enc.finish()]);
            if std::env::var_os("SENSENOVA_TIMING").is_some() {
                eprintln!("record {:.2}ms submit {:.2}ms", (t_rec - t_start).as_secs_f64() * 1e3, t_rec.elapsed().as_secs_f64() * 1e3);
            }
            self.pos += m;
        }
        if want_logits {
            return Ok(Some(self.gpu.map_read(&self.staging)?));
        }
        // Keep the CPU from running far ahead of the GPU on long prompts.
        self.gpu.device.poll(wgpu::PollType::wait_indefinitely()).ok();
        Ok(None)
    }

    /// Print accumulated per-op timings (only when SENSENOVA_PROFILE is set).
    pub fn print_profile(&self) {
        let Some(prof) = &self.profile else { return };
        let p = prof.borrow();
        let total: f64 = p.values().map(|v| v.0).sum();
        let mut rows: Vec<_> = p.iter().collect();
        rows.sort_by(|a, b| b.1.0.total_cmp(&a.1.0));
        eprintln!("{:<28} {:>10} {:>8} {:>10}", "op", "total ms", "share", "calls");
        for (k, (t, n)) in rows {
            eprintln!("{k:<28} {:>10.1} {:>7.1}% {n:>10}", t * 1e3, t / total * 100.0);
        }
    }

    fn kernel(&self, k: K) -> &Kernel {
        let ks = &self.kernels;
        match k {
            K::BranchNorm => &ks.branch_norm,
            K::MhcFinish => &ks.mhc_finish,
            K::Merge => &ks.merge,
            K::Swiglu => &ks.swiglu,
            K::Combine => &ks.combine,
            K::Conv => &ks.conv,
            K::Delta => &ks.delta,
        }
    }

    fn record(&self, pass: &mut wgpu::ComputePass, op: &Op, m: usize, parts: usize) {
        let split = |n: u32| if n <= 65535 { (n, 1) } else { (65535, n.div_ceil(65535)) };
        match op {
            Op::Mat { dtype, groups, m_one } => {
                let rows_m = if *m_one { 1 } else { m };
                let kind = MatKind::for_rows(rows_m);
                pass.set_pipeline(&self.kernels.matmul(*dtype, kind).pipeline);
                for (bg, rows) in groups {
                    pass.set_bind_group(0, bg, &[]);
                    if let MatKind::Tiled = kind {
                        pass.dispatch_workgroups(rows.div_ceil(128) as u32, rows_m.div_ceil(128) as u32, 1);
                    } else {
                        let (gx, gy) = split(rows.div_ceil(8) as u32);
                        pass.dispatch_workgroups(gx, gy, 1);
                    }
                }
            }
            Op::MhcDots { bg } => {
                pass.set_pipeline(&self.kernels.mhc_dots.pipeline);
                pass.set_bind_group(0, bg, &[]);
                pass.dispatch_workgroups(9, m as u32, 1);
            }
            Op::PerToken { kind, bg, single } => {
                if matches!(kind, K::Combine) && parts == 1 {
                    return; // split-K combine is only needed with several parts
                }
                pass.set_pipeline(&self.kernel(*kind).pipeline);
                pass.set_bind_group(0, bg, &[]);
                if matches!(kind, K::Combine) {
                    pass.dispatch_workgroups(self.cfg.n_heads as u32, m as u32, 1);
                } else {
                    pass.dispatch_workgroups(if *single { 1 } else { m as u32 }, 1, 1);
                }
            }
            Op::Elementwise { kind, bg, width } => {
                pass.set_pipeline(&self.kernel(*kind).pipeline);
                pass.set_bind_group(0, bg, &[]);
                let (gx, gy) = split((m * width).div_ceil(256) as u32);
                pass.dispatch_workgroups(gx, gy, 1);
            }
            Op::Fixed { kind, bg, x } => {
                pass.set_pipeline(&self.kernel(*kind).pipeline);
                pass.set_bind_group(0, bg, &[]);
                pass.dispatch_workgroups(*x, 1, 1);
            }
            Op::AttnPrep { bg, slots } => {
                pass.set_pipeline(&self.kernels.attn_prep.pipeline);
                pass.set_bind_group(0, bg, &[]);
                pass.dispatch_workgroups(*slots, m as u32, 1);
            }
            Op::Attend { bg, heads } => {
                pass.set_pipeline(&self.kernels.attend.pipeline);
                pass.set_bind_group(0, bg, &[]);
                pass.dispatch_workgroups(*heads, m as u32, parts as u32);
            }
        }
    }
}
