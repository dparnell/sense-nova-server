//! The vision tower: patch embedding, ViT blocks with 2D RoPE, and the 2x2 patch
//! merger that projects into the language model's embedding space.
//!
//! Frames are encoded one at a time (attention never crosses frames), and the
//! merged features are read back to the CPU so the language model can splice
//! them into its input like ordinary token embeddings.

use std::path::Path;

use anyhow::{Result, bail};
use bytemuck::{Pod, Zeroable};

use crate::config::VisionConfig;
use crate::engine::{QWeight, upload_f, upload_q};
use crate::gpu::{Bind, Gpu, Kernel, Kernels, MatKind, Res, bind_group, bind_layout, pipeline, shader};
use crate::weights::WeightFile;

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
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct VisP {
    rows: u32,
    cols: u32,
    mode: u32,
    act: u32,
    eps: f32,
    grid_w: u32,
    heads: u32,
    _pad: u32,
}

const EPS: f32 = 1e-6;
const ACT_GELU_TANH: u32 = 1;
const ACT_GELU_ERF: u32 = 2;

struct Block {
    n1: wgpu::Buffer,
    n2: wgpu::Buffer,
    qkv: QWeight,
    qkv_b: wgpu::Buffer,
    proj: QWeight,
    proj_b: wgpu::Buffer,
    fc1: QWeight,
    fc1_b: wgpu::Buffer,
    fc2: QWeight,
    fc2_b: wgpu::Buffer,
}

struct VisionKernels {
    layer_norm: Kernel,
    epilogue: Kernel,
    rope: Kernel,
    attend: Kernel,
}

pub struct Vision {
    pub cfg: VisionConfig,
    kernels: VisionKernels,
    patch: QWeight,
    patch_b: wgpu::Buffer,
    /// Learned [pos_side * pos_side, hidden] position table, interpolated on the CPU.
    pos: Vec<f32>,
    blocks: Vec<Block>,
    merge_norm: wgpu::Buffer,
    merge_fc1: QWeight,
    merge_fc1_b: wgpu::Buffer,
    merge_fc2: QWeight,
    merge_fc2_b: wgpu::Buffer,
}

/// One recorded GPU dispatch.
struct Dispatch<'a> {
    name: &'static str,
    kernel: &'a Kernel,
    bg: wgpu::BindGroup,
    groups: (u32, u32, u32),
}

fn split(n: u32) -> (u32, u32, u32) {
    if n <= 65535 { (n, 1, 1) } else { (65535, n.div_ceil(65535), 1) }
}

impl Vision {
    pub fn load(gpu: &Gpu, path: &Path) -> Result<Self> {
        let wf = WeightFile::open(path)?;
        let Some(cfg) = wf.header.config.vision.clone() else {
            bail!("{} has no vision configuration", path.display());
        };
        let dev = &gpu.device;
        let module = shader(dev, "vision", include_str!("shaders/vision.wgsl").into());
        let layout = bind_layout(dev, "vision", &[Bind::Uniform, Bind::ReadWrite, Bind::ReadWrite, Bind::Read, Bind::Read]);
        let kernels = VisionKernels {
            layer_norm: pipeline(dev, &module, "layer_norm", &layout),
            epilogue: pipeline(dev, &module, "epilogue", &layout),
            rope: pipeline(dev, &module, "rope", &layout),
            attend: pipeline(dev, &module, "attend", &layout),
        };
        let q = |n: &str| upload_q(gpu, &wf, n);
        let f = |n: &str| upload_f(gpu, &wf, n);
        let blocks = (0..cfg.depth)
            .map(|i| {
                Ok(Block {
                    n1: f(&format!("v{i}.n1"))?,
                    n2: f(&format!("v{i}.n2"))?,
                    qkv: q(&format!("v{i}.qkv"))?,
                    qkv_b: f(&format!("v{i}.qkv_b"))?,
                    proj: q(&format!("v{i}.proj"))?,
                    proj_b: f(&format!("v{i}.proj_b"))?,
                    fc1: q(&format!("v{i}.fc1"))?,
                    fc1_b: f(&format!("v{i}.fc1_b"))?,
                    fc2: q(&format!("v{i}.fc2"))?,
                    fc2_b: f(&format!("v{i}.fc2_b"))?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let (pos_bytes, _) = wf.bytes("v.pos")?;
        Ok(Vision {
            kernels,
            patch: q("v.patch")?,
            patch_b: f("v.patch_b")?,
            pos: bytemuck::cast_slice(pos_bytes).to_vec(),
            blocks,
            merge_norm: f("vm.norm")?,
            merge_fc1: q("vm.fc1")?,
            merge_fc1_b: f("vm.fc1_b")?,
            merge_fc2: q("vm.fc2")?,
            merge_fc2_b: f("vm.fc2_b")?,
            cfg,
        })
    }

    /// Largest number of patches per frame that fits the GPU's binding limit.
    pub fn max_patches(&self, gpu: &Gpu) -> usize {
        let widest = (3 * self.cfg.hidden).max(self.cfg.intermediate_padded).max(self.cfg.patch_dim());
        (gpu.max_binding() as usize / (widest * 4)) / 4 * 4
    }

    /// Bilinearly interpolated position embeddings for an h x w patch grid, in merge order.
    fn pos_embed(&self, h: usize, w: usize) -> Vec<f32> {
        let side = self.cfg.pos_side;
        let hid = self.cfg.hidden;
        let m = self.cfg.merge;
        // torch.linspace(0, side - 1, n) in f32.
        let lin = |n: usize| -> Vec<f32> {
            (0..n)
                .map(|i| if n == 1 { 0.0 } else { (side - 1) as f32 * i as f32 / (n - 1) as f32 })
                .collect()
        };
        let (hs, ws) = (lin(h), lin(w));
        let mut out = vec![0f32; h * w * hid];
        for (token, row_out) in out.chunks_exact_mut(hid).enumerate() {
            let blk = token / (m * m);
            let intra = token % (m * m);
            let r = (blk / (w / m)) * m + intra / m;
            let c = (blk % (w / m)) * m + intra % m;
            let (fh, fw) = (hs[r] as usize, ws[c] as usize);
            let (ch, cw) = ((fh + 1).min(side - 1), (fw + 1).min(side - 1));
            let (dh, dw) = (hs[r] - fh as f32, ws[c] - fw as f32);
            let corners = [
                (fh * side + fw, (1.0 - dh) * (1.0 - dw)),
                (fh * side + cw, (1.0 - dh) * dw),
                (ch * side + fw, dh * (1.0 - dw)),
                (ch * side + cw, dh * dw),
            ];
            for (idx, wt) in corners {
                for (o, v) in row_out.iter_mut().zip(&self.pos[idx * hid..(idx + 1) * hid]) {
                    *o += v * wt;
                }
            }
        }
        out
    }

    /// Encode `frames` frames of `h` x `w` patches (patch vectors in merge order,
    /// frame-major). Returns `frames * h * w / 4` rows of `out_hidden` floats.
    pub fn encode<'a>(&'a self, gpu: &Gpu, mk: &'a Kernels, patches: &[f32], frames: usize, h: usize, w: usize, slow: bool) -> Result<Vec<f32>> {
        let c = &self.cfg;
        let n = h * w;
        let pd = c.patch_dim();
        if h % c.merge != 0 || w % c.merge != 0 || patches.len() != frames * n * pd {
            bail!("vision input has the wrong shape");
        }
        if n > self.max_patches(gpu) {
            bail!("image is too large for this GPU ({n} patches per frame, at most {}); lower the pixel limit", self.max_patches(gpu));
        }
        let hid = c.hidden;
        let nq = n / (c.merge * c.merge);
        let f4 = |x: usize| (x * 4) as u64;

        let pin = gpu.scratch("v_patches", f4(n * pd));
        let x = gpu.scratch("v_x", f4(n * hid));
        let xn = gpu.scratch("v_xn", f4(n * hid));
        let tmp = gpu.scratch("v_tmp", f4(n * hid));
        let att = gpu.scratch("v_att", f4(n * hid));
        // qkv also holds the merger output; mlp also holds the merger's hidden layer.
        let qkv = gpu.scratch("v_qkv", f4(n * 3 * hid).max(f4(nq * c.out_hidden)));
        let mlp = gpu.scratch("v_mlp", f4(n * c.intermediate_padded).max(f4(nq * c.merged_dim())));
        let pos = gpu.scratch("v_pos", f4(n * hid));
        gpu.queue.write_buffer(&pos, 0, bytemuck::cast_slice(&self.pos_embed(h, w)));
        let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("v_out"),
            size: f4(nq * c.out_hidden),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let dummy_rw = gpu.scratch("v_dummy_rw", 16);
        let dummy_r = gpu.scratch("v_dummy_r", 16);
        let step_n = gpu.uniform("v_step", &StepU { m: n as u32, pos0: 0, parts: 1, _pad: 0 });
        let step_q = gpu.uniform("v_step_q", &StepU { m: nq as u32, pos0: 0, parts: 1, _pad: 0 });

        let mut ops: Vec<Dispatch> = Vec::new();
        let k = &self.kernels;
        let mat = |ops: &mut Vec<Dispatch<'a>>, wt: &QWeight, step: &wgpu::Buffer, rows: usize, xb: &wgpu::Buffer, yb: &wgpu::Buffer, y_stride: usize| {
            mat_ops(gpu, mk, ops, wt, step, rows, xb, yb, y_stride)
        };
        let push = |ops: &mut Vec<Dispatch<'a>>, kernel: &'a Kernel, p: VisP, a: &wgpu::Buffer, b: &wgpu::Buffer, wb: &wgpu::Buffer, e: &wgpu::Buffer, groups| {
            let name = if std::ptr::eq(kernel, &k.attend) {
                "attend"
            } else if std::ptr::eq(kernel, &k.layer_norm) {
                "layer_norm"
            } else if std::ptr::eq(kernel, &k.rope) {
                "rope"
            } else {
                "epilogue"
            };
            ops.push(vis_op(gpu, name, kernel, p, [a, b, wb, e], groups))
        };
        let norm = |rows: usize, cols: usize| (VisP { rows: rows as u32, cols: cols as u32, eps: EPS, ..Default::default() }, split(rows as u32));
        let epi = |rows: usize, cols: usize, mode: u32, act: u32| {
            (VisP { rows: rows as u32, cols: cols as u32, mode, act, ..Default::default() }, split((rows * cols).div_ceil(256) as u32))
        };

        // Patch embedding plus bias and interpolated position embedding.
        mat(&mut ops, &self.patch, &step_n, n, &pin, &tmp, hid);
        let (p, g) = epi(n, hid, 2, 0);
        push(&mut ops, &k.epilogue, p, &tmp, &x, &self.patch_b, &pos, g);
        let mut layer_ends = vec![ops.len()];

        for b in &self.blocks {
            let (p, g) = norm(n, hid);
            push(&mut ops, &k.layer_norm, p, &x, &xn, &b.n1, &dummy_r, g);
            mat(&mut ops, &b.qkv, &step_n, n, &xn, &qkv, 3 * hid);
            let (p, g) = epi(n, 3 * hid, 0, 0);
            push(&mut ops, &k.epilogue, p, &qkv, &dummy_rw, &b.qkv_b, &dummy_r, g);
            let rp = VisP { rows: n as u32, heads: c.n_heads as u32, grid_w: w as u32, ..Default::default() };
            push(&mut ops, &k.rope, rp, &qkv, &dummy_rw, &dummy_r, &dummy_r, split((n * c.n_heads * 36).div_ceil(256) as u32));
            push(&mut ops, &k.attend, rp, &qkv, &att, &dummy_r, &dummy_r, (n.div_ceil(64) as u32, c.n_heads as u32, 1));
            mat(&mut ops, &b.proj, &step_n, n, &att, &tmp, hid);
            let (p, g) = epi(n, hid, 1, 0);
            push(&mut ops, &k.epilogue, p, &tmp, &x, &b.proj_b, &dummy_r, g);

            let (p, g) = norm(n, hid);
            push(&mut ops, &k.layer_norm, p, &x, &xn, &b.n2, &dummy_r, g);
            mat(&mut ops, &b.fc1, &step_n, n, &xn, &mlp, c.intermediate_padded);
            let (p, g) = epi(n, c.intermediate_padded, 0, ACT_GELU_TANH);
            push(&mut ops, &k.epilogue, p, &mlp, &dummy_rw, &b.fc1_b, &dummy_r, g);
            mat(&mut ops, &b.fc2, &step_n, n, &mlp, &tmp, hid);
            let (p, g) = epi(n, hid, 1, 0);
            push(&mut ops, &k.epilogue, p, &tmp, &x, &b.fc2_b, &dummy_r, g);
            layer_ends.push(ops.len());
        }

        // Merger: per-patch LayerNorm, then each 2x2 block (4 consecutive rows) as one row.
        let (p, g) = norm(n, hid);
        push(&mut ops, &k.layer_norm, p, &x, &xn, &self.merge_norm, &dummy_r, g);
        mat(&mut ops, &self.merge_fc1, &step_q, nq, &xn, &mlp, c.merged_dim());
        let (p, g) = epi(nq, c.merged_dim(), 0, ACT_GELU_ERF);
        push(&mut ops, &k.epilogue, p, &mlp, &dummy_rw, &self.merge_fc1_b, &dummy_r, g);
        mat(&mut ops, &self.merge_fc2, &step_q, nq, &mlp, &qkv, c.out_hidden);
        let (p, g) = epi(nq, c.out_hidden, 0, 0);
        push(&mut ops, &k.epilogue, p, &qkv, &dummy_rw, &self.merge_fc2_b, &dummy_r, g);
        layer_ends.push(ops.len());

        if std::env::var_os("SENSENOVA_PROFILE").is_some() {
            // Time each dispatch on its own (wait after each) and report totals per kernel.
            gpu.queue.write_buffer(&pin, 0, bytemuck::cast_slice(&patches[..n * pd]));
            let mut totals: std::collections::BTreeMap<&str, f64> = Default::default();
            for d in &ops {
                let t = std::time::Instant::now();
                let mut enc = gpu.device.create_command_encoder(&Default::default());
                {
                    let mut pass = enc.begin_compute_pass(&Default::default());
                    pass.set_pipeline(&d.kernel.pipeline);
                    pass.set_bind_group(0, &d.bg, &[]);
                    pass.dispatch_workgroups(d.groups.0, d.groups.1, d.groups.2);
                }
                gpu.queue.submit([enc.finish()]);
                let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
                *totals.entry(d.name).or_default() += t.elapsed().as_secs_f64();
            }
            for (k, v) in totals {
                eprintln!("vision {k:<12} {:>8.1} ms", v * 1e3);
            }
        }
        let mut out = Vec::with_capacity(frames * nq * c.out_hidden);
        for fr in 0..frames {
            gpu.queue.write_buffer(&pin, 0, bytemuck::cast_slice(&patches[fr * n * pd..(fr + 1) * n * pd]));
            // One submission per block (or per op on slow GPUs) keeps each under the OS watchdog.
            let mut start = 0;
            for &end in &layer_ends {
                let groups: Vec<std::ops::Range<usize>> =
                    if slow { (start..end).map(|i| i..i + 1).collect() } else { vec![start..end] };
                for range in groups {
                    let mut enc = gpu.device.create_command_encoder(&Default::default());
                    {
                        let mut pass = enc.begin_compute_pass(&Default::default());
                        for d in &ops[range] {
                            pass.set_pipeline(&d.kernel.pipeline);
                            pass.set_bind_group(0, &d.bg, &[]);
                            pass.dispatch_workgroups(d.groups.0, d.groups.1, d.groups.2);
                        }
                    }
                    gpu.queue.submit([enc.finish()]);
                }
                start = end;
            }
            let mut enc = gpu.device.create_command_encoder(&Default::default());
            enc.copy_buffer_to_buffer(&qkv, 0, &staging, 0, staging.size());
            gpu.queue.submit([enc.finish()]);
            out.extend(gpu.map_read(&staging)?);
        }
        Ok(out)
    }
}

#[allow(clippy::too_many_arguments)]
fn mat_ops<'a>(
    gpu: &Gpu,
    mk: &'a Kernels,
    ops: &mut Vec<Dispatch<'a>>,
    wt: &QWeight,
    step: &wgpu::Buffer,
    rows: usize,
    xb: &wgpu::Buffer,
    yb: &wgpu::Buffer,
    y_stride: usize,
) {
    let kind = MatKind::for_rows(rows);
    let kernel = mk.matmul(wt.dtype, kind);
    for (qb, sb, wrows, r0) in &wt.chunks {
        let p = gpu.uniform("v_mat", &MatP {
            n: *wrows as u32,
            k: wt.k as u32,
            x_off: 0,
            x_stride: wt.k as u32,
            y_off: 0,
            y_stride: y_stride as u32,
            n_off: *r0 as u32,
            m_one: 0,
        });
        let bg = bind_group(gpu, kernel, "v_mat", &[
            Res::Whole(step), Res::Whole(&p), Res::Whole(xb), Res::Whole(qb), Res::Whole(sb), Res::Whole(yb),
        ]);
        let groups = match kind {
            MatKind::Tiled => (wrows.div_ceil(128) as u32, rows.div_ceil(128) as u32, 1),
            _ => split(wrows.div_ceil(kind.rows_per_group()) as u32),
        };
        ops.push(Dispatch { name: "matmul", kernel, bg, groups });
    }
}

fn vis_op<'a>(gpu: &Gpu, name: &'static str, kernel: &'a Kernel, p: VisP, bufs: [&wgpu::Buffer; 4], groups: (u32, u32, u32)) -> Dispatch<'a> {
    let pb = gpu.uniform("v_p", &p);
    let [a, b, w, e] = bufs;
    let bg = bind_group(gpu, kernel, "v", &[Res::Whole(&pb), Res::Whole(a), Res::Whole(b), Res::Whole(w), Res::Whole(e)]);
    Dispatch { name, kernel, bg, groups }
}
