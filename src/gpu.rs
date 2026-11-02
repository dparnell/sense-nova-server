//! wgpu device selection and compute pipelines (Vulkan, DX12 or Metal).

use std::borrow::Cow;

use anyhow::{Context, Result, bail};
use wgpu::util::DeviceExt;

use crate::weights::DType;

pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub info: wgpu::AdapterInfo,
    pub limits: wgpu::Limits,
    /// Bytes written through the queue since the last flush.
    pending_upload: std::cell::Cell<u64>,
    /// Total bytes of storage buffers allocated.
    pub allocated: std::cell::Cell<u64>,
}

fn backends(name: Option<&str>) -> Result<wgpu::Backends> {
    Ok(match name.map(|s| s.to_ascii_lowercase()) {
        None => wgpu::Backends::PRIMARY,
        Some(s) => match s.as_str() {
            "vulkan" | "vk" => wgpu::Backends::VULKAN,
            "dx12" | "d3d12" => wgpu::Backends::DX12,
            "metal" | "mtl" => wgpu::Backends::METAL,
            "gl" | "opengl" | "gles" => wgpu::Backends::GL,
            "auto" | "primary" => wgpu::Backends::PRIMARY,
            other => bail!("unknown backend `{other}` (vulkan, dx12, metal, gl)"),
        },
    })
}

fn instance(backend: Option<&str>) -> Result<wgpu::Instance> {
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle_from_env();
    desc.backends = backends(backend)?;
    Ok(wgpu::Instance::new(desc))
}

fn rank(info: &wgpu::AdapterInfo) -> u32 {
    let kind = match info.device_type {
        wgpu::DeviceType::DiscreteGpu => 0,
        wgpu::DeviceType::IntegratedGpu => 1,
        wgpu::DeviceType::VirtualGpu => 2,
        wgpu::DeviceType::Other => 3,
        wgpu::DeviceType::Cpu => 4,
    };
    // Prefer the native API of each platform when one GPU shows up under several backends.
    let api = match info.backend {
        wgpu::Backend::Metal => 0,
        wgpu::Backend::Vulkan => 1,
        wgpu::Backend::Dx12 => 2,
        _ => 3,
    };
    kind * 10 + api
}

pub fn list_adapters(backend: Option<&str>) -> Result<Vec<wgpu::AdapterInfo>> {
    let inst = instance(backend)?;
    let mut adapters: Vec<_> = pollster::block_on(inst.enumerate_adapters(inst_backends(backend)?))
        .into_iter()
        .map(|a| a.get_info())
        .collect();
    adapters.sort_by_key(rank);
    Ok(adapters)
}

fn inst_backends(backend: Option<&str>) -> Result<wgpu::Backends> {
    backends(backend)
}

impl Gpu {
    /// Open the best GPU, or the `device`-th entry of `list_adapters`.
    pub fn new(backend: Option<&str>, device: Option<usize>) -> Result<Self> {
        let inst = instance(backend)?;
        let mut adapters = pollster::block_on(inst.enumerate_adapters(inst_backends(backend)?));
        adapters.sort_by_key(|a| rank(&a.get_info()));
        if adapters.is_empty() {
            bail!("no GPU adapters found (tried backends: {:?})", inst_backends(backend)?);
        }
        let adapter = match device {
            Some(i) => adapters.into_iter().nth(i).with_context(|| format!("no adapter #{i}; see `devices`"))?,
            None => adapters.into_iter().next().unwrap(),
        };
        let info = adapter.get_info();
        let limits = adapter.limits();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("sensenova-loop"),
            required_features: wgpu::Features::empty(),
            required_limits: limits.clone(),
            experimental_features: wgpu::ExperimentalFeatures::default(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))
        .context("creating GPU device")?;
        if limits.max_compute_workgroup_storage_size < 16384 || limits.max_compute_invocations_per_workgroup < 256 {
            bail!("GPU `{}` does not meet the minimum compute limits", info.name);
        }
        Ok(Gpu { device, queue, info, limits, pending_upload: Default::default(), allocated: Default::default() })
    }

    /// Largest single storage binding we will create.
    pub fn max_binding(&self) -> u64 {
        (self.limits.max_storage_buffer_binding_size as u64).min(self.limits.max_buffer_size).min(1 << 28)
    }

    pub fn storage(&self, label: &str, size: u64) -> wgpu::Buffer {
        self.allocated.set(self.allocated.get() + size);
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size.max(16).div_ceil(4) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Create a storage buffer from bytes. Uploads go through the queue and are
    /// flushed periodically so staging memory stays bounded for large models.
    pub fn storage_init(&self, label: &str, bytes: &[u8]) -> wgpu::Buffer {
        let buf = self.storage(label, bytes.len() as u64);
        let whole = bytes.len() / 4 * 4;
        if whole > 0 {
            self.queue.write_buffer(&buf, 0, &bytes[..whole]);
        }
        if whole < bytes.len() {
            let mut tail = bytes[whole..].to_vec();
            tail.resize(4, 0);
            self.queue.write_buffer(&buf, whole as u64, &tail);
        }
        let pending = self.pending_upload.get() + bytes.len() as u64;
        if pending > 256 << 20 {
            self.queue.submit([]);
            let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
            self.pending_upload.set(0);
        } else {
            self.pending_upload.set(pending);
        }
        buf
    }

    pub fn uniform<T: bytemuck::Pod>(&self, label: &str, value: &T) -> wgpu::Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::bytes_of(value),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        })
    }

    pub fn map_read(&self, staging: &wgpu::Buffer) -> Result<Vec<f32>> {
        let (tx, rx) = std::sync::mpsc::channel();
        staging.map_async(wgpu::MapMode::Read, .., move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| anyhow::anyhow!("GPU poll failed: {e}"))?;
        rx.recv()?.context("mapping GPU buffer")?;
        let out = {
            let view = staging.get_mapped_range(..).context("reading mapped buffer")?;
            bytemuck::cast_slice::<u8, f32>(&view).to_vec()
        };
        staging.unmap();
        Ok(out)
    }
}

#[derive(Clone, Copy)]
pub enum Bind {
    Uniform,
    Read,
    ReadWrite,
}

pub struct Kernel {
    pub pipeline: wgpu::ComputePipeline,
    pub layout: wgpu::BindGroupLayout,
}

/// Which matrix kernel to use for a given number of token rows.
#[derive(Clone, Copy)]
pub enum MatKind {
    /// One row (decode).
    Vec1,
    /// Up to 8 rows.
    Vec,
    /// Tiled matmul for prefill.
    Tiled,
}

impl MatKind {
    pub fn for_rows(m: usize) -> Self {
        match m {
            1 => MatKind::Vec1,
            2..=8 => MatKind::Vec,
            _ => MatKind::Tiled,
        }
    }
}

pub struct Kernels {
    /// [q8_0, q4_0] x [Vec1, Vec, Tiled]; all share one bind group layout.
    pub mat: [[Kernel; 3]; 2],
    pub branch_norm: Kernel,
    pub mhc_dots: Kernel,
    pub mhc_finish: Kernel,
    pub merge: Kernel,
    pub swiglu: Kernel,
    pub attn_prep: Kernel,
    pub attend: Kernel,
    pub combine: Kernel,
    pub conv: Kernel,
    pub delta: Kernel,
}

const Q8_FNS: &str = r#"
const BLOCK_VEC4S: u32 = 2u;
fn block_dot(wbase: u32, xbase: u32) -> f32 {
    let a = wq[wbase];
    let b = wq[wbase + 1u];
    return dot(i8x4(a.x), x[xbase]) + dot(i8x4(a.y), x[xbase + 1u])
         + dot(i8x4(a.z), x[xbase + 2u]) + dot(i8x4(a.w), x[xbase + 3u])
         + dot(i8x4(b.x), x[xbase + 4u]) + dot(i8x4(b.y), x[xbase + 5u])
         + dot(i8x4(b.z), x[xbase + 6u]) + dot(i8x4(b.w), x[xbase + 7u]);
}
fn dequant8(wbase: u32, part: u32, scale: f32) -> array<f32, 8> {
    let v = wq[wbase + part / 2u];
    var w0 = v.x;
    var w1 = v.y;
    if (part % 2u == 1u) { w0 = v.z; w1 = v.w; }
    let a = i8x4(w0) * scale;
    let b = i8x4(w1) * scale;
    return array<f32, 8>(a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w);
}
"#;

const Q4_FNS: &str = r#"
const BLOCK_VEC4S: u32 = 1u;
fn block_dot(wbase: u32, xbase: u32) -> f32 {
    let v = wq[wbase];
    return dot(u4x4(v.x, 0u), x[xbase]) + dot(u4x4(v.x, 16u), x[xbase + 1u])
         + dot(u4x4(v.y, 0u), x[xbase + 2u]) + dot(u4x4(v.y, 16u), x[xbase + 3u])
         + dot(u4x4(v.z, 0u), x[xbase + 4u]) + dot(u4x4(v.z, 16u), x[xbase + 5u])
         + dot(u4x4(v.w, 0u), x[xbase + 6u]) + dot(u4x4(v.w, 16u), x[xbase + 7u]);
}
fn dequant8(wbase: u32, part: u32, scale: f32) -> array<f32, 8> {
    let w = wq[wbase][part];
    let a = u4x4(w, 0u) * scale;
    let b = u4x4(w, 16u) * scale;
    return array<f32, 8>(a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w);
}
"#;

impl Kernels {
    pub fn new(gpu: &Gpu) -> Result<Self> {
        use Bind::*;
        let dev = &gpu.device;
        let module = |label: &str, src: String| {
            dev.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(Cow::Owned(src)),
            })
        };
        let layout = |label: &str, binds: &[Bind]| {
            let entries: Vec<_> = binds
                .iter()
                .enumerate()
                .map(|(i, b)| wgpu::BindGroupLayoutEntry {
                    binding: i as u32,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: match b {
                        Uniform => wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        Read | ReadWrite => wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: matches!(b, Read) },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                    },
                    count: None,
                })
                .collect();
            dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some(label), entries: &entries })
        };
        // Kernels of one shader module share a bind group layout, so bind groups are interchangeable.
        let kernel = |m: &wgpu::ShaderModule, entry: &str, bgl: &wgpu::BindGroupLayout| {
            let bgl = bgl.clone();
            let pl = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(entry),
                bind_group_layouts: &[Some(&bgl)],
                immediate_size: 0,
            });
            let pipeline = dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pl),
                module: m,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            });
            Kernel { pipeline, layout: bgl }
        };

        let mm_src = include_str!("shaders/matmul.wgsl");
        let q8 = module("matmul_q8", mm_src.replace("//QUANT_FNS//", Q8_FNS));
        let q4 = module("matmul_q4", mm_src.replace("//QUANT_FNS//", Q4_FNS));
        let mm = layout("matmul", &[Uniform, Uniform, Read, Read, Read, ReadWrite]);
        let layers = module("layers", include_str!("shaders/layers.wgsl").into());
        let ly = layout("layers", &[Uniform, Uniform, ReadWrite, ReadWrite, Read, ReadWrite]);
        let attn = module("attention", include_str!("shaders/attention.wgsl").into());
        let at = layout("attention", &[Uniform, Uniform, ReadWrite, ReadWrite, ReadWrite, ReadWrite, Read, ReadWrite, ReadWrite]);
        let lin = module("linear", include_str!("shaders/linear.wgsl").into());
        let li = layout("linear", &[Uniform, Uniform, ReadWrite, ReadWrite, ReadWrite, ReadWrite, Read, ReadWrite]);
        let mats = |m: &wgpu::ShaderModule| [kernel(m, "matvec1", &mm), kernel(m, "matvec", &mm), kernel(m, "matmul", &mm)];

        Ok(Kernels {
            mat: [mats(&q8), mats(&q4)],
            branch_norm: kernel(&layers, "branch_norm", &ly),
            mhc_dots: kernel(&layers, "mhc_dots", &ly),
            mhc_finish: kernel(&layers, "mhc_finish", &ly),
            merge: kernel(&layers, "merge", &ly),
            swiglu: kernel(&layers, "swiglu", &ly),
            attn_prep: kernel(&attn, "prep", &at),
            attend: kernel(&attn, "attend", &at),
            combine: kernel(&attn, "combine", &at),
            conv: kernel(&lin, "conv", &li),
            delta: kernel(&lin, "delta", &li),
        })
    }

    pub fn matmul(&self, dtype: DType, kind: MatKind) -> &Kernel {
        &self.mat[(dtype == DType::Q4_0) as usize][kind as usize]
    }
}
/// A buffer range to bind: whole buffer, or (buffer, offset, size).
pub enum Res<'a> {
    Whole(&'a wgpu::Buffer),
    Range(&'a wgpu::Buffer, u64, u64),
}

pub fn bind_group(gpu: &Gpu, kernel: &Kernel, label: &str, res: &[Res]) -> wgpu::BindGroup {
    let entries: Vec<_> = res
        .iter()
        .enumerate()
        .map(|(i, r)| wgpu::BindGroupEntry {
            binding: i as u32,
            resource: match r {
                Res::Whole(b) => b.as_entire_binding(),
                Res::Range(b, off, size) => wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: b,
                    offset: *off,
                    size: wgpu::BufferSize::new(*size),
                }),
            },
        })
        .collect();
    gpu.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some(label), layout: &kernel.layout, entries: &entries })
}
