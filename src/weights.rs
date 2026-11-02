//! The `.snl` weight file: a JSON header followed by 256-byte aligned tensors.
//!
//! Quantized tensors are stored as two regions, packed quants then one f32
//! scale per 32-element block, which is exactly the layout the GPU kernels read.

use std::collections::BTreeMap;
use std::fs::File;
use std::path::Path;

use anyhow::{Context, Result, bail};
use memmap2::Mmap;
use serde::{Deserialize, Serialize};

use crate::config::ModelConfig;

pub const MAGIC: &[u8; 4] = b"SNLQ";
pub const VERSION: u32 = 1;
pub const ALIGN: u64 = 256;
pub const QBLOCK: usize = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DType {
    F32,
    Q8_0,
    Q4_0,
}

impl DType {
    pub fn parse(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "q8_0" | "q8" => Ok(DType::Q8_0),
            "q4_0" | "q4" => Ok(DType::Q4_0),
            "f32" => Ok(DType::F32),
            _ => bail!("unknown quantization `{s}` (expected q8_0 or q4_0)"),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::Q8_0 => "q8_0",
            DType::Q4_0 => "q4_0",
        }
    }

    /// (quant bytes, scale bytes) for a [rows, cols] tensor.
    pub fn sizes(self, rows: usize, cols: usize) -> (u64, u64) {
        let n = (rows * cols) as u64;
        match self {
            DType::F32 => (n * 4, 0),
            DType::Q8_0 => (n, n / QBLOCK as u64 * 4),
            DType::Q4_0 => (n / 2, n / QBLOCK as u64 * 4),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TensorInfo {
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub offset: u64,
    pub scales_offset: u64,
}

impl TensorInfo {
    pub fn rows(&self) -> usize {
        if self.shape.len() == 1 { 1 } else { self.shape[0] }
    }
    pub fn cols(&self) -> usize {
        *self.shape.last().unwrap()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Header {
    pub model_name: String,
    pub quant: String,
    pub config: ModelConfig,
    pub tensors: BTreeMap<String, TensorInfo>,
}

pub struct WeightFile {
    pub header: Header,
    mmap: Mmap,
}

impl WeightFile {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        // SAFETY: the weight file is treated as read-only for the lifetime of the map.
        let mmap = unsafe { Mmap::map(&file)? };
        if mmap.len() < 16 || &mmap[..4] != MAGIC {
            bail!("{} is not a SenseNova Loop weight file", path.display());
        }
        let version = u32::from_le_bytes(mmap[4..8].try_into().unwrap());
        if version != VERSION {
            bail!("unsupported weight file version {version}; re-run `pull`");
        }
        let len = u64::from_le_bytes(mmap[8..16].try_into().unwrap()) as usize;
        let header: Header = serde_json::from_slice(&mmap[16..16 + len]).context("parsing weight header")?;
        header.config.validate()?;
        Ok(WeightFile { header, mmap })
    }

    pub fn info(&self, name: &str) -> Result<&TensorInfo> {
        self.header
            .tensors
            .get(name)
            .with_context(|| format!("weight file is missing tensor `{name}`"))
    }

    /// Raw (quants, scales) byte regions of a tensor.
    pub fn bytes(&self, name: &str) -> Result<(&[u8], &[u8])> {
        let t = self.info(name)?;
        let (qb, sb) = t.dtype.sizes(t.rows(), t.cols());
        let q = &self.mmap[t.offset as usize..(t.offset + qb) as usize];
        let s = &self.mmap[t.scales_offset as usize..(t.scales_offset + sb) as usize];
        Ok((q, s))
    }
}

pub fn quantize_q8(src: &[f32], q: &mut [u8], scales: &mut [f32]) {
    for ((block, qb), s) in src.chunks_exact(QBLOCK).zip(q.chunks_exact_mut(QBLOCK)).zip(scales.iter_mut()) {
        let amax = block.iter().fold(0f32, |a, &v| a.max(v.abs()));
        let d = amax / 127.0;
        let id = if d > 0.0 { 1.0 / d } else { 0.0 };
        for (o, &v) in qb.iter_mut().zip(block) {
            *o = ((v * id).round().clamp(-127.0, 127.0) as i8) as u8;
        }
        *s = d;
    }
}

/// Q4_0 with nibble i of 32-bit word w holding element w * 8 + i.
pub fn quantize_q4(src: &[f32], q: &mut [u8], scales: &mut [f32]) {
    for ((block, qb), s) in src.chunks_exact(QBLOCK).zip(q.chunks_exact_mut(QBLOCK / 2)).zip(scales.iter_mut()) {
        // Same convention as ggml Q4_0: the largest-magnitude value maps to -8.
        let max = block.iter().fold(0f32, |a, &v| if v.abs() > a.abs() { v } else { a });
        let d = max / -8.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        for w in 0..4 {
            let mut word = 0u32;
            for i in 0..8 {
                let v = block[w * 8 + i];
                let n = ((v * id + 8.5) as i32).clamp(0, 15) as u32;
                word |= n << (4 * i);
            }
            qb[w * 4..w * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }
        *s = d;
    }
}

/// Dequantize one Q8_0 row (used for the CPU-side embedding lookup).
pub fn dequant_q8_row(q: &[u8], scales: &[f32], out: &mut [f32]) {
    for ((o, qb), &d) in out.chunks_exact_mut(QBLOCK).zip(q.chunks_exact(QBLOCK)).zip(scales) {
        for (v, &b) in o.iter_mut().zip(qb) {
            *v = (b as i8) as f32 * d;
        }
    }
}
