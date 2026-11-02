//! Convert the official Hugging Face safetensors checkpoint into a quantized `.snl` file.

use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use half::{bf16, f16};
use indicatif::{ProgressBar, ProgressStyle};
use memmap2::Mmap;
use rayon::prelude::*;
use serde_json::Value;

use crate::config::ModelConfig;
use crate::weights::{ALIGN, DType, Header, MAGIC, QBLOCK, TensorInfo, VERSION, quantize_q4, quantize_q8};

struct SourceTensor {
    file: usize,
    dtype: String,
    shape: Vec<usize>,
    start: usize,
    end: usize,
}

/// Read-only view over a (possibly sharded) safetensors checkpoint.
struct Safetensors {
    maps: Vec<Mmap>,
    tensors: HashMap<String, SourceTensor>,
}

impl Safetensors {
    fn open(dir: &Path) -> Result<Self> {
        let mut files: Vec<PathBuf> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
            .filter(|p| p.file_name().is_some_and(|n| n.to_string_lossy().starts_with("model")))
            .collect();
        files.sort();
        if files.is_empty() {
            bail!("no model*.safetensors files in {}", dir.display());
        }
        let mut maps = Vec::new();
        let mut tensors = HashMap::new();
        for (fi, path) in files.iter().enumerate() {
            let f = File::open(path).with_context(|| format!("opening {}", path.display()))?;
            // SAFETY: source checkpoints are not modified while converting.
            let map = unsafe { Mmap::map(&f)? };
            let n = u64::from_le_bytes(map[..8].try_into().unwrap()) as usize;
            let header: BTreeMap<String, Value> = serde_json::from_slice(&map[8..8 + n])
                .with_context(|| format!("parsing header of {}", path.display()))?;
            for (name, v) in header {
                if name == "__metadata__" {
                    continue;
                }
                let offs = v["data_offsets"].as_array().context("bad data_offsets")?;
                tensors.insert(
                    name,
                    SourceTensor {
                        file: fi,
                        dtype: v["dtype"].as_str().unwrap_or("").to_string(),
                        shape: v["shape"]
                            .as_array()
                            .context("bad shape")?
                            .iter()
                            .map(|x| x.as_u64().unwrap_or(0) as usize)
                            .collect(),
                        start: 8 + n + offs[0].as_u64().unwrap() as usize,
                        end: 8 + n + offs[1].as_u64().unwrap() as usize,
                    },
                );
            }
            maps.push(map);
        }
        Ok(Safetensors { maps, tensors })
    }

    fn shape(&self, name: &str) -> Result<&[usize]> {
        Ok(&self.get(name)?.shape)
    }

    fn get(&self, name: &str) -> Result<&SourceTensor> {
        self.tensors.get(name).with_context(|| format!("checkpoint is missing `{name}`"))
    }

    fn read(&self, name: &str) -> Result<Vec<f32>> {
        let t = self.get(name)?;
        let bytes = &self.maps[t.file][t.start..t.end];
        Ok(match t.dtype.as_str() {
            "F32" => bytes.par_chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
            "BF16" => bytes
                .par_chunks_exact(2)
                .map(|c| bf16::from_le_bytes(c.try_into().unwrap()).to_f32())
                .collect(),
            "F16" => bytes
                .par_chunks_exact(2)
                .map(|c| f16::from_le_bytes(c.try_into().unwrap()).to_f32())
                .collect(),
            other => bail!("tensor `{name}` has unsupported dtype {other}"),
        })
    }

    fn concat(&self, names: &[String]) -> Result<Vec<f32>> {
        let mut out = Vec::new();
        for n in names {
            out.extend(self.read(n)?);
        }
        Ok(out)
    }
}

enum Source {
    /// Row-wise concatenation of these checkpoint tensors.
    Concat(Vec<String>),
    /// mHC router: weight [8, 2H] followed by bias [8].
    WeightBias(String, String),
}

struct Planned {
    name: String,
    dtype: DType,
    shape: Vec<usize>,
    source: Source,
}

fn matrix(out: &mut Vec<Planned>, st: &Safetensors, name: String, dtype: DType, parts: Vec<String>) -> Result<()> {
    let mut rows = 0;
    let mut cols = 0;
    for p in &parts {
        let s = st.shape(p)?;
        rows += s[0];
        cols = s[1];
    }
    out.push(Planned { name, dtype, shape: vec![rows, cols], source: Source::Concat(parts) });
    Ok(())
}

fn vector(out: &mut Vec<Planned>, st: &Safetensors, name: String, parts: Vec<String>) -> Result<()> {
    let mut len = 0;
    for p in &parts {
        len += st.shape(p)?.iter().product::<usize>();
    }
    out.push(Planned { name, dtype: DType::F32, shape: vec![len], source: Source::Concat(parts) });
    Ok(())
}

fn plan(cfg: &ModelConfig, st: &Safetensors, quant: DType) -> Result<Vec<Planned>> {
    let lm = "model.language_model";
    let mut out = Vec::new();
    let o = &mut out;
    // Embedding and output head stay at 8 bits regardless of the requested quantization.
    matrix(o, st, "embed".into(), DType::Q8_0, vec![format!("{lm}.embed_tokens.weight")])?;
    matrix(o, st, "lm_head".into(), DType::Q8_0, vec!["lm_head.weight".into()])?;
    vector(o, st, "norm".into(), vec![format!("{lm}.norm.weight")])?;
    for i in 0..cfg.n_layers {
        let p = format!("{lm}.layers.{i}");
        vector(o, st, 
format!("l{i}.in_ln"), vec![format!("{p}.input_layernorm.weight")])?;
        vector(o, st, 
format!("l{i}.post_ln"), vec![format!("{p}.post_attention_layernorm.weight")])?;
        if cfg.full_attention[i] {
            vector(o, st, 
                format!("l{i}.qk_norm"),
                vec![format!("{p}.self_attn.q_norm.weight"), format!("{p}.self_attn.k_norm.weight")],
            )?;
        } else {
            vector(o, st, 
                format!("l{i}.lin"),
                vec![
                    format!("{p}.linear_attn.conv1d.weight"),
                    format!("{p}.linear_attn.A_log"),
                    format!("{p}.linear_attn.dt_bias"),
                    format!("{p}.linear_attn.norm.weight"),
                ],
            )?;
        }
        if cfg.is_mhc(i) {
            for r in ["mhc_attn", "mhc_ffn"] {
                let w = format!("{p}.{r}.weight");
                let b = format!("{p}.{r}.bias");
                let len = st.shape(&w)?.iter().product::<usize>() + st.shape(&b)?.iter().product::<usize>();
                o.push(Planned {
                    name: format!("l{i}.{r}"),
                    dtype: DType::F32,
                    shape: vec![len],
                    source: Source::WeightBias(w, b),
                });
            }
        }
    }
    for i in 0..cfg.n_layers {
        let p = format!("{lm}.layers.{i}");
        if cfg.full_attention[i] {
            let a = format!("{p}.self_attn");
            matrix(
                o,
                st,
                format!("l{i}.qkv"),
                quant,
                vec![format!("{a}.q_proj.weight"), format!("{a}.k_proj.weight"), format!("{a}.v_proj.weight")],
            )?;
            matrix(o, st, format!("l{i}.o_proj"), quant, vec![format!("{a}.o_proj.weight")])?;
        } else {
            let a = format!("{p}.linear_attn");
            // The beta/decay rows are tiny and sensitive, so this projection keeps 8 bits.
            matrix(
                o,
                st,
                format!("l{i}.in_proj"),
                DType::Q8_0,
                vec![
                    format!("{a}.in_proj_qkv.weight"),
                    format!("{a}.in_proj_z.weight"),
                    format!("{a}.in_proj_b.weight"),
                    format!("{a}.in_proj_a.weight"),
                ],
            )?;
            matrix(o, st, format!("l{i}.out_proj"), quant, vec![format!("{a}.out_proj.weight")])?;
        }
        matrix(
            o,
            st,
            format!("l{i}.gate_up"),
            quant,
            vec![format!("{p}.mlp.gate_proj.weight"), format!("{p}.mlp.up_proj.weight")],
        )?;
        matrix(o, st, format!("l{i}.down"), quant, vec![format!("{p}.mlp.down_proj.weight")])?;
    }
    Ok(out)
}

fn align(x: u64) -> u64 {
    x.div_ceil(ALIGN) * ALIGN
}

/// Convert the checkpoint in `src` (HF layout) into `dst`.
pub fn convert(src: &Path, dst: &Path, quant: DType, model_name: &str) -> Result<()> {
    let config: Value = serde_json::from_slice(&std::fs::read(src.join("config.json"))?)?;
    let generation: Option<Value> = std::fs::read(src.join("generation_config.json"))
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok());
    let cfg = ModelConfig::from_hf(&config, generation.as_ref())?;
    let st = Safetensors::open(src)?;
    let planned = plan(&cfg, &st, quant)?;

    // Lay out every tensor before writing so the header can be written first.
    let mut tensors = BTreeMap::new();
    let mut cursor = 0u64;
    for t in &planned {
        let (rows, cols) = if t.shape.len() == 1 { (1, t.shape[0]) } else { (t.shape[0], t.shape[1]) };
        if t.dtype != DType::F32 && cols % QBLOCK != 0 {
            bail!("{}: columns not divisible by {QBLOCK}", t.name);
        }
        let (qb, sb) = t.dtype.sizes(rows, cols);
        let offset = cursor;
        cursor = align(cursor + qb);
        let scales_offset = cursor;
        cursor = align(cursor + sb);
        tensors.insert(t.name.clone(), TensorInfo { dtype: t.dtype, shape: t.shape.clone(), offset, scales_offset });
    }
    let mut header = Header { model_name: model_name.to_string(), quant: quant.name().into(), config: cfg, tensors };
    let probe = serde_json::to_vec(&header)?;
    let data_start = align(16 + probe.len() as u64 + 4096);
    for t in header.tensors.values_mut() {
        t.offset += data_start;
        t.scales_offset += data_start;
    }
    let header_bytes = serde_json::to_vec(&header)?;
    if 16 + header_bytes.len() as u64 > data_start {
        bail!("internal error: header larger than reserved space");
    }

    let tmp = dst.with_extension("snl.partial");
    let mut w = BufWriter::with_capacity(16 << 20, File::create(&tmp)?);
    w.write_all(MAGIC)?;
    w.write_all(&VERSION.to_le_bytes())?;
    w.write_all(&(header_bytes.len() as u64).to_le_bytes())?;
    w.write_all(&header_bytes)?;

    let bar = ProgressBar::new(planned.len() as u64);
    bar.set_style(ProgressStyle::with_template("quantizing {bar:40} {pos}/{len} {msg}").unwrap());
    for t in &planned {
        bar.set_message(t.name.clone());
        let info = &header.tensors[&t.name];
        let data = match &t.source {
            Source::Concat(parts) => st.concat(parts)?,
            Source::WeightBias(wn, bn) => {
                let mut v = st.read(wn)?;
                v.extend(st.read(bn)?);
                v
            }
        };
        let (rows, cols) = (info.rows(), info.cols());
        if data.len() != rows * cols {
            bail!("{}: expected {} values, got {}", t.name, rows * cols, data.len());
        }
        let (q, s) = encode(&data, info.dtype, cols);
        w.seek(SeekFrom::Start(info.offset))?;
        w.write_all(&q)?;
        if !s.is_empty() {
            w.seek(SeekFrom::Start(info.scales_offset))?;
            w.write_all(&s)?;
        }
        bar.inc(1);
    }
    bar.finish_with_message("done");
    // Make sure the file covers the final alignment padding.
    let end = header.tensors.values().map(|t| {
        let (qb, sb) = t.dtype.sizes(t.rows(), t.cols());
        (t.offset + qb).max(t.scales_offset + sb)
    }).max().unwrap_or(0);
    let mut f = w.into_inner().map_err(|e| e.into_error())?;
    f.set_len(align(end))?;
    f.flush()?;
    drop(f);
    std::fs::rename(&tmp, dst)?;
    Ok(())
}

fn encode(data: &[f32], dtype: DType, cols: usize) -> (Vec<u8>, Vec<u8>) {
    match dtype {
        DType::F32 => (bytemuck::cast_slice(data).to_vec(), Vec::new()),
        DType::Q8_0 | DType::Q4_0 => {
            let (qb, sb) = dtype.sizes(data.len() / cols, cols);
            let mut q = vec![0u8; qb as usize];
            let mut s = vec![0f32; sb as usize / 4];
            let q_per_block = if dtype == DType::Q8_0 { QBLOCK } else { QBLOCK / 2 };
            // Parallelize over chunks of 1024 blocks.
            q.par_chunks_mut(q_per_block * 1024)
                .zip(s.par_chunks_mut(1024))
                .zip(data.par_chunks(QBLOCK * 1024))
                .for_each(|((qc, sc), dc)| {
                    if dtype == DType::Q8_0 {
                        quantize_q8(dc, qc, sc)
                    } else {
                        quantize_q4(dc, qc, sc)
                    }
                });
            (q, bytemuck::cast_slice(&s).to_vec())
        }
    }
}
