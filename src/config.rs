use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Text-model hyperparameters, extracted from the Hugging Face `config.json`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelConfig {
    pub hidden: usize,
    pub intermediate: usize,
    pub vocab: usize,
    pub n_layers: usize,
    /// true for full-attention layers, false for linear (Gated DeltaNet) layers.
    pub full_attention: Vec<bool>,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub rot_dim: usize,
    pub rope_theta: f32,
    /// Interleaved M-RoPE frequency counts for (temporal, height, width).
    #[serde(default = "default_mrope")]
    pub mrope_section: [usize; 3],
    pub lin_k_heads: usize,
    pub lin_v_heads: usize,
    pub lin_k_dim: usize,
    pub lin_v_dim: usize,
    pub conv_kernel: usize,
    pub eps: f32,
    pub loop_start: usize,
    pub loop_end: usize,
    pub loop_count: usize,
    pub mhc: Option<MhcConfig>,
    pub eos: Vec<u32>,
    pub max_position: usize,
    /// Vision tower, absent in weight files converted before vision support.
    #[serde(default)]
    pub vision: Option<VisionConfig>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VisionConfig {
    pub hidden: usize,
    /// MLP width, and the same padded to a multiple of the quantization block.
    pub intermediate: usize,
    pub intermediate_padded: usize,
    pub depth: usize,
    pub n_heads: usize,
    pub patch: usize,
    pub temporal_patch: usize,
    pub merge: usize,
    pub out_hidden: usize,
    /// The learned position table is a square grid of this many patches per side.
    pub pos_side: usize,
    pub image_token: u32,
    pub video_token: u32,
    pub vision_start: u32,
    pub vision_end: u32,
}

impl VisionConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden / self.n_heads
    }
    /// Values per flattened patch: channels x frames x pixels.
    pub fn patch_dim(&self) -> usize {
        3 * self.temporal_patch * self.patch * self.patch
    }
    pub fn merged_dim(&self) -> usize {
        self.hidden * self.merge * self.merge
    }
}

fn default_mrope() -> [usize; 3] {
    [11, 11, 10]
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MhcConfig {
    pub start: usize,
    pub end: usize,
    pub streams: usize,
    pub iterations: usize,
}

impl ModelConfig {
    pub fn from_hf(config: &Value, generation: Option<&Value>) -> Result<Self> {
        let t = config.get("text_config").unwrap_or(config);
        let u = |v: &Value, k: &str| -> Result<usize> {
            v.get(k)
                .and_then(Value::as_u64)
                .map(|x| x as usize)
                .with_context(|| format!("config.json: missing integer `{k}`"))
        };
        let layer_types: Vec<bool> = t["layer_types"]
            .as_array()
            .context("config.json: missing layer_types")?
            .iter()
            .map(|v| v.as_str() == Some("full_attention"))
            .collect();
        let rope = t.get("rope_parameters").unwrap_or(t);
        let head_dim = u(t, "head_dim")?;
        let partial = rope
            .get("partial_rotary_factor")
            .and_then(Value::as_f64)
            .unwrap_or(1.0);
        if rope.get("mrope_section").is_some() && rope.get("mrope_interleaved").and_then(Value::as_bool) == Some(false) {
            bail!("non-interleaved M-RoPE is not supported");
        }
        let lp = config.get("loop_settings").context("config.json: missing loop_settings")?;
        let mhc = match config.get("mhc") {
            Some(m) if !m.is_null() => {
                if m.get("fixed_identity").and_then(Value::as_bool) == Some(true) {
                    bail!("mHC with fixed_identity is not supported");
                }
                if m.get("pre_scale").and_then(Value::as_f64).unwrap_or(1.0) != 1.0 {
                    bail!("mHC pre_scale != 1 is not supported");
                }
                Some(MhcConfig {
                    start: u(m, "start")?,
                    end: u(m, "end")?,
                    streams: u(m, "streams")?,
                    iterations: u(m, "iterations")?,
                })
            }
            _ => None,
        };
        let mut eos: Vec<u32> = Vec::new();
        let mut push_eos = |v: Option<&Value>| match v {
            Some(Value::Number(n)) => eos.extend(n.as_u64().map(|x| x as u32)),
            Some(Value::Array(a)) => eos.extend(a.iter().filter_map(Value::as_u64).map(|x| x as u32)),
            _ => {}
        };
        push_eos(config.get("eos_token_id"));
        push_eos(t.get("eos_token_id"));
        push_eos(generation.and_then(|g| g.get("eos_token_id")));
        eos.sort_unstable();
        eos.dedup();

        let cfg = ModelConfig {
            hidden: u(t, "hidden_size")?,
            intermediate: u(t, "intermediate_size")?,
            vocab: u(t, "vocab_size")?,
            n_layers: u(t, "num_hidden_layers")?,
            full_attention: layer_types,
            n_heads: u(t, "num_attention_heads")?,
            n_kv_heads: u(t, "num_key_value_heads")?,
            head_dim,
            rot_dim: (head_dim as f64 * partial) as usize,
            rope_theta: rope.get("rope_theta").and_then(Value::as_f64).unwrap_or(1e7) as f32,
            mrope_section: match rope.get("mrope_section").and_then(Value::as_array) {
                Some(a) if a.len() == 3 => [0, 1, 2].map(|i| a[i].as_u64().unwrap_or(0) as usize),
                _ => default_mrope(),
            },
            lin_k_heads: u(t, "linear_num_key_heads")?,
            lin_v_heads: u(t, "linear_num_value_heads")?,
            lin_k_dim: u(t, "linear_key_head_dim")?,
            lin_v_dim: u(t, "linear_value_head_dim")?,
            conv_kernel: u(t, "linear_conv_kernel_dim")?,
            eps: t.get("rms_norm_eps").and_then(Value::as_f64).unwrap_or(1e-6) as f32,
            loop_start: u(lp, "start")?,
            loop_end: u(lp, "end")?,
            loop_count: u(lp, "eval_count")?,
            mhc,
            eos,
            max_position: u(t, "max_position_embeddings").unwrap_or(262_144),
            vision: match config.get("vision_config") {
                Some(v) if !v.is_null() => {
                    let tok = |k: &str| u(config, k).map(|x| x as u32);
                    let inter = u(v, "intermediate_size")?;
                    let pos = u(v, "num_position_embeddings")?;
                    Some(VisionConfig {
                        hidden: u(v, "hidden_size")?,
                        intermediate: inter,
                        intermediate_padded: inter.div_ceil(32) * 32,
                        depth: u(v, "depth")?,
                        n_heads: u(v, "num_heads")?,
                        patch: u(v, "patch_size")?,
                        temporal_patch: u(v, "temporal_patch_size")?,
                        merge: u(v, "spatial_merge_size")?,
                        out_hidden: u(v, "out_hidden_size")?,
                        pos_side: (pos as f64).sqrt() as usize,
                        image_token: tok("image_token_id")?,
                        video_token: tok("video_token_id")?,
                        vision_start: tok("vision_start_token_id")?,
                        vision_end: tok("vision_end_token_id")?,
                    })
                }
                _ => None,
            },
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// The GPU kernels are specialized for this model family; reject anything else early.
    pub fn validate(&self) -> Result<()> {
        let checks = [
            (self.head_dim == 256, "head_dim must be 256"),
            (self.lin_k_dim == 128 && self.lin_v_dim == 128, "linear head dims must be 128"),
            (self.conv_kernel == 4, "linear conv kernel must be 4"),
            (self.hidden % 256 == 0, "hidden size must be a multiple of 256"),
            (self.full_attention.len() == self.n_layers, "layer_types length mismatch"),
            (self.n_heads == 4 * self.n_kv_heads, "attention must have 4 query heads per KV head"),
            (self.lin_v_heads % self.lin_k_heads == 0, "linear value heads must be a multiple of key heads"),
            (self.rot_dim % 2 == 0 && self.rot_dim <= self.head_dim, "invalid rotary dim"),
            (self.mrope_section.iter().sum::<usize>() == self.rot_dim / 2, "M-RoPE sections must cover the rotary dims"),
            (
                self.loop_start < self.loop_end && self.loop_end <= self.n_layers && self.loop_count >= 1,
                "invalid loop range",
            ),
        ];
        for (ok, msg) in checks {
            if !ok {
                bail!("unsupported model configuration: {msg}");
            }
        }
        if let Some(v) = &self.vision {
            let checks = [
                (v.head_dim() == 72, "vision head dim must be 72"),
                (v.patch == 16 && v.temporal_patch == 2 && v.merge == 2, "vision patches must be 16x16x2 with 2x2 merge"),
                (v.hidden % 32 == 0 && v.hidden <= 4096, "unsupported vision hidden size"),
                (v.out_hidden == self.hidden, "vision output must match the text hidden size"),
                (v.pos_side * v.pos_side > 0, "invalid vision position table"),
            ];
            for (ok, msg) in checks {
                if !ok {
                    bail!("unsupported vision configuration: {msg}");
                }
            }
        }
        if let Some(m) = &self.mhc {
            if m.streams != 2 {
                bail!("unsupported model configuration: only 2 mHC streams are supported");
            }
            let inside = |a: usize, b: usize| a <= m.start && m.end <= b;
            if !(inside(0, self.loop_start) || inside(self.loop_start, self.loop_end) || inside(self.loop_end, self.n_layers)) {
                bail!("mHC range crosses a loop boundary");
            }
        }
        Ok(())
    }

    /// Physical layer indices in execution order: prefix, looped block `loop_count` times, suffix.
    pub fn execution_plan(&self) -> Vec<usize> {
        let mut plan: Vec<usize> = (0..self.loop_start).collect();
        for _ in 0..self.loop_count {
            plan.extend(self.loop_start..self.loop_end);
        }
        plan.extend(self.loop_end..self.n_layers);
        plan
    }

    pub fn is_mhc(&self, layer: usize) -> bool {
        self.mhc.as_ref().is_some_and(|m| m.start <= layer && layer < m.end)
    }

    pub fn q_dim(&self) -> usize {
        self.n_heads * self.head_dim
    }
    pub fn kv_dim(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }
    pub fn lin_key_dim(&self) -> usize {
        self.lin_k_heads * self.lin_k_dim
    }
    pub fn lin_value_dim(&self) -> usize {
        self.lin_v_heads * self.lin_v_dim
    }
    pub fn conv_dim(&self) -> usize {
        2 * self.lin_key_dim() + self.lin_value_dim()
    }
    /// Rows of the fused linear-attention input projection: qkv | z | b | a.
    pub fn lin_proj_rows(&self) -> usize {
        self.conv_dim() + self.lin_value_dim() + 2 * self.lin_v_heads
    }
    /// Rows of the fused full-attention projection: q+gate | k | v.
    pub fn attn_proj_rows(&self) -> usize {
        2 * self.q_dim() + 2 * self.kv_dim()
    }
}
