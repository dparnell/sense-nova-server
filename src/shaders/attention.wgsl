// Gated full attention (16 query heads, 4 KV heads, head_dim 256, partial RoPE).
//
// `proj` rows hold the fused q_proj|k_proj|v_proj output:
//   [h * 512 .. h * 512 + 256)       query of head h
//   [h * 512 + 256 .. h * 512 + 512) output gate of head h
//   [q_dim * 2 + kh * 256 ..)         key of KV head kh
//   [q_dim * 2 + kv_dim + kh * 256 ..) value of KV head kh
// The KV cache stores f16 pairs packed in u32: cache[(pos * kv_dim + kh * 256 + d) / 2].

struct Step {
    m: u32,
    pos0: u32,
    parts: u32,
    _pad: u32,
}

struct Params {
    n_heads: u32,
    n_kv: u32,
    rot_dim: u32,
    proj_stride: u32,
    rope_theta: f32,
    eps: f32,
    scale: f32,
    // Interleaved M-RoPE: frequency i uses the height position if i % 3 == 1 and
    // i < mrope >> 16, the width position if i % 3 == 2 and i < mrope & 0xffff,
    // and the temporal position otherwise.
    mrope: u32,
}

const HD: u32 = 256u;

@group(0) @binding(0) var<uniform> step: Step;
@group(0) @binding(1) var<uniform> p: Params;
@group(0) @binding(2) var<storage, read_write> proj: array<f32>;
@group(0) @binding(3) var<storage, read_write> qbuf: array<f32>;      // [M, n_heads * 256]
@group(0) @binding(4) var<storage, read_write> kcache: array<u32>;
@group(0) @binding(5) var<storage, read_write> vcache: array<u32>;
@group(0) @binding(6) var<storage, read> norms: array<f32>;          // q_norm[256] | k_norm[256]
@group(0) @binding(7) var<storage, read_write> out: array<f32>;      // [M, n_heads * 256]
@group(0) @binding(8) var<storage, read_write> partial: array<f32>;  // split-K scratch
@group(0) @binding(9) var<storage, read> rpos: array<vec4<u32>>;     // [M] rotary (t, h, w) positions

var<workgroup> sh: array<f32, 256>;
var<workgroup> red: array<f32, 256>;

fn reduce_sum(lid: u32, v: f32) -> f32 {
    red[lid] = v;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (lid < s) { red[lid] += red[lid + s]; }
        workgroupBarrier();
    }
    let r = red[0];
    workgroupBarrier();
    return r;
}

// Workgroup (head slot, token). Head slots: [0, n_heads) queries,
// [n_heads, n_heads + n_kv) keys, then values. One thread per head dim.
@compute @workgroup_size(256)
fn prep(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) d: u32) {
    let slot = wid.x;
    let m = wid.y;
    let pos = step.pos0 + m;
    let q_dim = p.n_heads * HD;
    let kv_dim = p.n_kv * HD;
    let row = m * p.proj_stride;

    if (slot >= p.n_heads + p.n_kv) {
        let kh = slot - p.n_heads - p.n_kv;
        sh[d] = proj[row + 2u * q_dim + kv_dim + kh * HD + d];
        workgroupBarrier();
        if (d % 2u == 0u) {
            vcache[(pos * kv_dim + kh * HD + d) / 2u] = pack2x16float(vec2<f32>(sh[d], sh[d + 1u]));
        }
        return;
    }

    var v: f32;
    var norm_off: u32;
    if (slot < p.n_heads) {
        v = proj[row + slot * 2u * HD + d];
        norm_off = 0u;
    } else {
        v = proj[row + 2u * q_dim + (slot - p.n_heads) * HD + d];
        norm_off = HD;
    }
    let total = reduce_sum(d, v * v);
    v = v * inverseSqrt(total / f32(HD) + p.eps) * (1.0 + norms[norm_off + d]);

    // Rotate-half RoPE over the first rot_dim dims.
    sh[d] = v;
    workgroupBarrier();
    let half_rot = p.rot_dim / 2u;
    if (d < p.rot_dim) {
        let fi = d % half_rot;
        let inv_freq = pow(p.rope_theta, -f32(2u * fi) / f32(p.rot_dim));
        let rp = rpos[m];
        var rot_pos = rp.x;
        if (fi % 3u == 1u && fi < (p.mrope >> 16u)) {
            rot_pos = rp.y;
        } else if (fi % 3u == 2u && fi < (p.mrope & 0xffffu)) {
            rot_pos = rp.z;
        }
        let ang = f32(rot_pos) * inv_freq;
        let cs = cos(ang);
        let sn = sin(ang);
        if (d < half_rot) {
            v = v * cs - sh[d + half_rot] * sn;
        } else {
            v = v * cs + sh[d - half_rot] * sn;
        }
    }
    workgroupBarrier();

    if (slot < p.n_heads) {
        qbuf[m * q_dim + slot * HD + d] = v;
    } else {
        sh[d] = v;
        workgroupBarrier();
        let kh = slot - p.n_heads;
        if (d % 2u == 0u) {
            kcache[(pos * kv_dim + kh * HD + d) / 2u] = pack2x16float(vec2<f32>(sh[d], sh[d + 1u]));
        }
    }
}

// Attention core. Workgroup (KV head, token, part); the GQA group of G = 4
// query heads sharing that KV head is processed together so K/V are read once.
// Keys are staged 16 at a time in shared memory (rows padded to 129 words to
// avoid bank conflicts) and folded into an online softmax per query head.
// With step.parts == 1 the gated output goes to `out`; otherwise
// (acc[256], max, sum) partials per query head go to `partial` for `combine`.
const G: u32 = 4u;
const KC: u32 = 16u;
const KROW: u32 = 129u;
var<workgroup> qs: array<f32, 1024>;    // [G][256]
var<workgroup> kt: array<u32, 2064>;    // [KC][KROW] packed f16 pairs
var<workgroup> partd: array<f32, 256>;  // partial dot products
var<workgroup> pr: array<f32, 64>;      // [G][KC] probabilities
var<workgroup> corr_sh: array<f32, 4>;
var<workgroup> m_sh: array<f32, 4>;
var<workgroup> l_sh: array<f32, 4>;

@compute @workgroup_size(256)
fn attend(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let kh = wid.x;
    let m = wid.y;
    let part = wid.z;
    let q_dim = p.n_heads * HD;
    let kv_dim = p.n_kv * HD;

    for (var g = 0u; g < G; g++) {
        qs[g * HD + lid] = qbuf[m * q_dim + (kh * G + g) * HD + lid];
    }
    if (lid < G) {
        m_sh[lid] = -1e30;
        l_sh[lid] = 0.0;
    }

    let total = step.pos0 + m + 1u;
    let chunk = (total + step.parts - 1u) / step.parts;
    let start = min(total, part * chunk);
    let end = min(total, start + chunk);

    var acc = vec4<f32>(0.0); // dim `lid` for each of the G heads
    // Score-phase mapping: dot = lid / 4 -> (head g, key j); seg = lid % 4 interleaves words.
    let dg = (lid / 4u) / KC;
    let dj = (lid / 4u) % KC;
    let seg = lid % 4u;

    var t0 = start;
    loop {
        if (t0 >= end) { break; }
        let n_keys = min(KC, end - t0);
        workgroupBarrier();
        // Load KC keys (128 words each), coalesced.
        for (var i = lid; i < KC * 128u; i += 256u) {
            let kj = i / 128u;
            let w = i % 128u;
            var v = 0u;
            if (kj < n_keys) { v = kcache[((t0 + kj) * kv_dim + kh * HD) / 2u + w]; }
            kt[kj * KROW + w] = v;
        }
        workgroupBarrier();

        var d = 0.0;
        for (var w = 0u; w < 32u; w++) {
            let word = w * 4u + seg;
            let kk = unpack2x16float(kt[dj * KROW + word]);
            d += qs[dg * HD + 2u * word] * kk.x + qs[dg * HD + 2u * word + 1u] * kk.y;
        }
        partd[lid] = d;
        workgroupBarrier();
        if (lid < G * KC) {
            let g = lid / KC;
            let j = lid % KC;
            let b = (g * KC + j) * 4u;
            var sc = -1e30;
            if (j < n_keys) { sc = (partd[b] + partd[b + 1u] + partd[b + 2u] + partd[b + 3u]) * p.scale; }
            pr[lid] = sc;
        }
        workgroupBarrier();
        if (lid < G) {
            var mx = m_sh[lid];
            for (var j = 0u; j < n_keys; j++) { mx = max(mx, pr[lid * KC + j]); }
            let c = exp(m_sh[lid] - mx);
            var sum = 0.0;
            for (var j = 0u; j < KC; j++) {
                var pe = 0.0;
                if (j < n_keys) { pe = exp(pr[lid * KC + j] - mx); }
                pr[lid * KC + j] = pe;
                sum += pe;
            }
            corr_sh[lid] = c;
            m_sh[lid] = mx;
            l_sh[lid] = l_sh[lid] * c + sum;
        }
        workgroupBarrier();
        acc *= vec4<f32>(corr_sh[0], corr_sh[1], corr_sh[2], corr_sh[3]);
        for (var j = 0u; j < n_keys; j++) {
            let word = vcache[((t0 + j) * kv_dim + kh * HD + lid) / 2u];
            let pair = unpack2x16float(word);
            let v = select(pair.y, pair.x, lid % 2u == 0u);
            acc += v * vec4<f32>(pr[j], pr[KC + j], pr[2u * KC + j], pr[3u * KC + j]);
        }
        t0 += KC;
    }
    workgroupBarrier();

    for (var g = 0u; g < G; g++) {
        let h = kh * G + g;
        if (step.parts == 1u) {
            let gate = proj[m * p.proj_stride + h * 2u * HD + HD + lid];
            out[m * q_dim + h * HD + lid] = acc[g] / l_sh[g] / (1.0 + exp(-gate));
        } else {
            let pi = (m * p.n_heads + h) * step.parts + part;
            partial[pi * (HD + 2u) + lid] = acc[g];
            if (lid == 0u) {
                partial[pi * (HD + 2u) + HD] = m_sh[g];
                partial[pi * (HD + 2u) + HD + 1u] = l_sh[g];
            }
        }
    }
}

// Merge split-K partials. Workgroup (head, token), thread per dim.
@compute @workgroup_size(256)
fn combine(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let h = wid.x;
    let m = wid.y;
    let q_dim = p.n_heads * HD;
    let base = (m * p.n_heads + h) * step.parts;
    var mx = -1e30;
    for (var i = 0u; i < step.parts; i++) {
        mx = max(mx, partial[(base + i) * (HD + 2u) + HD]);
    }
    var l = 0.0;
    var acc = 0.0;
    for (var i = 0u; i < step.parts; i++) {
        let o = (base + i) * (HD + 2u);
        let f = exp(partial[o + HD] - mx);
        l += partial[o + HD + 1u] * f;
        acc += partial[o + lid] * f;
    }
    let gate = proj[m * p.proj_stride + h * 2u * HD + HD + lid];
    out[m * q_dim + h * HD + lid] = acc / l / (1.0 + exp(-gate));
}
