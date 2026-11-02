// Element-wise and normalization kernels around the decoder layers:
// branch input norm, mHC routing and merging, SwiGLU.
//
// The residual buffer `x` always has room for two streams per token:
// x[m * 2H + s * H + i]. Layers outside the mHC range only use stream 0.

struct Step {
    m: u32,
    pos0: u32,
    parts: u32,
    _pad: u32,
}

struct Params {
    h: u32,          // hidden size
    mode: u32,       // kernel specific, see below
    last_row: u32,   // 1: read only the last token row of the step (final norm)
    eps: f32,
    weight_one: f32, // 1.0 for zero-centered (1 + w) norms, 0.0 for plain w
    iterations: u32, // sinkhorn iterations
    inter: u32,      // MLP intermediate size (swiglu)
    _pad: u32,
}

@group(0) @binding(0) var<uniform> step: Step;
@group(0) @binding(1) var<uniform> p: Params;
@group(0) @binding(2) var<storage, read_write> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> coef: array<f32>;   // [M, 8]
@group(0) @binding(4) var<storage, read> w: array<f32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;

var<workgroup> red: array<f32, 2048>;

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

// out[m, :] = rmsnorm(branch input) * (weight_one + w)
// mode 0: branch input is stream 0; mode 1: mHC weighted sum of both streams.
// One workgroup per token.
@compute @workgroup_size(256)
fn branch_norm(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let m = wid.x + select(0u, step.m - 1u, p.last_row == 1u);
    let base = m * 2u * p.h;
    var pre0 = 1.0;
    var pre1 = 0.0;
    if (p.mode == 1u) {
        pre0 = coef[m * 8u];
        pre1 = coef[m * 8u + 1u];
    }
    var ss = 0.0;
    for (var i = lid; i < p.h; i += 256u) {
        var v = pre0 * x[base + i];
        if (p.mode == 1u) { v += pre1 * x[base + p.h + i]; }
        ss += v * v;
    }
    let total = reduce_sum(lid, ss);
    let rinv = inverseSqrt(total / f32(p.h) + p.eps);
    let ob = wid.x * p.h;
    for (var i = lid; i < p.h; i += 256u) {
        var v = pre0 * x[base + i];
        if (p.mode == 1u) { v += pre1 * x[base + p.h + i]; }
        out[ob + i] = v * rinv * (p.weight_one + w[i]);
    }
}

// mHC router, step 1: workgroup (o, token). For o < 8 computes the dot of
// router row o (FP32, 8 x 2H followed by 8 biases in `w`) with the flattened
// streams; o == 8 computes their sum of squares. Results go to out[m * 9 + o].
@compute @workgroup_size(256)
fn mhc_dots(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let o = wid.x;
    let m = wid.y;
    let d = 2u * p.h;
    let base = m * d;
    var acc = 0.0;
    if (o < 8u) {
        for (var i = lid; i < d; i += 256u) { acc += w[o * d + i] * x[base + i]; }
    } else {
        for (var i = lid; i < d; i += 256u) { let v = x[base + i]; acc += v * v; }
    }
    let total = reduce_sum(lid, acc);
    if (lid == 0u) { out[m * 9u + o] = total; }
}

// mHC router, step 2: one thread per token. Applies the RMS normalization and
// bias, then coef = [pre0, pre1, post0, post1, mix00, mix01, mix10, mix11]
// with the 2x2 mixing matrix Sinkhorn-normalized in log space.
@compute @workgroup_size(256)
fn mhc_finish(@builtin(global_invocation_id) gid: vec3<u32>) {
    let m = gid.x;
    if (m >= step.m) { return; }
    let d = 2u * p.h;
    let rinv = inverseSqrt(out[m * 9u + 8u] / f32(d) + 1e-6);
    var logits: array<f32, 8>;
    for (var o = 0u; o < 8u; o++) { logits[o] = out[m * 9u + o] * rinv + w[8u * d + o]; }
    var a = logits[4];
    var b = logits[5];
    var c = logits[6];
    var e = logits[7];
    for (var it = 0u; it < p.iterations; it++) {
        let r0 = max(a, b) + log(exp(a - max(a, b)) + exp(b - max(a, b)));
        let r1 = max(c, e) + log(exp(c - max(c, e)) + exp(e - max(c, e)));
        a -= r0; b -= r0; c -= r1; e -= r1;
        let c0 = max(a, c) + log(exp(a - max(a, c)) + exp(c - max(a, c)));
        let c1 = max(b, e) + log(exp(b - max(b, e)) + exp(e - max(b, e)));
        a -= c0; c -= c0; b -= c1; e -= c1;
    }
    let cb = m * 8u;
    coef[cb + 0u] = 1.0 / (1.0 + exp(-logits[0]));
    coef[cb + 1u] = 1.0 / (1.0 + exp(-logits[1]));
    coef[cb + 2u] = 2.0 / (1.0 + exp(-logits[2]));
    coef[cb + 3u] = 2.0 / (1.0 + exp(-logits[3]));
    coef[cb + 4u] = exp(a);
    coef[cb + 5u] = exp(b);
    coef[cb + 6u] = exp(c);
    coef[cb + 7u] = exp(e);
}

// Residual update, one thread per (token, hidden index).
// mode 0: x0 += branch
// mode 1: mHC merge: x_s = sum_t mix[s,t] x_t + post_s * branch
// mode 2: mHC merge, then collapse streams to their mean in stream 0
// mode 3: expand stream 0 into stream 1 (entering the mHC range)
// `out` holds the branch output [M, H].
@compute @workgroup_size(256)
fn merge(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.x + gid.y * nwg.x * 256u;
    if (idx >= step.m * p.h) { return; }
    let m = idx / p.h;
    let i = idx % p.h;
    let b0 = m * 2u * p.h + i;
    let b1 = b0 + p.h;
    if (p.mode == 0u) {
        x[b0] += out[idx];
        return;
    }
    if (p.mode == 3u) {
        x[b1] = x[b0];
        return;
    }
    let c = m * 8u;
    let x0 = x[b0];
    let x1 = x[b1];
    let br = out[idx];
    let n0 = coef[c + 4u] * x0 + coef[c + 5u] * x1 + coef[c + 2u] * br;
    let n1 = coef[c + 6u] * x0 + coef[c + 7u] * x1 + coef[c + 3u] * br;
    if (p.mode == 2u) {
        x[b0] = 0.5 * (n0 + n1);
    } else {
        x[b0] = n0;
        x[b1] = n1;
    }
}

// act[m, i] = silu(gate[m, i]) * up[m, i]; `x` holds [M, 2I] gate|up rows,
// `out` receives [M, I].
@compute @workgroup_size(256)
fn swiglu(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.x + gid.y * nwg.x * 256u;
    if (idx >= step.m * p.inter) { return; }
    let m = idx / p.inter;
    let i = idx % p.inter;
    let g = x[m * 2u * p.inter + i];
    let u = x[m * 2u * p.inter + p.inter + i];
    out[idx] = g / (1.0 + exp(-g)) * u;
}
