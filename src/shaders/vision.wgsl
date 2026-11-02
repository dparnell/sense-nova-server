// Vision tower kernels. Matrix products reuse matmul.wgsl; these are the pieces around them.
//
// Token rows of one frame are in merge order: each run of 4 rows is a 2x2 block
// of patches, blocks in row-major order over a (grid_h / 2) x (grid_w / 2) grid.

struct Params {
    rows: u32,
    cols: u32,
    mode: u32,
    act: u32,
    eps: f32,
    grid_w: u32,   // patches per image row (rope)
    heads: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> a: array<f32>;
@group(0) @binding(2) var<storage, read_write> b: array<f32>;
@group(0) @binding(3) var<storage, read> w: array<f32>;
@group(0) @binding(4) var<storage, read> e: array<f32>;

const VHD: u32 = 72u;     // head dim
const VHD4: u32 = 18u;    // head dim in vec4s
const HALF: u32 = 36u;    // rotary pairs (j, j + 36)

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

// b[r] = LayerNorm(a[r]) with w = weight[cols] | bias[cols]. One workgroup per row.
@compute @workgroup_size(256)
fn layer_norm(@builtin(workgroup_id) wid: vec3<u32>,
              @builtin(num_workgroups) nwg: vec3<u32>,
              @builtin(local_invocation_index) lid: u32) {
    let r = wid.x + wid.y * nwg.x;
    if (r >= p.rows) { return; }
    let base = r * p.cols;
    var s = 0.0;
    for (var c = lid; c < p.cols; c += 256u) { s += a[base + c]; }
    let mean = reduce_sum(lid, s) / f32(p.cols);
    var v = 0.0;
    for (var c = lid; c < p.cols; c += 256u) {
        let d = a[base + c] - mean;
        v += d * d;
    }
    let inv = inverseSqrt(reduce_sum(lid, v) / f32(p.cols) + p.eps);
    for (var c = lid; c < p.cols; c += 256u) {
        b[base + c] = (a[base + c] - mean) * inv * w[c] + w[p.cols + c];
    }
}

fn gelu_tanh(x: f32) -> f32 {
    let u = clamp(0.7978845608 * (x + 0.044715 * x * x * x), -15.0, 15.0);
    return 0.5 * x * (1.0 + tanh(u));
}

// Abramowitz & Stegun 7.1.26, |error| < 1.5e-7.
fn erf(x: f32) -> f32 {
    let t = 1.0 / (1.0 + 0.3275911 * abs(x));
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-x * x);
    return select(-y, y, x >= 0.0);
}

fn gelu_erf(x: f32) -> f32 {
    return 0.5 * x * (1.0 + erf(x * 0.7071067812));
}

// Bias, activation and residual after a matmul into `a` ([rows, cols], w = bias[cols]):
//   mode 0: a = act(a + bias)
//   mode 1: b += a + bias
//   mode 2: b = a + bias + e
@compute @workgroup_size(256)
fn epilogue(@builtin(workgroup_id) wid: vec3<u32>,
            @builtin(num_workgroups) nwg: vec3<u32>,
            @builtin(local_invocation_index) lid: u32) {
    let i = (wid.x + wid.y * nwg.x) * 256u + lid;
    if (i >= p.rows * p.cols) { return; }
    var v = a[i] + w[i % p.cols];
    if (p.act == 1u) { v = gelu_tanh(v); } else if (p.act == 2u) { v = gelu_erf(v); }
    if (p.mode == 0u) {
        a[i] = v;
    } else if (p.mode == 1u) {
        b[i] = b[i] + v;
    } else {
        b[i] = v + e[i];
    }
}

// 2D rotary embedding on q and k in the fused qkv rows ([rows, 3, heads, 72]).
// Pair j < 18 rotates by the patch row, 18 <= j < 36 by the patch column.
@compute @workgroup_size(256)
fn rope(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let i = (wid.x + wid.y * nwg.x) * 256u + lid;
    let per_token = p.heads * HALF;
    if (i >= p.rows * per_token) { return; }
    let token = i / per_token;
    let h = (i % per_token) / HALF;
    let j = i % HALF;
    let mw = p.grid_w / 2u;
    let blk = token / 4u;
    let intra = token % 4u;
    let row = (blk / mw) * 2u + intra / 2u;
    let col = (blk % mw) * 2u + intra % 2u;
    let fi = j % (HALF / 2u);
    let pos = select(row, col, j >= HALF / 2u);
    let ang = f32(pos) * pow(10000.0, -f32(2u * fi) / f32(HALF));
    let cs = cos(ang);
    let sn = sin(ang);
    let hidden = p.heads * VHD;
    for (var part = 0u; part < 2u; part++) {
        let base = token * 3u * hidden + part * hidden + h * VHD;
        let x1 = a[base + j];
        let x2 = a[base + j + HALF];
        a[base + j] = x1 * cs - x2 * sn;
        a[base + j + HALF] = x2 * cs + x1 * sn;
    }
}

const TQ: u32 = 64u;
const TK: u32 = 16u;
var<workgroup> ks: array<vec4<f32>, 288>;  // TK keys x 18 vec4
var<workgroup> vs: array<vec4<f32>, 288>;

fn load4(i: u32) -> vec4<f32> {
    return vec4<f32>(a[i], a[i + 1u], a[i + 2u], a[i + 3u]);
}

// Bidirectional attention over all rows of one frame. Workgroup (query tile, head);
// each thread owns one query and keeps an online softmax over 16-key tiles.
@compute @workgroup_size(64)
fn attend(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let h = wid.y;
    let qi = wid.x * TQ + lid;
    let n = p.rows;
    let hidden = p.heads * VHD;
    let stride = 3u * hidden;
    let valid = qi < n;

    var q: array<vec4<f32>, 18>;
    var acc: array<vec4<f32>, 18>;
    let scale = inverseSqrt(f32(VHD));
    for (var d = 0u; d < VHD4; d++) {
        acc[d] = vec4<f32>(0.0);
        q[d] = vec4<f32>(0.0);
        if (valid) { q[d] = load4(qi * stride + h * VHD + d * 4u) * scale; }
    }
    var mx = -1e30;
    var l = 0.0;
    var sc: array<f32, 16>;
    for (var k0 = 0u; k0 < n; k0 += TK) {
        workgroupBarrier();
        for (var t = lid; t < TK * VHD4; t += TQ) {
            let key = k0 + t / VHD4;
            let d = t % VHD4;
            if (key < n) {
                let o = key * stride + hidden + h * VHD + d * 4u;
                ks[t] = load4(o);
                vs[t] = load4(o + hidden);
            } else {
                ks[t] = vec4<f32>(0.0);
                vs[t] = vec4<f32>(0.0);
            }
        }
        workgroupBarrier();
        let cnt = min(TK, n - k0);
        var tmax = mx;
        for (var j = 0u; j < cnt; j++) {
            var s = 0.0;
            for (var d = 0u; d < VHD4; d++) { s += dot(q[d], ks[j * VHD4 + d]); }
            sc[j] = s;
            tmax = max(tmax, s);
        }
        let corr = exp(mx - tmax);
        l *= corr;
        for (var d = 0u; d < VHD4; d++) { acc[d] *= corr; }
        for (var j = 0u; j < cnt; j++) {
            let pj = exp(sc[j] - tmax);
            l += pj;
            for (var d = 0u; d < VHD4; d++) { acc[d] += pj * vs[j * VHD4 + d]; }
        }
        mx = tmax;
    }
    if (valid) {
        let o = qi * hidden + h * VHD;
        for (var d = 0u; d < VHD4; d++) {
            let v = acc[d] / l;
            b[o + d * 4u] = v.x;
            b[o + d * 4u + 1u] = v.y;
            b[o + d * 4u + 2u] = v.z;
            b[o + d * 4u + 3u] = v.w;
        }
    }
}
