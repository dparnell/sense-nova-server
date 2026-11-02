// y[m, n_off + n] = sum_k x[m, k] * W[n, k] for a block-quantized weight W.
//
// Two entry points share one binding layout:
//   * `matvec` for small M (decode): 8 rows per workgroup, 32 lanes per row.
//   * `matmul` for larger M (prefill): 64x64 output tiles staged in shared memory.
//
// The quantization format is spliced in by the host: it provides
// `BLOCK_VEC4S` (vec4<u32> words per 32-element block), `block_dot` and
// `dequant8` (8 consecutive weights of one block, `part` in 0..4).

struct Step {
    m: u32,
    pos0: u32,
    parts: u32,
    _pad: u32,
}

struct Params {
    n: u32,          // rows in this weight chunk
    k: u32,
    x_off: u32,      // in floats, multiple of 4
    x_stride: u32,   // in floats, multiple of 4
    y_off: u32,
    y_stride: u32,
    n_off: u32,      // first output column written by this chunk
    m_one: u32,      // > 0: ignore step.m and use this many rows
}

@group(0) @binding(0) var<uniform> step: Step;
@group(0) @binding(1) var<uniform> p: Params;
@group(0) @binding(2) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> wq: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read> ws: array<f32>;
@group(0) @binding(5) var<storage, read_write> y: array<f32>;

fn rows_m() -> u32 {
    if (p.m_one > 0u) { return p.m_one; }
    return step.m;
}

fn i8x4(w: u32) -> vec4<f32> {
    return vec4<f32>(
        f32(i32(w << 24u) >> 24u),
        f32(i32(w << 16u) >> 24u),
        f32(i32(w << 8u) >> 24u),
        f32(i32(w) >> 24u));
}

fn u4x4(w: u32, shift: u32) -> vec4<f32> {
    return vec4<f32>(
        f32((w >> shift) & 15u),
        f32((w >> (shift + 4u)) & 15u),
        f32((w >> (shift + 8u)) & 15u),
        f32((w >> (shift + 12u)) & 15u)) - vec4<f32>(8.0);
}

//QUANT_FNS//

const MV_TOKENS: u32 = 8u;
var<workgroup> red: array<f32, 2048>;

@compute @workgroup_size(256)
fn matvec(@builtin(workgroup_id) wid: vec3<u32>,
          @builtin(num_workgroups) nwg: vec3<u32>,
          @builtin(local_invocation_index) lid: u32) {
    let group = wid.x + wid.y * nwg.x;
    let row = group * 8u + lid / 32u;
    let lane = lid % 32u;
    let m_rows = min(rows_m(), MV_TOKENS);
    let nb = p.k / 32u;

    var acc: array<f32, 8>;
    for (var t = 0u; t < MV_TOKENS; t++) { acc[t] = 0.0; }

    if (row < p.n) {
        for (var b = lane; b < nb; b += 32u) {
            let scale = ws[row * nb + b];
            let wbase = (row * nb + b) * BLOCK_VEC4S;
            for (var t = 0u; t < m_rows; t++) {
                let xbase = (p.x_off + t * p.x_stride) / 4u + b * 8u;
                acc[t] += scale * block_dot(wbase, xbase);
            }
        }
    }
    for (var t = 0u; t < MV_TOKENS; t++) { red[t * 256u + lid] = acc[t]; }
    workgroupBarrier();
    for (var s = 16u; s > 0u; s >>= 1u) {
        if (lane < s) {
            for (var t = 0u; t < MV_TOKENS; t++) {
                red[t * 256u + lid] += red[t * 256u + lid + s];
            }
        }
        workgroupBarrier();
    }
    if (lane == 0u && row < p.n) {
        for (var t = 0u; t < m_rows; t++) {
            y[p.y_off + t * p.y_stride + p.n_off + row] = red[t * 256u + lid];
        }
    }
}

// One- and two-row paths (decode and speculative verification). Each 32-lane
// group owns MV_ROWS weight rows: a lane loads a block of x into registers once
// and applies it to all of its rows, which keeps the kernel bound by weight
// bandwidth rather than by re-reading x for every row.
const MV_ROWS: u32 = 4u;
var<workgroup> red1: array<f32, 1024>;
var<workgroup> red2: array<f32, 1024>;

fn load_x(base: u32) -> array<vec4<f32>, 8> {
    return array<vec4<f32>, 8>(x[base], x[base + 1u], x[base + 2u], x[base + 3u],
                               x[base + 4u], x[base + 5u], x[base + 6u], x[base + 7u]);
}

// Sum lane partials of `red1` (and `red2`) in place; lane 0 of each row ends with the total.
fn reduce_rows(lid: u32, lane: u32, two: bool) {
    workgroupBarrier();
    for (var s = 16u; s > 0u; s >>= 1u) {
        if (lane < s) {
            for (var r = 0u; r < MV_ROWS; r++) {
                red1[r * 256u + lid] += red1[r * 256u + lid + s];
                if (two) { red2[r * 256u + lid] += red2[r * 256u + lid + s]; }
            }
        }
        workgroupBarrier();
    }
}

@compute @workgroup_size(256)
fn matvec1(@builtin(workgroup_id) wid: vec3<u32>,
           @builtin(num_workgroups) nwg: vec3<u32>,
           @builtin(local_invocation_index) lid: u32) {
    let group = wid.x + wid.y * nwg.x;
    let row0 = (group * 8u + lid / 32u) * MV_ROWS;
    let lane = lid % 32u;
    let nb = p.k / 32u;
    let xb = p.x_off / 4u;
    var acc: array<f32, 4>;
    for (var r = 0u; r < MV_ROWS; r++) { acc[r] = 0.0; }
    if (row0 < p.n) {
        let rows = min(MV_ROWS, p.n - row0);
        for (var b = lane; b < nb; b += 32u) {
            var xv = load_x(xb + b * 8u);
            for (var r = 0u; r < rows; r++) {
                let rb = (row0 + r) * nb + b;
                acc[r] += ws[rb] * block_dot_v(rb * BLOCK_VEC4S, &xv);
            }
        }
    }
    for (var r = 0u; r < MV_ROWS; r++) { red1[r * 256u + lid] = acc[r]; }
    reduce_rows(lid, lane, false);
    if (lane == 0u) {
        for (var r = 0u; r < MV_ROWS; r++) {
            if (row0 + r < p.n) { y[p.y_off + p.n_off + row0 + r] = red1[r * 256u + lid]; }
        }
    }
}

@compute @workgroup_size(256)
fn matvec2(@builtin(workgroup_id) wid: vec3<u32>,
           @builtin(num_workgroups) nwg: vec3<u32>,
           @builtin(local_invocation_index) lid: u32) {
    let group = wid.x + wid.y * nwg.x;
    let row0 = (group * 8u + lid / 32u) * MV_ROWS;
    let lane = lid % 32u;
    let nb = p.k / 32u;
    let xa = p.x_off / 4u;
    let xc = (p.x_off + p.x_stride) / 4u;
    var acc0: array<f32, 4>;
    var acc1: array<f32, 4>;
    for (var r = 0u; r < MV_ROWS; r++) { acc0[r] = 0.0; acc1[r] = 0.0; }
    if (row0 < p.n) {
        let rows = min(MV_ROWS, p.n - row0);
        for (var b = lane; b < nb; b += 32u) {
            var x0 = load_x(xa + b * 8u);
            var x1 = load_x(xc + b * 8u);
            for (var r = 0u; r < rows; r++) {
                let rb = (row0 + r) * nb + b;
                let s = ws[rb];
                acc0[r] += s * block_dot_v(rb * BLOCK_VEC4S, &x0);
                acc1[r] += s * block_dot_v(rb * BLOCK_VEC4S, &x1);
            }
        }
    }
    for (var r = 0u; r < MV_ROWS; r++) {
        red1[r * 256u + lid] = acc0[r];
        red2[r * 256u + lid] = acc1[r];
    }
    reduce_rows(lid, lane, true);
    if (lane == 0u) {
        for (var r = 0u; r < MV_ROWS; r++) {
            if (row0 + r < p.n) {
                y[p.y_off + p.n_off + row0 + r] = red1[r * 256u + lid];
                y[p.y_off + p.y_stride + p.n_off + row0 + r] = red2[r * 256u + lid];
            }
        }
    }
}

// Tiled matmul for prefill: 128 (n) x 128 (m) outputs per workgroup. Thread
// (tx, ty) owns rows {tx*4..+4, 64+tx*4..+4} x tokens {ty*4..+4, 64+ty*4..+4},
// accumulated in four explicit 4x4 matrices so everything stays in registers.
// k advances 16 at a time; the next slice is prefetched into registers.
const TK: u32 = 16u;
var<workgroup> w_tile: array<vec4<f32>, 512>; // [TK][128 / 4]
var<workgroup> x_tile: array<vec4<f32>, 512>; // [TK][128 / 4]

struct Slice {
    wa: vec4<f32>,
    wb: vec4<f32>,
    xa: vec4<f32>,
    xb: vec4<f32>,
}

fn load_slice(step_k: u32, r: u32, half: u32, n0: u32, m0: u32, m_rows: u32, nb: u32) -> Slice {
    var sl: Slice;
    let b = step_k / 2u;
    let part = (step_k % 2u) * 2u + half;
    let wrow = n0 + r;
    sl.wa = vec4<f32>(0.0);
    sl.wb = vec4<f32>(0.0);
    if (wrow < p.n) {
        let w = dequant8((wrow * nb + b) * BLOCK_VEC4S, part, ws[wrow * nb + b]);
        sl.wa = vec4<f32>(w[0], w[1], w[2], w[3]);
        sl.wb = vec4<f32>(w[4], w[5], w[6], w[7]);
    }
    let xrow = m0 + r;
    sl.xa = vec4<f32>(0.0);
    sl.xb = vec4<f32>(0.0);
    if (xrow < m_rows) {
        let xbase = (p.x_off + xrow * p.x_stride) / 4u + b * 8u + part * 2u;
        sl.xa = x[xbase];
        sl.xb = x[xbase + 1u];
    }
    return sl;
}

fn outer(a: vec4<f32>, b: vec4<f32>) -> mat4x4<f32> {
    return mat4x4<f32>(a * b.x, a * b.y, a * b.z, a * b.w);
}

@compute @workgroup_size(16, 16)
fn matmul(@builtin(workgroup_id) wid: vec3<u32>,
          @builtin(local_invocation_id) l: vec3<u32>,
          @builtin(local_invocation_index) lid: u32) {
    let n0 = wid.x * 128u;
    let m0 = wid.y * 128u;
    let m_rows = rows_m();
    let nb = p.k / 32u;
    let steps = p.k / TK;

    var c00 = mat4x4<f32>();
    var c01 = mat4x4<f32>();
    var c10 = mat4x4<f32>();
    var c11 = mat4x4<f32>();

    let r = lid / 2u;    // tile row (weight row / token) this thread loads
    let half = lid % 2u; // which 8 of the 16 k values
    let col = r / 4u;
    let comp = r % 4u;
    var cur = load_slice(0u, r, half, n0, m0, m_rows, nb);

    for (var s = 0u; s < steps; s++) {
        let kb = half * 8u;
        w_tile[(kb + 0u) * 32u + col][comp] = cur.wa.x;
        w_tile[(kb + 1u) * 32u + col][comp] = cur.wa.y;
        w_tile[(kb + 2u) * 32u + col][comp] = cur.wa.z;
        w_tile[(kb + 3u) * 32u + col][comp] = cur.wa.w;
        w_tile[(kb + 4u) * 32u + col][comp] = cur.wb.x;
        w_tile[(kb + 5u) * 32u + col][comp] = cur.wb.y;
        w_tile[(kb + 6u) * 32u + col][comp] = cur.wb.z;
        w_tile[(kb + 7u) * 32u + col][comp] = cur.wb.w;
        x_tile[(kb + 0u) * 32u + col][comp] = cur.xa.x;
        x_tile[(kb + 1u) * 32u + col][comp] = cur.xa.y;
        x_tile[(kb + 2u) * 32u + col][comp] = cur.xa.z;
        x_tile[(kb + 3u) * 32u + col][comp] = cur.xa.w;
        x_tile[(kb + 4u) * 32u + col][comp] = cur.xb.x;
        x_tile[(kb + 5u) * 32u + col][comp] = cur.xb.y;
        x_tile[(kb + 6u) * 32u + col][comp] = cur.xb.z;
        x_tile[(kb + 7u) * 32u + col][comp] = cur.xb.w;
        workgroupBarrier();
        if (s + 1u < steps) {
            cur = load_slice(s + 1u, r, half, n0, m0, m_rows, nb);
        }
        for (var k = 0u; k < TK; k++) {
            let w0 = w_tile[k * 32u + l.x];
            let w1 = w_tile[k * 32u + 16u + l.x];
            let x0 = x_tile[k * 32u + l.y];
            let x1 = x_tile[k * 32u + 16u + l.y];
            c00 += outer(w0, x0);
            c01 += outer(w0, x1);
            c10 += outer(w1, x0);
            c11 += outer(w1, x1);
        }
        workgroupBarrier();
    }

    // c_ab[j][i]: weight row group a (i within), token group b (j within).
    let na = n0 + l.x * 4u;
    let nb2 = n0 + 64u + l.x * 4u;
    let ma = m0 + l.y * 4u;
    let mb = m0 + 64u + l.y * 4u;
    for (var j = 0u; j < 4u; j++) {
        store4(ma + j, na, c00[j], m_rows);
        store4(mb + j, na, c01[j], m_rows);
        store4(ma + j, nb2, c10[j], m_rows);
        store4(mb + j, nb2, c11[j], m_rows);
    }
}

fn store4(m: u32, n: u32, v: vec4<f32>, m_rows: u32) {
    if (m >= m_rows) { return; }
    let base = p.y_off + m * p.y_stride + p.n_off;
    if (n < p.n) { y[base + n] = v.x; }
    if (n + 1u < p.n) { y[base + n + 1u] = v.y; }
    if (n + 2u < p.n) { y[base + n + 2u] = v.z; }
    if (n + 3u < p.n) { y[base + n + 3u] = v.w; }
}
