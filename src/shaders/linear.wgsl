// Gated DeltaNet linear attention.
//
// `proj` rows hold the fused in_proj output: qkv[conv_dim] | z[v_dim] | b[nv] | a[nv].
// qkv is split as q[nk * dk] | k[nk * dk] | v[nv * dv] after the causal conv.

struct Step {
    m: u32,
    pos0: u32,
    parts: u32,
    _pad: u32,
}

struct Params {
    conv_dim: u32,
    proj_stride: u32,
    n_k: u32,
    n_v: u32,
    eps: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

const DK: u32 = 128u;
const DV: u32 = 128u;

@group(0) @binding(0) var<uniform> step: Step;
@group(0) @binding(1) var<uniform> p: Params;
@group(0) @binding(2) var<storage, read_write> proj: array<f32>;
@group(0) @binding(3) var<storage, read_write> conv_out: array<f32>;   // [M, conv_dim]
@group(0) @binding(4) var<storage, read_write> conv_state: array<f32>; // [3, conv_dim]
@group(0) @binding(5) var<storage, read_write> state: array<f32>;      // [n_v, DK, DV]
@group(0) @binding(6) var<storage, read> lw: array<f32>;              // conv w [conv_dim*4] | A_log[nv] | dt_bias[nv] | norm[DV]
@group(0) @binding(7) var<storage, read_write> out: array<f32>;        // [M, n_v * DV]

// Depthwise causal conv (kernel 4) + SiLU, one thread per channel walking the tokens.
@compute @workgroup_size(256)
fn conv(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if (c >= p.conv_dim) { return; }
    let w0 = lw[c * 4u];
    let w1 = lw[c * 4u + 1u];
    let w2 = lw[c * 4u + 2u];
    let w3 = lw[c * 4u + 3u];
    var r0 = conv_state[c];
    var r1 = conv_state[p.conv_dim + c];
    var r2 = conv_state[2u * p.conv_dim + c];
    for (var m = 0u; m < step.m; m++) {
        let cur = proj[m * p.proj_stride + c];
        let o = w0 * r0 + w1 * r1 + w2 * r2 + w3 * cur;
        conv_out[m * p.conv_dim + c] = o / (1.0 + exp(-o));
        r0 = r1;
        r1 = r2;
        r2 = cur;
    }
    conv_state[c] = r0;
    conv_state[p.conv_dim + c] = r1;
    conv_state[2u * p.conv_dim + c] = r2;
}

var<workgroup> qs: array<f32, 128>;
var<workgroup> ks: array<f32, 128>;
var<workgroup> red: array<vec2<f32>, 128>;

fn reduce2(j: u32, v: vec2<f32>) -> vec2<f32> {
    red[j] = v;
    workgroupBarrier();
    for (var s = 64u; s > 0u; s >>= 1u) {
        if (j < s) { red[j] += red[j + s]; }
        workgroupBarrier();
    }
    let r = red[0];
    workgroupBarrier();
    return r;
}

// Recurrent gated delta rule + gated RMS norm. Workgroup per value head,
// thread j owns column j of the head's [DK, DV] state.
@compute @workgroup_size(128)
fn delta(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) j: u32) {
    let h = wid.x;
    let kh = h / (p.n_v / p.n_k);
    let key_dim = p.n_k * DK;
    let sbase = h * DK * DV;
    let a_log = lw[p.conv_dim * 4u + h];
    let dt_bias = lw[p.conv_dim * 4u + p.n_v + h];
    let norm_w = lw[p.conv_dim * 4u + 2u * p.n_v + j];
    let z_off = p.conv_dim;
    let b_off = p.conv_dim + p.n_v * DV;
    let a_off = b_off + p.n_v;

    var s: array<f32, 128>;
    for (var i = 0u; i < DK; i++) { s[i] = state[sbase + i * DV + j]; }

    for (var m = 0u; m < step.m; m++) {
        let crow = m * p.conv_dim;
        let qv = conv_out[crow + kh * DK + j];
        let kv = conv_out[crow + key_dim + kh * DK + j];
        let vv = conv_out[crow + 2u * key_dim + h * DV + j];
        let sums = reduce2(j, vec2<f32>(qv * qv, kv * kv));
        qs[j] = qv * inverseSqrt(sums.x + 1e-6) * inverseSqrt(f32(DK));
        ks[j] = kv * inverseSqrt(sums.y + 1e-6);
        workgroupBarrier();

        let prow = m * p.proj_stride;
        let beta = 1.0 / (1.0 + exp(-proj[prow + b_off + h]));
        let araw = proj[prow + a_off + h] + dt_bias;
        var sp = araw;
        if (araw <= 20.0) { sp = log(1.0 + exp(araw)); }
        let decay = exp(-exp(a_log) * sp);

        var kv_mem = 0.0;
        for (var i = 0u; i < DK; i++) {
            s[i] *= decay;
            kv_mem += s[i] * ks[i];
        }
        let dlt = (vv - kv_mem) * beta;
        var o = 0.0;
        for (var i = 0u; i < DK; i++) {
            s[i] += ks[i] * dlt;
            o += s[i] * qs[i];
        }

        let var_o = reduce2(j, vec2<f32>(o * o, 0.0)).x / f32(DV);
        let z = proj[prow + z_off + h * DV + j];
        out[m * p.n_v * DV + h * DV + j] = o * inverseSqrt(var_o + p.eps) * norm_w * (z / (1.0 + exp(-z)));
    }

    for (var i = 0u; i < DK; i++) { state[sbase + i * DV + j] = s[i]; }
}
