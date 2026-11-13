//go:build metal

package metal

// mslSource is the Metal Shading Language source for the B1 decode kernels, compiled at runtime by runtime_darwin.go (newLibraryWithSource).
// Kernel names are listed in B1Kernels; struct layouts must match the param structs in driver_darwin.m (all 4-byte scalars).
//
// Launch geometry (see driver_darwin.m):
//   - matmul_vec_*: one simdgroup per output row, 4 rows per threadgroup (threadgroup = 4 * threadExecutionWidth threads)
//   - rmsnorm / attn_qk_softmax_v: one threadgroup per vector / per query head, reductions via simd_sum/simd_max + threadgroup scratch
//   - rope_heads / swiglu: one thread per element pair / element, grid rounded up to whole threadgroups (kernels bounds-check)
const mslSource = `
#include <metal_stdlib>
using namespace metal;

#define ROWS_PER_TG 4

struct MatParams {
    uint rows;
    uint cols;
};

struct RMSParams {
    uint dim;
    float eps;
};

struct RopeParams {
    uint nHeads;
    uint headDim;
};

struct SwigluParams {
    uint n;
};

struct AttnParams {
    uint seqLen;
    uint nHeads;
    uint nKVHeads;
    uint headDim;
    uint scoresStride;
    float scale;
};

// ggml block_q8_0: fp16 scale + 32 int8 quants (34 bytes, matches pkg/quant.BlockQ8_0Size)
struct block_q8_0 {
    half d;
    char qs[32];
};
static_assert(sizeof(block_q8_0) == 34, "block_q8_0 must match the ggml Q8_0 layout");

// ---- threadgroup reductions (partial has >= 32 floats: threadgroup <= 1024 threads, simd width >= 32) ----

static float tg_sum(float v, threadgroup float *partial, uint lane, uint sg, uint nsg) {
    v = simd_sum(v);
    if (lane == 0) {
        partial[sg] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float t = 0.0f;
    for (uint i = 0; i < nsg; i++) {
        t += partial[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return t;
}

static float tg_max(float v, threadgroup float *partial, uint lane, uint sg, uint nsg) {
    v = simd_max(v);
    if (lane == 0) {
        partial[sg] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float t = partial[0];
    for (uint i = 1; i < nsg; i++) {
        t = max(t, partial[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return t;
}

// ---- matmul_vec_f32: out[row] = dot(W[row][:], x) ----

kernel void matmul_vec_f32(
    device const float *W [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant MatParams &p [[buffer(3)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sgw [[threads_per_simdgroup]])
{
    uint row = tg * ROWS_PER_TG + sg;
    if (row >= p.rows) {
        return;
    }

    device const float *w = W + (ulong)row * p.cols;
    float sum = 0.0f;
    if ((p.cols & 3u) == 0u) {
        device const float4 *w4 = (device const float4 *)w;
        device const float4 *x4 = (device const float4 *)x;
        uint n4 = p.cols >> 2;
        for (uint i = lane; i < n4; i += sgw) {
            sum += dot(w4[i], x4[i]);
        }
    } else {
        for (uint i = lane; i < p.cols; i += sgw) {
            sum += w[i] * x[i];
        }
    }

    sum = simd_sum(sum);
    if (lane == 0) {
        out[row] = sum;
    }
}

// ---- matmul_vec_q8_0: fused dequant (d * q) + dot, cols % 32 == 0 ----

kernel void matmul_vec_q8_0(
    device const block_q8_0 *W [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant MatParams &p [[buffer(3)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sgw [[threads_per_simdgroup]])
{
    uint row = tg * ROWS_PER_TG + sg;
    if (row >= p.rows) {
        return;
    }

    uint nb = p.cols >> 5;
    device const block_q8_0 *blk = W + (ulong)row * nb;
    float sum = 0.0f;
    for (uint i = lane; i < p.cols; i += sgw) {
        device const block_q8_0 &b = blk[i >> 5];
        sum += float(b.d) * float(b.qs[i & 31u]) * x[i];
    }

    sum = simd_sum(sum);
    if (lane == 0) {
        out[row] = sum;
    }
}

// ---- rmsnorm: dst = x * rsqrt(mean(x^2) + eps) * weight (dst may alias x) ----

kernel void rmsnorm(
    device const float *x [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device float *dst [[buffer(2)]],
    constant RMSParams &p [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgSize [[threads_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sgw [[threads_per_simdgroup]])
{
    threadgroup float partial[32];
    uint nsg = (tgSize + sgw - 1) / sgw;

    float s = 0.0f;
    for (uint i = tid; i < p.dim; i += tgSize) {
        float v = x[i];
        s += v * v;
    }

    float total = tg_sum(s, partial, lane, sg, nsg);
    float scale = 1.0f / sqrt(total / float(p.dim) + p.eps);

    for (uint i = tid; i < p.dim; i += tgSize) {
        dst[i] = x[i] * scale * weight[i];
    }
}

// ---- rope_heads: NeoX layout, pairs (i, i + headDim/2); cs = cos[headDim/2] followed by sin[headDim/2] ----

kernel void rope_heads(
    device float *v [[buffer(0)]],
    device const float *cs [[buffer(1)]],
    constant RopeParams &p [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    uint hd2 = p.headDim >> 1;
    if (gid >= p.nHeads * hd2) {
        return;
    }

    uint h = gid / hd2;
    uint i = gid - h * hd2;
    uint base = h * p.headDim;

    float c = cs[i];
    float s = cs[hd2 + i];
    float x0 = v[base + i];
    float x1 = v[base + hd2 + i];
    v[base + i] = x0 * c - x1 * s;
    v[base + hd2 + i] = x0 * s + x1 * c;
}

// ---- swiglu: gate = silu(gate) * up ----

kernel void swiglu(
    device float *gate [[buffer(0)]],
    device const float *up [[buffer(1)]],
    constant SwigluParams &p [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.n) {
        return;
    }

    float g = gate[gid];
    gate[gid] = (g / (1.0f + exp(-g))) * up[gid];
}

// ---- attn_qk_softmax_v: one threadgroup per query head, GQA (kv head = head / (nHeads / nKVHeads)).
// K/V are [seqLen][nKVHeads*headDim] (device KV buffers or uploaded host k/v); scores is [nHeads][scoresStride] scratch. ----

kernel void attn_qk_softmax_v(
    device const float *q [[buffer(0)]],
    device const float *K [[buffer(1)]],
    device const float *V [[buffer(2)]],
    device float *scores [[buffer(3)]],
    device float *out [[buffer(4)]],
    constant AttnParams &p [[buffer(5)]],
    uint h [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgSize [[threads_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sgw [[threads_per_simdgroup]])
{
    threadgroup float partial[32];
    uint nsg = (tgSize + sgw - 1) / sgw;

    uint group = p.nHeads / p.nKVHeads;
    uint kvh = h / group;
    ulong kvDim = (ulong)p.nKVHeads * p.headDim;
    ulong kvOff = (ulong)kvh * p.headDim;

    device const float *qh = q + (ulong)h * p.headDim;
    device float *sc = scores + (ulong)h * p.scoresStride;

    // 1. scaled scores + running max
    float mx = -INFINITY;
    for (uint t = tid; t < p.seqLen; t += tgSize) {
        device const float *kt = K + (ulong)t * kvDim + kvOff;
        float s = 0.0f;
        for (uint d = 0; d < p.headDim; d++) {
            s += qh[d] * kt[d];
        }
        s *= p.scale;
        sc[t] = s;
        mx = max(mx, s);
    }
    mx = tg_max(mx, partial, lane, sg, nsg);

    // 2. exp(s - max) and sum (each thread rewrites only the positions it produced)
    float sum = 0.0f;
    for (uint t = tid; t < p.seqLen; t += tgSize) {
        float e = exp(sc[t] - mx);
        sc[t] = e;
        sum += e;
    }
    sum = tg_sum(sum, partial, lane, sg, nsg);
    threadgroup_barrier(mem_flags::mem_device);

    // 3. out[d] = sum_t softmax(t) * V[t][d]
    float inv = 1.0f / sum;
    for (uint d = tid; d < p.headDim; d += tgSize) {
        float acc = 0.0f;
        for (uint t = 0; t < p.seqLen; t++) {
            acc += sc[t] * V[(ulong)t * kvDim + kvOff + d];
        }
        out[(ulong)h * p.headDim + d] = acc * inv;
    }
}
`
