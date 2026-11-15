//go:build metal

package metal

// mslSource is the Metal Shading Language source for the B1 decode kernels plus the B2 additions (add, matmul_vec_q4_0, matmul_vec_q4_k) and the B3 prefill GEMM (matmul_mat_q8_0), compiled at runtime by runtime_darwin.go (newLibraryWithSource).
// Kernel names are listed in B1Kernels / B2Kernels / B3Kernels; struct layouts must match the param structs in driver_darwin.m (all 4-byte scalars).
//
// Launch geometry (see driver_darwin.m):
//   - matmul_vec_*: one simdgroup per output row, 4 rows per threadgroup (threadgroup = 4 * threadExecutionWidth threads)
//   - matmul_mat_q8_0: one thread per (batch, row) output, grid = batch * rows rounded up to whole threadgroups (kernel bounds-checks)
//   - rmsnorm / attn_qk_softmax_v: one threadgroup per vector / per query head, reductions via simd_sum/simd_max + threadgroup scratch (rmsnorm normalizes `rows` consecutive dim-vectors, one threadgroup each: rows = nHeads is the per-head QK-norm, rows = 1 the plain norm)
//   - rope_heads / swiglu / add: one thread per element pair / element, grid rounded up to whole threadgroups (kernels bounds-check)
const mslSource = `
#include <metal_stdlib>
using namespace metal;

#define ROWS_PER_TG 4

struct MatParams {
    uint rows;
    uint cols;
};

struct MatMatParams {
    uint rows;
    uint cols;
    uint batch;
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

struct AddParams {
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

// block_q4_0 as read by pkg/quant (BlockQ4_0Size = 18): fp16 scale + 16 bytes, element i is nibble (i & 1) of byte i / 2, value = d * (nibble - 8)
struct block_q4_0 {
    half d;
    uchar qs[16];
};
static_assert(sizeof(block_q4_0) == 18, "block_q4_0 must match pkg/quant BlockQ4_0Size");

// ggml block_q4_K (pkg/quant BlockQ4_KSize = 144): fp16 d, fp16 dmin, 12 bytes of packed 6-bit scales/mins, 128 bytes of nibbles (256 values)
struct block_q4_K {
    half d;
    half dmin;
    uchar scales[12];
    uchar qs[128];
};
static_assert(sizeof(block_q4_K) == 144, "block_q4_K must match the ggml Q4_K layout");

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

// ---- matmul_mat_q8_0 (B3 prefill GEMM): W[rows][cols] * X[batch][cols] -> out[batch][rows].
// CUDA-parity thread-per-output algorithm (cuda/ptx.go matmul_mat_q8_0): thread gid = b * rows + r owns out[b * rows + r] and walks the Q8_0 row block by block (per block: sum of q * x, then * d, accumulated in block order like compute.MatMulMatQ8_0Into). cols % 32 == 0.
// Not simdgroup_matrix: that is a follow-up (needs tile staging); this kernel is correct everywhere and one launch covers the whole batch. ----

kernel void matmul_mat_q8_0(
    device const block_q8_0 *W [[buffer(0)]],
    device const float *X [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant MatMatParams &p [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if ((ulong)gid >= (ulong)p.rows * (ulong)p.batch) {
        return;
    }

    uint r = gid % p.rows;
    uint b = gid / p.rows;
    uint nb = p.cols >> 5;
    device const block_q8_0 *blk = W + (ulong)r * nb;
    device const float *x = X + (ulong)b * p.cols;

    float sum = 0.0f;
    for (uint i = 0; i < nb; i++) {
        float d = float(blk[i].d);
        float s = 0.0f;
        for (uint j = 0; j < 32; j++) {
            s += float(blk[i].qs[j]) * x[(i << 5) + j];
        }
        sum += d * s;
    }

    out[(ulong)b * p.rows + r] = sum;
}

// ---- matmul_vec_q4_0: fused dequant (d * (nibble - 8)) + dot, cols % 32 == 0 ----

kernel void matmul_vec_q4_0(
    device const block_q4_0 *W [[buffer(0)]],
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
    device const block_q4_0 *blk = W + (ulong)row * nb;
    float sum = 0.0f;
    for (uint i = lane; i < p.cols; i += sgw) {
        device const block_q4_0 &b = blk[i >> 5];
        uint q = (uint(b.qs[(i & 31u) >> 1]) >> (4u * (i & 1u))) & 0xFu;
        sum += float(b.d) * float(int(q) - 8) * x[i];
    }

    sum = simd_sum(sum);
    if (lane == 0) {
        out[row] = sum;
    }
}

// ---- matmul_vec_q4_k: ggml Q4_K super-blocks (256 values), cols % 256 == 0.
// Per 64-value group g: low nibbles of qs[g*32 .. +32) are values g*64 + l (scale/min 2g), high nibbles are values g*64 + 32 + l (scale/min 2g+1).
// Lanes of the simdgroup own l = lane, lane + width, ... so qs reads are contiguous. ----

static void get_scale_min_k4(uint j, device const uchar *q, thread uint &sc, thread uint &mn) {
    if (j < 4) {
        sc = uint(q[j]) & 63u;
        mn = uint(q[j + 4]) & 63u;
    } else {
        sc = (uint(q[j + 4]) & 0xFu) | ((uint(q[j - 4]) >> 6) << 4);
        mn = (uint(q[j + 4]) >> 4) | ((uint(q[j]) >> 6) << 4);
    }
}

kernel void matmul_vec_q4_k(
    device const block_q4_K *W [[buffer(0)]],
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

    uint nb = p.cols >> 8;
    device const block_q4_K *blk = W + (ulong)row * nb;
    float sum = 0.0f;
    for (uint sb = 0; sb < nb; sb++) {
        device const block_q4_K &b = blk[sb];
        float d = float(b.d);
        float dmin = float(b.dmin);
        device const uchar *sc = &b.scales[0];
        device const float *xs = x + (ulong)sb * 256u;
        for (uint g = 0; g < 4; g++) {
            uint s1, m1, s2, m2;
            get_scale_min_k4(2 * g, sc, s1, m1);
            get_scale_min_k4(2 * g + 1, sc, s2, m2);
            float d1 = d * float(s1);
            float d2 = d * float(s2);
            float mm1 = dmin * float(m1);
            float mm2 = dmin * float(m2);
            for (uint l = lane; l < 32; l += sgw) {
                uint q = b.qs[g * 32 + l];
                sum += (d1 * float(q & 0xFu) - mm1) * xs[g * 64 + l];
                sum += (d2 * float(q >> 4) - mm2) * xs[g * 64 + 32 + l];
            }
        }
    }

    sum = simd_sum(sum);
    if (lane == 0) {
        out[row] = sum;
    }
}

// ---- rmsnorm: dst = x * rsqrt(mean(x^2) + eps) * weight (dst may alias x).
// One threadgroup per dim-vector: threadgroup r normalizes x[r*dim ..) into dst[r*dim ..) with the shared weight[dim] (grid of 1 = plain RMSNorm, grid of nHeads = per-head QK-norm) ----

kernel void rmsnorm(
    device const float *xIn [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device float *dstOut [[buffer(2)]],
    constant RMSParams &p [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgSize [[threads_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sgw [[threads_per_simdgroup]])
{
    threadgroup float partial[32];
    uint nsg = (tgSize + sgw - 1) / sgw;

    device const float *x = xIn + (ulong)row * p.dim;
    device float *dst = dstOut + (ulong)row * p.dim;

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

// ---- add: a[i] += b[i] (residual add on the device-resident hidden state) ----

kernel void add(
    device float *a [[buffer(0)]],
    device const float *b [[buffer(1)]],
    constant AddParams &p [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.n) {
        return;
    }

    a[gid] += b[gid];
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
