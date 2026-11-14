#ifndef GGUF_METAL_DRIVER_H
#define GGUF_METAL_DRIVER_H

#include <stddef.h>
#include <stdint.h>

// Plain-C surface over the Objective-C Metal driver (driver_darwin.m).
// Handles are retained Objective-C objects behind void*; every dispatch is synchronous (command buffer completed on return) and returns 0 on success, non-zero with gguf_metal_last_error() text otherwise.
// Buffers are MTLResourceStorageModeShared: gguf_metal_buf_ptr() is CPU-visible memory the GPU reads/writes directly (unified memory).

typedef void *gguf_metal_t;
typedef void *gguf_metal_buf_t;

// Matvec kinds for gguf_metal_matmul_off (weight layouts: F32 rows, ggml Q8_0 / Q4_0 / Q4_K blocks as read by pkg/quant)
enum {
    GGUF_METAL_MAT_F32 = 0,
    GGUF_METAL_MAT_Q8_0 = 1,
    GGUF_METAL_MAT_Q4_0 = 2,
    GGUF_METAL_MAT_Q4_K = 3,
};

// gguf_metal_open creates device + queue and compiles msl_src plus every B1/B2 pipeline; NULL on failure with message in err
gguf_metal_t gguf_metal_open(const char *msl_src, char *err, size_t err_len);

void gguf_metal_close(gguf_metal_t ctx);

const char *gguf_metal_device_name(gguf_metal_t ctx);

const char *gguf_metal_last_error(gguf_metal_t ctx);

uint64_t gguf_metal_allocated(gguf_metal_t ctx);

uint64_t gguf_metal_recommended(gguf_metal_t ctx);

uint64_t gguf_metal_max_buffer(gguf_metal_t ctx);

gguf_metal_buf_t gguf_metal_buf_new(gguf_metal_t ctx, size_t bytes);

gguf_metal_buf_t gguf_metal_buf_from(gguf_metal_t ctx, const void *src, size_t bytes);

void *gguf_metal_buf_ptr(gguf_metal_buf_t buf);

void gguf_metal_buf_free(gguf_metal_buf_t buf);

int gguf_metal_matmul_f32(gguf_metal_t ctx, gguf_metal_buf_t w, gguf_metal_buf_t x, gguf_metal_buf_t out, uint32_t rows, uint32_t cols);

int gguf_metal_matmul_q8_0(gguf_metal_t ctx, gguf_metal_buf_t w, gguf_metal_buf_t x, gguf_metal_buf_t out, uint32_t rows, uint32_t cols);

// gguf_metal_matmul_off: out[out_off bytes ..) = W * x[x_off bytes ..) for any kind above (rows x cols); offsets are byte offsets, multiples of 4
int gguf_metal_matmul_off(gguf_metal_t ctx, int kind, gguf_metal_buf_t w, gguf_metal_buf_t x, size_t x_off, gguf_metal_buf_t out, size_t out_off, uint32_t rows, uint32_t cols);

int gguf_metal_rmsnorm(gguf_metal_t ctx, gguf_metal_buf_t x, gguf_metal_buf_t weight, gguf_metal_buf_t dst, uint32_t dim, float eps);

// gguf_metal_rmsnorm_rows normalizes `rows` consecutive dim-vectors (one threadgroup each) with a shared weight[dim]; x and dst may alias (per-head QK-norm in place)
int gguf_metal_rmsnorm_rows(gguf_metal_t ctx, gguf_metal_buf_t x, size_t x_off, gguf_metal_buf_t weight, gguf_metal_buf_t dst, size_t dst_off, uint32_t dim, uint32_t rows, float eps);

int gguf_metal_rope_heads(gguf_metal_t ctx, gguf_metal_buf_t v, gguf_metal_buf_t cos_sin, uint32_t n_heads, uint32_t head_dim);

// gguf_metal_rope_heads_off rotates the heads at byte offset v_off of v (NeoX pairs, cos_sin = cos[head_dim/2] then sin[head_dim/2])
int gguf_metal_rope_heads_off(gguf_metal_t ctx, gguf_metal_buf_t v, size_t v_off, gguf_metal_buf_t cos_sin, uint32_t n_heads, uint32_t head_dim);

// gguf_metal_add: a[i] += b[i] for i < n (residual add)
int gguf_metal_add(gguf_metal_t ctx, gguf_metal_buf_t a, gguf_metal_buf_t b, uint32_t n);

int gguf_metal_swiglu(gguf_metal_t ctx, gguf_metal_buf_t gate, gguf_metal_buf_t up, uint32_t n);

int gguf_metal_attention(gguf_metal_t ctx, gguf_metal_buf_t q, gguf_metal_buf_t k, gguf_metal_buf_t v, gguf_metal_buf_t scores, gguf_metal_buf_t out, uint32_t seq_len, uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_dim, uint32_t scores_stride);

// Batch mode: between gguf_metal_batch_begin and gguf_metal_batch_end every dispatch above is only encoded (one command buffer, one serial compute encoder, buffers are hazard-tracked so dispatches run in order); gguf_metal_batch_end commits and waits (non-zero = the GPU work failed: buffers written by the batch may be partially updated).
// gguf_metal_batch_abort drops a batch that was never committed (nothing ran). Outside a batch each dispatch is committed and awaited on its own.
int gguf_metal_batch_begin(gguf_metal_t ctx);

int gguf_metal_batch_end(gguf_metal_t ctx);

void gguf_metal_batch_abort(gguf_metal_t ctx);

#endif
