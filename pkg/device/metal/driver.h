#ifndef GGUF_METAL_DRIVER_H
#define GGUF_METAL_DRIVER_H

#include <stddef.h>
#include <stdint.h>

// Plain-C surface over the Objective-C Metal driver (driver_darwin.m).
// Handles are retained Objective-C objects behind void*; every dispatch is synchronous (command buffer completed on return) and returns 0 on success, non-zero with gguf_metal_last_error() text otherwise.
// Buffers are MTLResourceStorageModeShared: gguf_metal_buf_ptr() is CPU-visible memory the GPU reads/writes directly (unified memory).

typedef void *gguf_metal_t;
typedef void *gguf_metal_buf_t;

// gguf_metal_open creates device + queue and compiles msl_src plus every B1 pipeline; NULL on failure with message in err
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

int gguf_metal_rmsnorm(gguf_metal_t ctx, gguf_metal_buf_t x, gguf_metal_buf_t weight, gguf_metal_buf_t dst, uint32_t dim, float eps);

int gguf_metal_rope_heads(gguf_metal_t ctx, gguf_metal_buf_t v, gguf_metal_buf_t cos_sin, uint32_t n_heads, uint32_t head_dim);

int gguf_metal_swiglu(gguf_metal_t ctx, gguf_metal_buf_t gate, gguf_metal_buf_t up, uint32_t n);

int gguf_metal_attention(gguf_metal_t ctx, gguf_metal_buf_t q, gguf_metal_buf_t k, gguf_metal_buf_t v, gguf_metal_buf_t scores, gguf_metal_buf_t out, uint32_t seq_len, uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_dim, uint32_t scores_stride);

#endif
