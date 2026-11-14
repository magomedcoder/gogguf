//go:build metal && darwin && cgo

// Objective-C Metal driver: device/queue/library, shared-storage buffers and synchronous kernel dispatch.
// The MSL source arrives from Go (msl.go) and is compiled in gguf_metal_open; see driver.h for the C surface.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "driver.h"

#pragma clang diagnostic ignored "-Wdeprecated-declarations"

// Param structs mirror the MSL ones in msl.go (4-byte scalars only)
typedef struct {
    uint32_t rows;
    uint32_t cols;
} MatParams;

typedef struct {
    uint32_t dim;
    float eps;
} RMSParams;

typedef struct {
    uint32_t nHeads;
    uint32_t headDim;
} RopeParams;

typedef struct {
    uint32_t n;
} SwigluParams;

typedef struct {
    uint32_t n;
} AddParams;

typedef struct {
    uint32_t seqLen;
    uint32_t nHeads;
    uint32_t nKVHeads;
    uint32_t headDim;
    uint32_t scoresStride;
    float scale;
} AttnParams;

enum {
    kRowsPerTG = 4,
    kRMSThreads = 256,
    kAttnThreads = 128,
    kElemThreads = 256,
};

@interface GgufMetal : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) id<MTLCommandQueue> queue;
@property(nonatomic, strong) id<MTLLibrary> library;
@property(nonatomic, strong) id<MTLComputePipelineState> matF32;
@property(nonatomic, strong) id<MTLComputePipelineState> matQ8;
@property(nonatomic, strong) id<MTLComputePipelineState> matQ4_0;
@property(nonatomic, strong) id<MTLComputePipelineState> matQ4K;
@property(nonatomic, strong) id<MTLComputePipelineState> addPSO;
@property(nonatomic, strong) id<MTLComputePipelineState> rms;
@property(nonatomic, strong) id<MTLComputePipelineState> rope;
@property(nonatomic, strong) id<MTLComputePipelineState> swiglu;
@property(nonatomic, strong) id<MTLComputePipelineState> attn;
// open batch (gguf_metal_batch_begin): dispatches are encoded into batchEnc and committed together by gguf_metal_batch_end
@property(nonatomic, strong) id<MTLCommandBuffer> batchCB;
@property(nonatomic, strong) id<MTLComputeCommandEncoder> batchEnc;
@property(nonatomic, copy) NSString *lastError;
@property(nonatomic, copy) NSString *deviceName;
@end

@implementation GgufMetal
@end

static void set_err(char *dst, size_t n, NSString *msg) {
    if (dst == NULL || n == 0) {
        return;
    }
    snprintf(dst, n, "%s", msg != nil ? [msg UTF8String] : "unknown error");
}

static id<MTLComputePipelineState> make_pso(GgufMetal *m, const char *name) {
    id<MTLFunction> fn = [m.library newFunctionWithName:[NSString stringWithUTF8String:name]];
    if (fn == nil) {
        m.lastError = [NSString stringWithFormat:@"metal: kernel %s not found in library", name];
        return nil;
    }

    NSError *e = nil;
    id<MTLComputePipelineState> pso = [m.device newComputePipelineStateWithFunction:fn error:&e];
    if (pso == nil) {
        m.lastError = [NSString stringWithFormat:@"metal: pipeline %s: %@", name, e.localizedDescription];
    }

    return pso;
}

gguf_metal_t gguf_metal_open(const char *msl_src, char *err, size_t err_len) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (dev == nil) {
            set_err(err, err_len, @"metal: no Metal device (MTLCreateSystemDefaultDevice returned nil)");
            return NULL;
        }

        GgufMetal *m = [[GgufMetal alloc] init];
        m.device = dev;
        m.deviceName = [NSString stringWithFormat:@"Metal %@", dev.name];
        m.queue = [dev newCommandQueue];
        if (m.queue == nil) {
            set_err(err, err_len, @"metal: newCommandQueue failed");
            return NULL;
        }

        MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
        // precise exp/sqrt: keeps parity with the CPU reference (fast-math would trade ulps for speed)
        opts.fastMathEnabled = NO;

        NSError *e = nil;
        m.library = [dev newLibraryWithSource:[NSString stringWithUTF8String:msl_src] options:opts error:&e];
        if (m.library == nil) {
            set_err(err, err_len, [NSString stringWithFormat:@"metal: compile MSL: %@", e.localizedDescription]);
            return NULL;
        }

        m.matF32 = make_pso(m, "matmul_vec_f32");
        m.matQ8 = make_pso(m, "matmul_vec_q8_0");
        m.matQ4_0 = make_pso(m, "matmul_vec_q4_0");
        m.matQ4K = make_pso(m, "matmul_vec_q4_k");
        m.addPSO = make_pso(m, "add");
        m.rms = make_pso(m, "rmsnorm");
        m.rope = make_pso(m, "rope_heads");
        m.swiglu = make_pso(m, "swiglu");
        m.attn = make_pso(m, "attn_qk_softmax_v");
        if (m.matF32 == nil || m.matQ8 == nil || m.matQ4_0 == nil || m.matQ4K == nil || m.addPSO == nil || m.rms == nil || m.rope == nil || m.swiglu == nil || m.attn == nil) {
            set_err(err, err_len, m.lastError);
            return NULL;
        }

        return (__bridge_retained void *)m;
    }
}

void gguf_metal_close(gguf_metal_t ctx) {
    if (ctx == NULL) {
        return;
    }
    gguf_metal_batch_abort(ctx);

    // transfer ownership back to ARC and drop it
    GgufMetal *m = (__bridge_transfer GgufMetal *)ctx;
    m = nil;
    (void)m;
}

const char *gguf_metal_device_name(gguf_metal_t ctx) {
    GgufMetal *m = (__bridge GgufMetal *)ctx;
    return [m.deviceName UTF8String];
}

const char *gguf_metal_last_error(gguf_metal_t ctx) {
    GgufMetal *m = (__bridge GgufMetal *)ctx;
    return m.lastError != nil ? [m.lastError UTF8String] : "";
}

uint64_t gguf_metal_allocated(gguf_metal_t ctx) {
    GgufMetal *m = (__bridge GgufMetal *)ctx;
    return (uint64_t)m.device.currentAllocatedSize;
}

uint64_t gguf_metal_recommended(gguf_metal_t ctx) {
    GgufMetal *m = (__bridge GgufMetal *)ctx;
    return (uint64_t)m.device.recommendedMaxWorkingSetSize;
}

uint64_t gguf_metal_max_buffer(gguf_metal_t ctx) {
    GgufMetal *m = (__bridge GgufMetal *)ctx;
    return (uint64_t)m.device.maxBufferLength;
}

gguf_metal_buf_t gguf_metal_buf_new(gguf_metal_t ctx, size_t bytes) {
    GgufMetal *m = (__bridge GgufMetal *)ctx;
    if (bytes == 0) {
        bytes = 16;
    }
    if (bytes > m.device.maxBufferLength) {
        m.lastError = [NSString stringWithFormat:@"metal: buffer of %zu bytes exceeds maxBufferLength %lu", bytes, (unsigned long)m.device.maxBufferLength];
        return NULL;
    }

    id<MTLBuffer> buf = [m.device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (buf == nil) {
        m.lastError = [NSString stringWithFormat:@"metal: newBufferWithLength(%zu) failed (out of memory)", bytes];
        return NULL;
    }

    return (__bridge_retained void *)buf;
}

gguf_metal_buf_t gguf_metal_buf_from(gguf_metal_t ctx, const void *src, size_t bytes) {
    GgufMetal *m = (__bridge GgufMetal *)ctx;
    if (bytes > m.device.maxBufferLength) {
        m.lastError = [NSString stringWithFormat:@"metal: buffer of %zu bytes exceeds maxBufferLength %lu", bytes, (unsigned long)m.device.maxBufferLength];
        return NULL;
    }

    id<MTLBuffer> buf = [m.device newBufferWithBytes:src length:bytes options:MTLResourceStorageModeShared];
    if (buf == nil) {
        m.lastError = [NSString stringWithFormat:@"metal: newBufferWithBytes(%zu) failed (out of memory)", bytes];
        return NULL;
    }

    return (__bridge_retained void *)buf;
}

void *gguf_metal_buf_ptr(gguf_metal_buf_t buf) {
    id<MTLBuffer> b = (__bridge id<MTLBuffer>)buf;
    return [b contents];
}

void gguf_metal_buf_free(gguf_metal_buf_t buf) {
    if (buf == NULL) {
        return;
    }
    id<MTLBuffer> b = (__bridge_transfer id<MTLBuffer>)buf;
    b = nil;
    (void)b;
}

// encode_dispatch binds buffers at index 0..n-1 (optional byte offsets), params via setBytes at index n, and dispatches
static void encode_dispatch(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pso, void *const *bufs, const size_t *offs, int nbufs, const void *params, size_t plen, MTLSize groups, MTLSize threads) {
    [enc setComputePipelineState:pso];
    for (int i = 0; i < nbufs; i++) {
        [enc setBuffer:(__bridge id<MTLBuffer>)bufs[i] offset:(offs != NULL ? (NSUInteger)offs[i] : 0) atIndex:(NSUInteger)i];
    }
    [enc setBytes:params length:plen atIndex:(NSUInteger)nbufs];
    [enc dispatchThreadgroups:groups threadsPerThreadgroup:threads];
}

// run_off encodes one dispatch (buffers at index 0..n-1 with optional byte offsets, params via setBytes at index n).
// Inside a batch it only encodes; otherwise it commits and waits.
static int run_off(GgufMetal *m, id<MTLComputePipelineState> pso, void *const *bufs, const size_t *offs, int nbufs, const void *params, size_t plen, MTLSize groups, MTLSize threads) {
    @autoreleasepool {
        if (m.batchEnc != nil) {
            encode_dispatch(m.batchEnc, pso, bufs, offs, nbufs, params, plen, groups, threads);
            return 0;
        }

        id<MTLCommandBuffer> cb = [m.queue commandBuffer];
        if (cb == nil) {
            m.lastError = @"metal: commandBuffer failed";
            return 1;
        }

        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (enc == nil) {
            m.lastError = @"metal: computeCommandEncoder failed";
            return 1;
        }

        encode_dispatch(enc, pso, bufs, offs, nbufs, params, plen, groups, threads);
        [enc endEncoding];

        [cb commit];
        [cb waitUntilCompleted];

        if (cb.status != MTLCommandBufferStatusCompleted) {
            m.lastError = [NSString stringWithFormat:@"metal: command buffer failed: %@", cb.error.localizedDescription];
            return 2;
        }

        return 0;
    }
}

static int run(GgufMetal *m, id<MTLComputePipelineState> pso, void *const *bufs, int nbufs, const void *params, size_t plen, MTLSize groups, MTLSize threads) {
    return run_off(m, pso, bufs, NULL, nbufs, params, plen, groups, threads);
}

int gguf_metal_batch_begin(gguf_metal_t ctx) {
    GgufMetal *m = (__bridge GgufMetal *)ctx;
    if (m.batchCB != nil) {
        m.lastError = @"metal: batch already open";
        return 1;
    }

    id<MTLCommandBuffer> cb = [m.queue commandBuffer];
    if (cb == nil) {
        m.lastError = @"metal: commandBuffer failed";
        return 1;
    }

    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    if (enc == nil) {
        m.lastError = @"metal: computeCommandEncoder failed";
        return 1;
    }

    m.batchCB = cb;
    m.batchEnc = enc;

    return 0;
}

int gguf_metal_batch_end(gguf_metal_t ctx) {
    GgufMetal *m = (__bridge GgufMetal *)ctx;
    if (m.batchCB == nil) {
        m.lastError = @"metal: no open batch";
        return 1;
    }

    id<MTLCommandBuffer> cb = m.batchCB;
    id<MTLComputeCommandEncoder> enc = m.batchEnc;
    m.batchCB = nil;
    m.batchEnc = nil;

    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];

    if (cb.status != MTLCommandBufferStatusCompleted) {
        m.lastError = [NSString stringWithFormat:@"metal: command buffer failed: %@", cb.error.localizedDescription];
        return 2;
    }

    return 0;
}

void gguf_metal_batch_abort(gguf_metal_t ctx) {
    GgufMetal *m = (__bridge GgufMetal *)ctx;
    if (m.batchEnc != nil) {
        [m.batchEnc endEncoding];
    }

    m.batchCB = nil;
    m.batchEnc = nil;
}

static NSUInteger min_uint(NSUInteger a, NSUInteger b) {
    return a < b ? a : b;
}

static int matmul(gguf_metal_t ctx, id<MTLComputePipelineState> pso, gguf_metal_buf_t w, gguf_metal_buf_t x, size_t x_off, gguf_metal_buf_t out, size_t out_off, uint32_t rows, uint32_t cols) {
    GgufMetal *m = (__bridge GgufMetal *)ctx;
    MatParams p = {rows, cols};
    void *bufs[3] = {w, x, out};
    size_t offs[3] = {0, x_off, out_off};

    NSUInteger tg = min_uint(pso.threadExecutionWidth * kRowsPerTG, pso.maxTotalThreadsPerThreadgroup);
    if (tg < pso.threadExecutionWidth * kRowsPerTG) {
        m.lastError = @"metal: matmul pipeline cannot run 4 simdgroups per threadgroup";
        return 3;
    }

    NSUInteger groups = (rows + kRowsPerTG - 1) / kRowsPerTG;
    return run_off(m, pso, bufs, offs, 3, &p, sizeof(p), MTLSizeMake(groups, 1, 1), MTLSizeMake(tg, 1, 1));
}

int gguf_metal_matmul_off(gguf_metal_t ctx, int kind, gguf_metal_buf_t w, gguf_metal_buf_t x, size_t x_off, gguf_metal_buf_t out, size_t out_off, uint32_t rows, uint32_t cols) {
    GgufMetal *m = (__bridge GgufMetal *)ctx;
    id<MTLComputePipelineState> pso = nil;
    switch (kind) {
    case GGUF_METAL_MAT_F32:
        pso = m.matF32;
        break;
    case GGUF_METAL_MAT_Q8_0:
        pso = m.matQ8;
        break;
    case GGUF_METAL_MAT_Q4_0:
        pso = m.matQ4_0;
        break;
    case GGUF_METAL_MAT_Q4_K:
        pso = m.matQ4K;
        break;
    default:
        m.lastError = [NSString stringWithFormat:@"metal: unknown matvec kind %d", kind];
        return 3;
    }

    return matmul(ctx, pso, w, x, x_off, out, out_off, rows, cols);
}

int gguf_metal_matmul_f32(gguf_metal_t ctx, gguf_metal_buf_t w, gguf_metal_buf_t x, gguf_metal_buf_t out, uint32_t rows, uint32_t cols) {
    return gguf_metal_matmul_off(ctx, GGUF_METAL_MAT_F32, w, x, 0, out, 0, rows, cols);
}

int gguf_metal_matmul_q8_0(gguf_metal_t ctx, gguf_metal_buf_t w, gguf_metal_buf_t x, gguf_metal_buf_t out, uint32_t rows, uint32_t cols) {
    return gguf_metal_matmul_off(ctx, GGUF_METAL_MAT_Q8_0, w, x, 0, out, 0, rows, cols);
}

int gguf_metal_rmsnorm_rows(gguf_metal_t ctx, gguf_metal_buf_t x, size_t x_off, gguf_metal_buf_t weight, gguf_metal_buf_t dst, size_t dst_off, uint32_t dim, uint32_t rows, float eps) {
    GgufMetal *m = (__bridge GgufMetal *)ctx;
    RMSParams p = {dim, eps};
    void *bufs[3] = {x, weight, dst};
    size_t offs[3] = {x_off, 0, dst_off};
    NSUInteger tg = min_uint(kRMSThreads, m.rms.maxTotalThreadsPerThreadgroup);
    return run_off(m, m.rms, bufs, offs, 3, &p, sizeof(p), MTLSizeMake(rows, 1, 1), MTLSizeMake(tg, 1, 1));
}

int gguf_metal_rmsnorm(gguf_metal_t ctx, gguf_metal_buf_t x, gguf_metal_buf_t weight, gguf_metal_buf_t dst, uint32_t dim, float eps) {
    return gguf_metal_rmsnorm_rows(ctx, x, 0, weight, dst, 0, dim, 1, eps);
}

int gguf_metal_rope_heads_off(gguf_metal_t ctx, gguf_metal_buf_t v, size_t v_off, gguf_metal_buf_t cos_sin, uint32_t n_heads, uint32_t head_dim) {
    GgufMetal *m = (__bridge GgufMetal *)ctx;
    RopeParams p = {n_heads, head_dim};
    void *bufs[2] = {v, cos_sin};
    size_t offs[2] = {v_off, 0};
    NSUInteger total = (NSUInteger)n_heads * (head_dim / 2);
    NSUInteger tg = min_uint(kElemThreads, m.rope.maxTotalThreadsPerThreadgroup);
    NSUInteger groups = (total + tg - 1) / tg;
    return run_off(m, m.rope, bufs, offs, 2, &p, sizeof(p), MTLSizeMake(groups, 1, 1), MTLSizeMake(tg, 1, 1));
}

int gguf_metal_rope_heads(gguf_metal_t ctx, gguf_metal_buf_t v, gguf_metal_buf_t cos_sin, uint32_t n_heads, uint32_t head_dim) {
    return gguf_metal_rope_heads_off(ctx, v, 0, cos_sin, n_heads, head_dim);
}

int gguf_metal_add(gguf_metal_t ctx, gguf_metal_buf_t a, gguf_metal_buf_t b, uint32_t n) {
    GgufMetal *m = (__bridge GgufMetal *)ctx;
    AddParams p = {n};
    void *bufs[2] = {a, b};
    NSUInteger tg = min_uint(kElemThreads, m.addPSO.maxTotalThreadsPerThreadgroup);
    NSUInteger groups = ((NSUInteger)n + tg - 1) / tg;
    return run(m, m.addPSO, bufs, 2, &p, sizeof(p), MTLSizeMake(groups, 1, 1), MTLSizeMake(tg, 1, 1));
}

int gguf_metal_swiglu(gguf_metal_t ctx, gguf_metal_buf_t gate, gguf_metal_buf_t up, uint32_t n) {
    GgufMetal *m = (__bridge GgufMetal *)ctx;
    SwigluParams p = {n};
    void *bufs[2] = {gate, up};
    NSUInteger tg = min_uint(kElemThreads, m.swiglu.maxTotalThreadsPerThreadgroup);
    NSUInteger groups = ((NSUInteger)n + tg - 1) / tg;
    return run(m, m.swiglu, bufs, 2, &p, sizeof(p), MTLSizeMake(groups, 1, 1), MTLSizeMake(tg, 1, 1));
}

int gguf_metal_attention(gguf_metal_t ctx, gguf_metal_buf_t q, gguf_metal_buf_t k, gguf_metal_buf_t v, gguf_metal_buf_t scores, gguf_metal_buf_t out, uint32_t seq_len, uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_dim, uint32_t scores_stride) {
    GgufMetal *m = (__bridge GgufMetal *)ctx;
    AttnParams p = {seq_len, n_heads, n_kv_heads, head_dim, scores_stride, (float)(1.0 / sqrt((double)head_dim))};
    void *bufs[5] = {q, k, v, scores, out};
    NSUInteger tg = min_uint(kAttnThreads, m.attn.maxTotalThreadsPerThreadgroup);
    return run(m, m.attn, bufs, 5, &p, sizeof(p), MTLSizeMake(n_heads, 1, 1), MTLSizeMake(tg, 1, 1));
}
