#ifndef GGUF_CUDA_DRIVER_H
#define GGUF_CUDA_DRIVER_H

#include <stddef.h>
#include <stdint.h>

#define CUDA_SUCCESS 0

typedef int CUresult;

typedef int CUdevice;

typedef struct CUctx_st *CUcontext;

typedef struct CUmod_st *CUmodule;

typedef struct CUfunc_st *CUfunction;

typedef struct CUstream_st *CUstream;

typedef struct CUgraph_st *CUgraph;

typedef struct CUgraphExec_st *CUgraphExec;

typedef unsigned long long CUdeviceptr;

typedef CUresult (*PFN_cuInit)(unsigned int flags);

typedef CUresult (*PFN_cuDeviceGetCount)(int *count);

typedef CUresult (*PFN_cuDeviceGet)(CUdevice *device, int ordinal);

typedef CUresult (*PFN_cuDeviceGetName)(char *name, int len, CUdevice dev);

typedef CUresult (*PFN_cuCtxCreate_v2)(CUcontext *pctx, unsigned int flags, CUdevice dev);

typedef CUresult (*PFN_cuCtxDestroy_v2)(CUcontext ctx);

typedef CUresult (*PFN_cuMemAlloc_v2)(CUdeviceptr *dptr, size_t bytesize);

typedef CUresult (*PFN_cuMemFree_v2)(CUdeviceptr dptr);

typedef CUresult (*PFN_cuMemGetInfo_v2)(size_t *free, size_t *total);

typedef CUresult (*PFN_cuMemcpyHtoD_v2)(CUdeviceptr dst, const void *src, size_t bytes);

typedef CUresult (*PFN_cuMemcpyDtoH_v2)(void *dst, CUdeviceptr src, size_t bytes);

typedef CUresult (*PFN_cuMemcpyDtoD_v2)(CUdeviceptr dst, CUdeviceptr src, size_t bytes);

typedef CUresult (*PFN_cuMemcpyHtoDAsync_v2)(CUdeviceptr dst, const void *src, size_t bytes, CUstream stream);

typedef CUresult (*PFN_cuMemcpyDtoHAsync_v2)(void *dst, CUdeviceptr src, size_t bytes, CUstream stream);

typedef CUresult (*PFN_cuGetErrorName)(CUresult error, const char **pStr);

typedef CUresult (*PFN_cuGetErrorString)(CUresult error, const char **pStr);

typedef CUresult (*PFN_cuCtxSetCurrent)(CUcontext ctx);

typedef CUresult (*PFN_cuDeviceGetAttribute)(int *pi, int attrib, CUdevice dev);

typedef CUresult (*PFN_cuModuleLoadData)(CUmodule *module, const void *image);

typedef CUresult (*PFN_cuModuleLoadDataEx)(CUmodule *module, const void *image, unsigned int numOptions, void *options, void **optionValues);

typedef CUresult (*PFN_cuModuleGetFunction)(CUfunction *hfunc, CUmodule hmod, const char *name);

typedef CUresult (*PFN_cuLaunchKernel)(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void **kernelParams, void **extra);

typedef CUresult (*PFN_cuStreamCreate)(CUstream *phStream, unsigned int flags);

typedef CUresult (*PFN_cuStreamDestroy_v2)(CUstream hStream);

typedef CUresult (*PFN_cuStreamSynchronize)(CUstream hStream);

typedef CUresult (*PFN_cuStreamBeginCapture_v2)(CUstream hStream, int mode);

typedef CUresult (*PFN_cuStreamEndCapture)(CUstream hStream, CUgraph *phGraph);

typedef CUresult (*PFN_cuGraphDestroy)(CUgraph graph);

typedef CUresult (*PFN_cuGraphInstantiateWithFlags)(CUgraphExec *phGraphExec, CUgraph graph, unsigned long long flags);

typedef CUresult (*PFN_cuGraphInstantiate_v2)(CUgraphExec *phGraphExec, CUgraph graph, void *phErrorNode, char *logBuffer, size_t bufferSize);

typedef CUresult (*PFN_cuGraphLaunch)(CUgraphExec hGraphExec, CUstream hStream);

typedef CUresult (*PFN_cuGraphExecDestroy)(CUgraphExec hGraphExec);

typedef struct {
	PFN_cuInit cuInit;
	PFN_cuDeviceGetCount cuDeviceGetCount;
	PFN_cuDeviceGet cuDeviceGet;
	PFN_cuDeviceGetName cuDeviceGetName;
	PFN_cuDeviceGetAttribute cuDeviceGetAttribute;
	PFN_cuCtxCreate_v2 cuCtxCreate;
	PFN_cuCtxDestroy_v2 cuCtxDestroy;
	PFN_cuMemAlloc_v2 cuMemAlloc;
	PFN_cuMemFree_v2 cuMemFree;
	PFN_cuMemGetInfo_v2 cuMemGetInfo;
	PFN_cuMemcpyHtoD_v2 cuMemcpyHtoD;
	PFN_cuMemcpyDtoH_v2 cuMemcpyDtoH;
	PFN_cuMemcpyDtoD_v2 cuMemcpyDtoD;
	PFN_cuMemcpyHtoDAsync_v2 cuMemcpyHtoDAsync;
	PFN_cuMemcpyDtoHAsync_v2 cuMemcpyDtoHAsync;
	PFN_cuGetErrorName cuGetErrorName;
	PFN_cuGetErrorString cuGetErrorString;
	PFN_cuCtxSetCurrent cuCtxSetCurrent;
	PFN_cuModuleLoadData cuModuleLoadData;
	PFN_cuModuleLoadDataEx cuModuleLoadDataEx;
	PFN_cuModuleGetFunction cuModuleGetFunction;
	PFN_cuLaunchKernel cuLaunchKernel;
	PFN_cuStreamCreate cuStreamCreate;
	PFN_cuStreamDestroy_v2 cuStreamDestroy;
	PFN_cuStreamSynchronize cuStreamSynchronize;
	PFN_cuStreamBeginCapture_v2 cuStreamBeginCapture;
	PFN_cuStreamEndCapture cuStreamEndCapture;
	PFN_cuGraphDestroy cuGraphDestroy;
	PFN_cuGraphInstantiateWithFlags cuGraphInstantiateWithFlags;
	PFN_cuGraphInstantiate_v2 cuGraphInstantiate;
	PFN_cuGraphLaunch cuGraphLaunch;
	PFN_cuGraphExecDestroy cuGraphExecDestroy;
	int has_graphs;
} cuda_driver_t;

#define GGUF_CUDA_MIN_CC 60

typedef struct gguf_matmul_graph_entry {
	CUdeviceptr d_matrix;
	int rows;
	int cols;
	int is_q8;
	int kernel_only; // 1 = no HtoD (vec already on GPU)
	CUgraphExec exec;
	struct gguf_matmul_graph_entry *next;
} gguf_matmul_graph_entry_t;

#define GGUF_LAYER_GRAPH_FFN 1
#define GGUF_LAYER_GRAPH_RESIDUAL 2
#define GGUF_LAYER_GRAPH_QKV 3
#define GGUF_LAYER_GRAPH_LOGITS 4

typedef struct gguf_layer_graph_entry {
	int kind;
	CUdeviceptr d_wo;
	CUdeviceptr d_ffn_norm;
	CUdeviceptr d_gate_w;
	CUdeviceptr d_up_w;
	CUdeviceptr d_down_w;
	CUdeviceptr d_wq;
	CUdeviceptr d_wk;
	CUdeviceptr d_wv;
	CUdeviceptr d_q_norm;
	CUdeviceptr d_k_norm;
	CUdeviceptr d_attn_norm; // QKV: RMSNorm of resident residual -> d_vec
	int rope_mode; // QKV: 0 = NeoX (rope_heads), 1 = Llama NORM (rope_heads_norm)
	int embd;
	int attn_dim;
	int ffn;
	int n_heads;
	int n_kv_heads;
	int head_dim;
	int skip_attn;  // residual: attn already in d_vec
	int skip_vec;   // ffn: x already in d_vec
	int resid_dev;  // 1 = x already in d_resid (HtoD not needed)
	int keep_resid; // 1 = residual stays on device (no DtoH)
	CUgraphExec exec;
	struct gguf_layer_graph_entry *next;
} gguf_layer_graph_entry_t;

// gguf_matmul_pool_t - reusable d_vec/d_out/d_aux/d_resid + host staging + CUDA Graph cache
typedef struct {
	CUdeviceptr d_vec;
	CUdeviceptr d_out;
	CUdeviceptr d_aux;   // FFN up
	CUdeviceptr d_resid; // residual stream (layer residency)
	float *h_vec;
	float *h_out;
	float *h_resid; // staging for residual HtoD(x)
	int vec_cap;
	int out_cap;
	int aux_cap;
	int resid_cap;
	CUstream stream;
	gguf_matmul_graph_entry_t *graphs;
	gguf_layer_graph_entry_t *layer_graphs;
	int skip_vec_htod; // 1 = vec already on GPU (set from Go)
	int skip_attn_htod; // 1 = attn already in d_vec (QKV residency)
	int resid_on_device; // 1 = d_resid holds current hidden (HtoD x not needed)
	int keep_resid_device; // 1 = do not copy residual back to host
	int resid_len; // length of valid residual in d_resid
} gguf_matmul_pool_t;

// gguf_cuda_init loads libcuda.so and creates context on GPU with ordinal device (numbering after CUDA_VISIBLE_DEVICES; 0 = first visible device)
// cc_out: compute capability (major*10+minor), e.g. 120 for sm_120
int gguf_cuda_init(cuda_driver_t *drv, void **lib_out, CUcontext *ctx, int device, char *name, size_t name_len, char *errbuf, size_t errbuf_len, int *cc_out);

// gguf_cuda_shutdown destroys context
void gguf_cuda_shutdown(cuda_driver_t *drv, CUcontext ctx);

// gguf_cuda_mem_info returns free/total context VRAM (cuMemGetInfo)
int gguf_cuda_mem_info(cuda_driver_t *drv, CUcontext ctx, size_t *free_out, size_t *total_out);

// gguf_cuda_last_error returns text of last CUDA error (if available)
const char *gguf_cuda_last_error(cuda_driver_t *drv, CUresult err);

// gguf_cuda_load_module loads PTX module; fn/fn_q8/fn_rmsnorm/fn_rope/fn_swiglu may be NULL
int gguf_cuda_load_module(cuda_driver_t *drv, CUcontext ctx, const char *ptx, CUmodule *module, CUfunction *fn, CUfunction *fn_q8, CUfunction *fn_rmsnorm, CUfunction *fn_rope, CUfunction *fn_swiglu, char *errbuf, size_t errbuf_len);

// gguf_cuda_upload_matrix uploads matrix to GPU
int gguf_cuda_upload_matrix(cuda_driver_t *drv, CUcontext ctx, CUdeviceptr *d_matrix, const float *matrix, int rows, int cols);

// gguf_cuda_matmul_pool_init creates stream and empty pool
int gguf_cuda_matmul_pool_init(cuda_driver_t *drv, CUcontext ctx, gguf_matmul_pool_t *pool);

// gguf_cuda_matmul_pool_free frees pool, graphs and stream
void gguf_cuda_matmul_pool_free(cuda_driver_t *drv, gguf_matmul_pool_t *pool);

// gguf_cuda_matmul_pool_clear_graphs clears graph cache (after free/replace weights)
void gguf_cuda_matmul_pool_clear_graphs(cuda_driver_t *drv, gguf_matmul_pool_t *pool);

// gguf_cuda_matmul_vec_device matmul with matrix already on GPU (pool required)
int gguf_cuda_matmul_vec_device(cuda_driver_t *drv, CUcontext ctx, CUfunction fn, gguf_matmul_pool_t *pool, CUdeviceptr d_matrix, const float *vec, float *out, int rows, int cols);

// gguf_cuda_free frees GPU buffer
void gguf_cuda_free(cuda_driver_t *drv, CUdeviceptr ptr);

// gguf_cuda_matmul_vec uploads matrix and runs kernel (no weight cache)
int gguf_cuda_matmul_vec(cuda_driver_t *drv, CUcontext ctx, CUfunction fn, gguf_matmul_pool_t *pool, const float *matrix, const float *vec, float *out, int rows, int cols);

// gguf_cuda_upload_q8_0 uploads Q8_0 matrix to GPU
int gguf_cuda_upload_q8_0(cuda_driver_t *drv, CUcontext ctx, CUdeviceptr *d_matrix, const void *raw, size_t nbytes);

// gguf_cuda_matmul_vec_q8_0_device matmul Q8_0 with weights already on GPU
int gguf_cuda_matmul_vec_q8_0_device(cuda_driver_t *drv, CUcontext ctx, CUfunction fn, gguf_matmul_pool_t *pool, CUdeviceptr d_matrix, const float *vec, float *out, int rows, int cols);

// gguf_cuda_upload_q4_0 uploads Q4_0 matrix to GPU (scale -> FP32)
int gguf_cuda_upload_q4_0(cuda_driver_t *drv, CUcontext ctx, CUdeviceptr *d_matrix, const void *raw, size_t nbytes);

// gguf_cuda_matmul_vec_q4_0_device matmul Q4_0 with weights already on GPU
int gguf_cuda_matmul_vec_q4_0_device(cuda_driver_t *drv, CUcontext ctx, CUfunction fn, gguf_matmul_pool_t *pool, CUdeviceptr d_matrix, const float *vec, float *out, int rows, int cols);

// gguf_cuda_upload_q4_k uploads Q4_K (scales->fp32 d_sc/d_mn + qs)
int gguf_cuda_upload_q4_k(cuda_driver_t *drv, CUcontext ctx, CUdeviceptr *d_matrix, const void *raw, size_t nbytes);

// gguf_cuda_matmul_vec_q4_k_device matmul Q4_K with weights already on GPU
int gguf_cuda_matmul_vec_q4_k_device(cuda_driver_t *drv, CUcontext ctx, CUfunction fn, gguf_matmul_pool_t *pool, CUdeviceptr d_matrix, const float *vec, float *out, int rows, int cols);

// gguf_cuda_upload_q5_k uploads Q5_K (scales->fp32 d_sc/d_mn + qh + qs)
int gguf_cuda_upload_q5_k(cuda_driver_t *drv, CUcontext ctx, CUdeviceptr *d_matrix, const void *raw, size_t nbytes);

// gguf_cuda_matmul_vec_q5_k_device matmul Q5_K with weights already on GPU
int gguf_cuda_matmul_vec_q5_k_device(cuda_driver_t *drv, CUcontext ctx, CUfunction fn, gguf_matmul_pool_t *pool, CUdeviceptr d_matrix, const float *vec, float *out, int rows, int cols);

// gguf_cuda_upload_q6_k uploads Q6_K (scales->fp32 d*sc + ql + qh)
int gguf_cuda_upload_q6_k(cuda_driver_t *drv, CUcontext ctx, CUdeviceptr *d_matrix, const void *raw, size_t nbytes);

// gguf_cuda_matmul_vec_q6_k_device matmul Q6_K with weights already on GPU
int gguf_cuda_matmul_vec_q6_k_device(cuda_driver_t *drv, CUcontext ctx, CUfunction fn, gguf_matmul_pool_t *pool, CUdeviceptr d_matrix, const float *vec, float *out, int rows, int cols);

// gguf_cuda_ffn_swiglu_device FFN: gate/up matmul + SwiGLU + down (CUDA Graph when has_graphs)
int gguf_cuda_ffn_swiglu_device(cuda_driver_t *drv, CUcontext ctx, CUfunction fn_matmul, CUfunction fn_swiglu, gguf_matmul_pool_t *pool, CUdeviceptr d_gate_w, CUdeviceptr d_up_w, CUdeviceptr d_down_w, const float *x, float *out, int embd, int ffn);

// gguf_cuda_attn_ffn_residual_device: WO + residual + RMSNorm + FFN + residual (CUDA Graph)
// 2*HtoD + 1*DtoH; if pool->skip_attn_htod - attn already in d_vec (1*HtoD x + 1*DtoH).
// pool->resid_on_device: x already in d_resid (HtoD not needed, x may be NULL);
// pool->keep_resid_device: result stays in d_resid (no DtoH, x_out may be NULL).
// resid_dirty (may be NULL): 1 = d_resid changed before error.
int gguf_cuda_attn_ffn_residual_device(
    cuda_driver_t *drv,
    CUcontext ctx,
    CUfunction fn_matmul,
    CUfunction fn_rmsnorm,
    CUfunction fn_swiglu,
    CUfunction fn_add,
    gguf_matmul_pool_t *pool,
    CUdeviceptr d_wo,
    CUdeviceptr d_ffn_norm,
    CUdeviceptr d_gate_w,
    CUdeviceptr d_up_w,
    CUdeviceptr d_down_w,
    const float *x,
    const float *attn,
    float *x_out,
    int embd,
    int attn_dim,
    int ffn,
    float eps,
    int *resid_dirty
);

// gguf_cuda_hidden_upload puts hidden state in d_resid (one HtoD per token)
int gguf_cuda_hidden_upload(cuda_driver_t *drv, CUcontext ctx, gguf_matmul_pool_t *pool, const float *x, int n);

// gguf_cuda_hidden_download copies resident hidden state from d_resid to host
int gguf_cuda_hidden_download(cuda_driver_t *drv, CUcontext ctx, gguf_matmul_pool_t *pool, float *dst, int n);

// gguf_cuda_logits_from_device: RMSNorm(d_resid) + lm_head matmul on GPU, DtoH only logits
int gguf_cuda_logits_from_device(
    cuda_driver_t *drv,
    CUcontext ctx,
    CUfunction fn_matmul,
    CUfunction fn_rmsnorm,
    gguf_matmul_pool_t *pool,
    CUdeviceptr d_out_norm,
    CUdeviceptr d_head,
    float *logits,
    int vocab,
    int embd,
    float eps
);

// gguf_cuda_rmsnorm RMSNorm on GPU
int gguf_cuda_rmsnorm(cuda_driver_t *drv, CUcontext ctx, CUfunction fn, const float *x, const float *weight, float *out, int n, float eps);

// gguf_cuda_rope_heads RoPE for nHeads heads (cos/sin on CPU, rotate on GPU)
int gguf_cuda_rope_heads(cuda_driver_t *drv, CUcontext ctx, CUfunction fn, float *v, const float *cos_tbl, const float *sin_tbl, int nheads, int head_dim, int half);

// gguf_cuda_swiglu silu(gate)*up in-place (result in gate)
int gguf_cuda_swiglu(cuda_driver_t *drv, CUcontext ctx, CUfunction fn, float *gate, const float *up, int n);

// gguf_cuda_module_function gets function from loaded module
int gguf_cuda_module_function(cuda_driver_t *drv, CUmodule module, const char *name, CUfunction *fn_out);

// gguf_cuda_attention scaled dot-product attention (fn_softmax on GPU; NULL = host softmax)
int gguf_cuda_attention(cuda_driver_t *drv, CUcontext ctx, CUfunction fn_qk, CUfunction fn_v, CUfunction fn_softmax, float *dst, const float *q, const float *k, const float *v, int seq_len, int n_heads, int n_kv_heads, int head_dim);

typedef struct {
	CUdeviceptr d_k;
	CUdeviceptr d_v;
	int max_seq;
	int kv_dim;
} gguf_kv_layer_t;

typedef struct {
	gguf_kv_layer_t *layers;
	int num_layers;
} gguf_kv_cache_t;

typedef struct gguf_attn_graph_entry {
	CUdeviceptr d_k;
	CUdeviceptr d_v;
	int seq_len;
	int n_heads;
	int n_kv_heads;
	int head_dim;
	int skip_q_htod;
	int skip_dst_dtoh;
	CUgraphExec exec;
	struct gguf_attn_graph_entry *next;
} gguf_attn_graph_entry_t;

typedef struct {
	CUdeviceptr d_q;
	CUdeviceptr d_dst;
	CUdeviceptr d_scores;
	CUdeviceptr d_k_tok; // current token K (kv_dim)
	CUdeviceptr d_v_tok; // current token V
	CUdeviceptr d_cos; // RoPE cos (head_dim/2)
	CUdeviceptr d_sin; // RoPE sin
	float *h_cos;
	float *h_sin;
	int q_elems;
	int max_seq;
	int kv_dim;
	int rope_half;
	gguf_attn_graph_entry_t *graphs;
	int graph_count;
} gguf_attn_pool_t;

// gguf_cuda_kv_init allocates GPU K/V buffers for num_layers layers
int gguf_cuda_kv_init(cuda_driver_t *drv, CUcontext ctx, gguf_kv_cache_t *cache, int num_layers, int max_seq, int kv_dim);

// gguf_cuda_kv_free frees GPU KV-cache
void gguf_cuda_kv_free(cuda_driver_t *drv, gguf_kv_cache_t *cache);

// gguf_cuda_kv_append copies K/V of one token to position pos
int gguf_cuda_kv_append(cuda_driver_t *drv, CUcontext ctx, gguf_kv_cache_t *cache, int layer, int pos, const float *k, const float *v);

// gguf_cuda_kv_append_n copies K/V of n tokens starting at pos (batch prefill)
int gguf_cuda_kv_append_n(cuda_driver_t *drv, CUcontext ctx, gguf_kv_cache_t *cache, int layer, int pos, const float *k, const float *v, int n);

// gguf_cuda_kv_attention attention with K/V already on GPU
int gguf_cuda_kv_attention(cuda_driver_t *drv, CUcontext ctx, CUfunction fn_qk, CUfunction fn_v, CUfunction fn_softmax, gguf_kv_cache_t *cache, gguf_attn_pool_t *pool, int layer, float *dst, const float *q, int seq_len, int n_heads, int n_kv_heads, int head_dim);

// gguf_cuda_attn_pool_init allocates reusable attention buffers (+ K/V token, RoPE)
int gguf_cuda_attn_pool_init(cuda_driver_t *drv, CUcontext ctx, gguf_attn_pool_t *pool, int q_elems, int max_seq, int kv_dim, int rope_half);

// gguf_cuda_attn_pool_free frees attention buffers
void gguf_cuda_attn_pool_free(cuda_driver_t *drv, gguf_attn_pool_t *pool);

// gguf_cuda_qkv_rope_attn_device: h->QKV->head RMSNorm->RoPE (CUDA Graph) -> KV append->attn
// After successful attn lies in matmul_pool.d_vec and skip_attn_htod=1 for residual.
// If pool->resid_on_device and d_attn_norm != 0, h computed on GPU as RMSNorm(d_resid)
// and host buffer h is not read (may be NULL).
// d_q_norm / d_k_norm == 0 - layer without QK-norm (Llama / Mistral).
// rope_mode distinguishes graph cache for fn_rope: 0 = NeoX, 1 = Llama NORM.
int gguf_cuda_qkv_rope_attn_device(
    cuda_driver_t *drv,
    CUcontext ctx,
    CUfunction fn_matmul,
    CUfunction fn_rmsnorm,
    CUfunction fn_rope,
    CUfunction fn_qk,
    CUfunction fn_v,
    CUfunction fn_softmax,
    gguf_matmul_pool_t *mpool,
    gguf_attn_pool_t *apool,
    gguf_kv_cache_t *kv,
    CUdeviceptr d_wq,
    CUdeviceptr d_wk,
    CUdeviceptr d_wv,
    CUdeviceptr d_q_norm,
    CUdeviceptr d_k_norm,
    CUdeviceptr d_attn_norm,
    const float *h,
    const float *cos_tbl,
    const float *sin_tbl,
    float *attn_out,
    float *k_out,
    float *v_out,
    int embd,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int layer,
    int kv_pos,
    int seq_len,
    int rope_mode,
    float eps
);

#endif
