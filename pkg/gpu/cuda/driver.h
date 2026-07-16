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

typedef CUresult (*PFN_cuMemcpyHtoD_v2)(CUdeviceptr dst, const void *src, size_t bytes);

typedef CUresult (*PFN_cuMemcpyDtoH_v2)(void *dst, CUdeviceptr src, size_t bytes);

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
	PFN_cuMemcpyHtoD_v2 cuMemcpyHtoD;
	PFN_cuMemcpyDtoH_v2 cuMemcpyDtoH;
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
	CUgraphExec exec;
	struct gguf_matmul_graph_entry *next;
} gguf_matmul_graph_entry_t;

// gguf_matmul_pool_t - переиспользуемые d_vec/d_out + host staging + CUDA Graph cache
typedef struct {
	CUdeviceptr d_vec;
	CUdeviceptr d_out;
	float *h_vec;
	float *h_out;
	int vec_cap;
	int out_cap;
	CUstream stream;
	gguf_matmul_graph_entry_t *graphs;
} gguf_matmul_pool_t;

// gguf_cuda_init загружает libcuda.so и создаёт контекст на GPU 0
// cc_out: compute capability (major*10+minor), например 120 для sm_120
int gguf_cuda_init(cuda_driver_t *drv, void **lib_out, CUcontext *ctx, char *name, size_t name_len, char *errbuf, size_t errbuf_len, int *cc_out);

// gguf_cuda_shutdown уничтожает контекст
void gguf_cuda_shutdown(cuda_driver_t *drv, CUcontext ctx);

// gguf_cuda_last_error возвращает текст последней CUDA-ошибки (если доступен)
const char *gguf_cuda_last_error(cuda_driver_t *drv, CUresult err);

// gguf_cuda_load_module загружает PTX-модуль; fn/fn_q8/fn_rmsnorm/fn_rope/fn_swiglu могут быть NULL
int gguf_cuda_load_module(cuda_driver_t *drv, CUcontext ctx, const char *ptx, CUmodule *module, CUfunction *fn, CUfunction *fn_q8, CUfunction *fn_rmsnorm, CUfunction *fn_rope, CUfunction *fn_swiglu, char *errbuf, size_t errbuf_len);

// gguf_cuda_upload_matrix загружает matrix на GPU
int gguf_cuda_upload_matrix(cuda_driver_t *drv, CUcontext ctx, CUdeviceptr *d_matrix, const float *matrix, int rows, int cols);

// gguf_cuda_matmul_pool_init создаёт stream и пустой pool
int gguf_cuda_matmul_pool_init(cuda_driver_t *drv, CUcontext ctx, gguf_matmul_pool_t *pool);

// gguf_cuda_matmul_pool_free освобождает pool, graphs и stream
void gguf_cuda_matmul_pool_free(cuda_driver_t *drv, gguf_matmul_pool_t *pool);

// gguf_cuda_matmul_pool_clear_graphs сбрасывает graph cache (после free/replace весов)
void gguf_cuda_matmul_pool_clear_graphs(cuda_driver_t *drv, gguf_matmul_pool_t *pool);

// gguf_cuda_matmul_vec_device matmul с matrix уже на GPU (pool обязателен)
int gguf_cuda_matmul_vec_device(cuda_driver_t *drv, CUcontext ctx, CUfunction fn, gguf_matmul_pool_t *pool, CUdeviceptr d_matrix, const float *vec, float *out, int rows, int cols);

// gguf_cuda_free освобождает GPU-буфер
void gguf_cuda_free(cuda_driver_t *drv, CUdeviceptr ptr);

// gguf_cuda_matmul_vec загружает matrix и запускает kernel (без кеша весов)
int gguf_cuda_matmul_vec(cuda_driver_t *drv, CUcontext ctx, CUfunction fn, gguf_matmul_pool_t *pool, const float *matrix, const float *vec, float *out, int rows, int cols);

// gguf_cuda_upload_q8_0 загружает Q8_0-матрицу на GPU
int gguf_cuda_upload_q8_0(cuda_driver_t *drv, CUcontext ctx, CUdeviceptr *d_matrix, const void *raw, size_t nbytes);

// gguf_cuda_matmul_vec_q8_0_device matmul Q8_0 с весами уже на GPU
int gguf_cuda_matmul_vec_q8_0_device(cuda_driver_t *drv, CUcontext ctx, CUfunction fn, gguf_matmul_pool_t *pool, CUdeviceptr d_matrix, const float *vec, float *out, int rows, int cols);

// gguf_cuda_rmsnorm RMSNorm на GPU
int gguf_cuda_rmsnorm(cuda_driver_t *drv, CUcontext ctx, CUfunction fn, const float *x, const float *weight, float *out, int n, float eps);

// gguf_cuda_rope_heads RoPE для nHeads голов (cos/sin на CPU, rotate на GPU)
int gguf_cuda_rope_heads(cuda_driver_t *drv, CUcontext ctx, CUfunction fn, float *v, const float *cos_tbl, const float *sin_tbl, int nheads, int head_dim, int half);

// gguf_cuda_swiglu silu(gate)*up in-place (результат в gate)
int gguf_cuda_swiglu(cuda_driver_t *drv, CUcontext ctx, CUfunction fn, float *gate, const float *up, int n);

// gguf_cuda_module_function получает функцию из уже загруженного модуля
int gguf_cuda_module_function(cuda_driver_t *drv, CUmodule module, const char *name, CUfunction *fn_out);

// gguf_cuda_attention scaled dot-product attention (softmax на CPU для точности)
int gguf_cuda_attention(cuda_driver_t *drv, CUcontext ctx, CUfunction fn_qk, CUfunction fn_v, float *dst, const float *q, const float *k, const float *v, int seq_len, int n_heads, int n_kv_heads, int head_dim);

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

typedef struct {
	CUdeviceptr d_q;
	CUdeviceptr d_dst;
	CUdeviceptr d_scores;
	int q_elems;
	int max_seq;
} gguf_attn_pool_t;

// gguf_cuda_kv_init выделяет GPU-буферы K/V для num_layers слоёв
int gguf_cuda_kv_init(cuda_driver_t *drv, CUcontext ctx, gguf_kv_cache_t *cache, int num_layers, int max_seq, int kv_dim);

// gguf_cuda_kv_free освобождает GPU KV-cache
void gguf_cuda_kv_free(cuda_driver_t *drv, gguf_kv_cache_t *cache);

// gguf_cuda_kv_append копирует K/V одного токена в позицию pos
int gguf_cuda_kv_append(cuda_driver_t *drv, CUcontext ctx, gguf_kv_cache_t *cache, int layer, int pos, const float *k, const float *v);

// gguf_cuda_kv_attention attention с K/V уже на GPU
int gguf_cuda_kv_attention(cuda_driver_t *drv, CUcontext ctx, CUfunction fn_qk, CUfunction fn_v, gguf_kv_cache_t *cache, gguf_attn_pool_t *pool, int layer, float *dst, const float *q, int seq_len, int n_heads, int n_kv_heads, int head_dim);

// gguf_cuda_attn_pool_init выделяет переиспользуемые буферы attention
int gguf_cuda_attn_pool_init(cuda_driver_t *drv, CUcontext ctx, gguf_attn_pool_t *pool, int q_elems, int max_seq);

// gguf_cuda_attn_pool_free освобождает буферы attention
void gguf_cuda_attn_pool_free(cuda_driver_t *drv, gguf_attn_pool_t *pool);

#endif
