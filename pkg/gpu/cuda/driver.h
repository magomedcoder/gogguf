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

typedef CUresult (*PFN_cuGetErrorName)(CUresult error, const char **pStr);

typedef CUresult (*PFN_cuGetErrorString)(CUresult error, const char **pStr);

typedef CUresult (*PFN_cuCtxSetCurrent)(CUcontext ctx);

typedef CUresult (*PFN_cuDeviceGetAttribute)(int *pi, int attrib, CUdevice dev);

typedef CUresult (*PFN_cuModuleLoadData)(CUmodule *module, const void *image);

typedef CUresult (*PFN_cuModuleLoadDataEx)(CUmodule *module, const void *image, unsigned int numOptions, void *options, void **optionValues);

typedef CUresult (*PFN_cuModuleGetFunction)(CUfunction *hfunc, CUmodule hmod, const char *name);

typedef CUresult (*PFN_cuLaunchKernel)(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, void *hStream, void **kernelParams, void **extra);

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
	PFN_cuGetErrorName cuGetErrorName;
	PFN_cuGetErrorString cuGetErrorString;
	PFN_cuCtxSetCurrent cuCtxSetCurrent;
	PFN_cuModuleLoadData cuModuleLoadData;
	PFN_cuModuleLoadDataEx cuModuleLoadDataEx;
	PFN_cuModuleGetFunction cuModuleGetFunction;
	PFN_cuLaunchKernel cuLaunchKernel;
} cuda_driver_t;

#define GGUF_CUDA_MIN_CC 60

// gguf_cuda_init загружает libcuda.so и создаёт контекст на GPU 0
// cc_out: compute capability (major*10+minor), например 120 для sm_120
int gguf_cuda_init(cuda_driver_t *drv, void **lib_out, CUcontext *ctx, char *name, size_t name_len, char *errbuf, size_t errbuf_len, int *cc_out);

// gguf_cuda_shutdown уничтожает контекст
void gguf_cuda_shutdown(cuda_driver_t *drv, CUcontext ctx);

// gguf_cuda_last_error возвращает текст последней CUDA-ошибки (если доступен)
const char *gguf_cuda_last_error(cuda_driver_t *drv, CUresult err);

// gguf_cuda_load_module загружает PTX-модуль; fn/fn_q8/fn_rmsnorm/fn_rope могут быть NULL
int gguf_cuda_load_module(cuda_driver_t *drv, CUcontext ctx, const char *ptx, CUmodule *module, CUfunction *fn, CUfunction *fn_q8, CUfunction *fn_rmsnorm, CUfunction *fn_rope, char *errbuf, size_t errbuf_len);

// gguf_cuda_upload_matrix загружает matrix на GPU
int gguf_cuda_upload_matrix(cuda_driver_t *drv, CUcontext ctx, CUdeviceptr *d_matrix, const float *matrix, int rows, int cols);

// gguf_cuda_matmul_vec_device matmul с matrix уже на GPU
int gguf_cuda_matmul_vec_device(cuda_driver_t *drv, CUcontext ctx, CUfunction fn, CUdeviceptr d_matrix, const float *vec, float *out, int rows, int cols);

// gguf_cuda_free освобождает GPU-буфер
void gguf_cuda_free(cuda_driver_t *drv, CUdeviceptr ptr);

// gguf_cuda_matmul_vec загружает matrix и запускает kernel (без кеша)
int gguf_cuda_matmul_vec(cuda_driver_t *drv, CUcontext ctx, CUfunction fn, const float *matrix, const float *vec, float *out, int rows, int cols);

// gguf_cuda_upload_q8_0 загружает Q8_0-матрицу на GPU
int gguf_cuda_upload_q8_0(cuda_driver_t *drv, CUcontext ctx, CUdeviceptr *d_matrix, const void *raw, size_t nbytes);

// gguf_cuda_matmul_vec_q8_0_device matmul Q8_0 с весами уже на GPU
int gguf_cuda_matmul_vec_q8_0_device(cuda_driver_t *drv, CUcontext ctx, CUfunction fn, CUdeviceptr d_matrix, const float *vec, float *out, int rows, int cols);

// gguf_cuda_rmsnorm RMSNorm на GPU
int gguf_cuda_rmsnorm(cuda_driver_t *drv, CUcontext ctx, CUfunction fn, const float *x, const float *weight, float *out, int n, float eps);

// gguf_cuda_rope_heads RoPE для nHeads голов (cos/sin на CPU, rotate на GPU)
int gguf_cuda_rope_heads(cuda_driver_t *drv, CUcontext ctx, CUfunction fn, float *v, const float *cos_tbl, const float *sin_tbl, int nheads, int head_dim, int half);

#endif
