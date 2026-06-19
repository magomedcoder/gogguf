#include "driver.h"

#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static void *load_sym(void *lib, const char *name) {
	return dlsym(lib, name);
}

static int load_driver(cuda_driver_t *drv, void **lib_out) {
	void *lib = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
	if (!lib) {
		lib = dlopen("libcuda.so", RTLD_LAZY | RTLD_LOCAL);
	}

	if (!lib) {
		return -1;
	}

	drv->cuInit = (PFN_cuInit)load_sym(lib, "cuInit");
	drv->cuDeviceGetCount = (PFN_cuDeviceGetCount)load_sym(lib, "cuDeviceGetCount");
	drv->cuDeviceGet = (PFN_cuDeviceGet)load_sym(lib, "cuDeviceGet");
	drv->cuDeviceGetName = (PFN_cuDeviceGetName)load_sym(lib, "cuDeviceGetName");
	drv->cuCtxCreate = (PFN_cuCtxCreate_v2)load_sym(lib, "cuCtxCreate_v2");
	drv->cuCtxDestroy = (PFN_cuCtxDestroy_v2)load_sym(lib, "cuCtxDestroy_v2");
	drv->cuMemAlloc = (PFN_cuMemAlloc_v2)load_sym(lib, "cuMemAlloc_v2");
	drv->cuMemFree = (PFN_cuMemFree_v2)load_sym(lib, "cuMemFree_v2");
	drv->cuMemcpyHtoD = (PFN_cuMemcpyHtoD_v2)load_sym(lib, "cuMemcpyHtoD_v2");
	drv->cuMemcpyDtoH = (PFN_cuMemcpyDtoH_v2)load_sym(lib, "cuMemcpyDtoH_v2");
	drv->cuGetErrorName = (PFN_cuGetErrorName)load_sym(lib, "cuGetErrorName");
	drv->cuGetErrorString = (PFN_cuGetErrorString)load_sym(lib, "cuGetErrorString");
	drv->cuCtxSetCurrent = (PFN_cuCtxSetCurrent)load_sym(lib, "cuCtxSetCurrent");
	drv->cuModuleLoadData = (PFN_cuModuleLoadData)load_sym(lib, "cuModuleLoadData");
	drv->cuModuleLoadDataEx = (PFN_cuModuleLoadDataEx)load_sym(lib, "cuModuleLoadDataEx");
	drv->cuModuleGetFunction = (PFN_cuModuleGetFunction)load_sym(lib, "cuModuleGetFunction");
	drv->cuLaunchKernel = (PFN_cuLaunchKernel)load_sym(lib, "cuLaunchKernel");

	if (!drv->cuInit || !drv->cuDeviceGetCount || !drv->cuDeviceGet || !drv->cuDeviceGetName || !drv->cuCtxCreate || !drv->cuCtxDestroy || !drv->cuMemAlloc || !drv->cuMemFree || !drv->cuMemcpyHtoD || !drv->cuMemcpyDtoH || !drv->cuModuleLoadData || !drv->cuModuleGetFunction || !drv->cuLaunchKernel) {
		dlclose(lib);
		return -2;
	}

	*lib_out = lib;
	return 0;
}

int gguf_cuda_init(cuda_driver_t *drv, void **lib_out, CUcontext *ctx, char *name, size_t name_len) {
	void *lib = NULL;
	memset(drv, 0, sizeof(*drv));

	if (load_driver(drv, &lib) != 0) {
		return -1;
	}

	if (drv->cuInit(0) != CUDA_SUCCESS) {
		dlclose(lib);
		return -3;
	}

	int count = 0;
	if (drv->cuDeviceGetCount(&count) != CUDA_SUCCESS || count <= 0) {
		dlclose(lib);
		return -4;
	}

	CUdevice dev = 0;
	if (drv->cuDeviceGet(&dev, 0) != CUDA_SUCCESS) {
		dlclose(lib);
		return -5;
	}

	if (name && name_len > 0) {
		drv->cuDeviceGetName(name, (int)name_len, dev);
		name[name_len - 1] = '\0';
	}

	if (drv->cuCtxCreate(ctx, 0, dev) != CUDA_SUCCESS) {
		dlclose(lib);
		return -6;
	}

	if (lib_out) {
		*lib_out = lib;
	}

	return 0;
}

void gguf_cuda_shutdown(cuda_driver_t *drv, CUcontext ctx) {
	if (drv && drv->cuCtxDestroy && ctx) {
		drv->cuCtxDestroy(ctx);
	}
}

const char *gguf_cuda_last_error(cuda_driver_t *drv, CUresult err) {
	static const char *unknown = "unknown CUDA error";
	const char *msg = unknown;
	if (drv && drv->cuGetErrorString) {
		if (drv->cuGetErrorString(err, &msg) != CUDA_SUCCESS || !msg) {
			msg = unknown;
		}
	}

	return msg;
}

static int gguf_cuda_set_context(cuda_driver_t *drv, CUcontext ctx) {
	if (!drv->cuCtxSetCurrent) {
		return 0;
	}

	if (drv->cuCtxSetCurrent(ctx) != CUDA_SUCCESS) {
		return -1;
	}

	return 0;
}

// Значения CUjit_option из cuda.h
enum {
	GGUF_JIT_ERROR_LOG_BUFFER = 1,
	GGUF_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 2,
};

int gguf_cuda_load_module(cuda_driver_t *drv, CUcontext ctx, const char *ptx,
	CUmodule *module, CUfunction *fn, CUfunction *fn_q8, char *errbuf, size_t errbuf_len) {
	CUresult err;

	if (drv->cuCtxSetCurrent) {
		err = drv->cuCtxSetCurrent(ctx);
		if (err != CUDA_SUCCESS) {
			if (errbuf && errbuf_len > 0) {
				snprintf(errbuf, errbuf_len, "cuCtxSetCurrent: %s",
					gguf_cuda_last_error(drv, err));
			}
			return -1;
		}
	}

	if (drv->cuModuleLoadDataEx) {
		char jit_log[8192];
		jit_log[0] = '\0';
		unsigned int opts[] = {
			GGUF_JIT_ERROR_LOG_BUFFER,
			GGUF_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
		};

		void *opt_vals[] = {
			jit_log,
			(void *)(uintptr_t)sizeof(jit_log),
		};

		err = drv->cuModuleLoadDataEx(module, ptx, 2, opts, opt_vals);
		if (err != CUDA_SUCCESS) {
			if (errbuf && errbuf_len > 0) {
				snprintf(errbuf, errbuf_len, "cuModuleLoadDataEx: %s; jit: %s", gguf_cuda_last_error(drv, err), jit_log);
			}
			return -1;
		}
	} else {
		err = drv->cuModuleLoadData(module, ptx);
		if (err != CUDA_SUCCESS) {
			if (errbuf && errbuf_len > 0) {
				snprintf(errbuf, errbuf_len, "cuModuleLoadData: %s", gguf_cuda_last_error(drv, err));
			}
			return -1;
		}
	}

	err = drv->cuModuleGetFunction(fn, *module, "matmul_vec");
	if (err != CUDA_SUCCESS) {
		if (errbuf && errbuf_len > 0) {
			snprintf(errbuf, errbuf_len, "cuModuleGetFunction matmul_vec: %s", gguf_cuda_last_error(drv, err));
		}
		return -2;
	}

	if (fn_q8) {
		err = drv->cuModuleGetFunction(fn_q8, *module, "matmul_vec_q8_0");
		if (err != CUDA_SUCCESS) {
			if (errbuf && errbuf_len > 0) {
				snprintf(errbuf, errbuf_len, "cuModuleGetFunction matmul_vec_q8_0: %s", gguf_cuda_last_error(drv, err));
			}
			return -3;
		}
	}

	return 0;
}

void gguf_cuda_free(cuda_driver_t *drv, CUdeviceptr ptr) {
	if (drv && drv->cuMemFree && ptr) {
		drv->cuMemFree(ptr);
	}
}

int gguf_cuda_upload_matrix(cuda_driver_t *drv, CUcontext ctx, CUdeviceptr *d_matrix,
	const float *matrix, int rows, int cols) {
	if (gguf_cuda_set_context(drv, ctx) != 0) {
		return -10;
	}

	size_t matrix_bytes = (size_t)rows * (size_t)cols * sizeof(float);
	if (drv->cuMemAlloc(d_matrix, matrix_bytes) != CUDA_SUCCESS) {
		return -1;
	}

	if (drv->cuMemcpyHtoD(*d_matrix, matrix, matrix_bytes) != CUDA_SUCCESS) {
		drv->cuMemFree(*d_matrix);
		*d_matrix = 0;
		return -2;
	}

	return 0;
}

int gguf_cuda_matmul_vec_device(cuda_driver_t *drv, CUcontext ctx, CUfunction fn,
	CUdeviceptr d_matrix, const float *vec, float *out, int rows, int cols) {
	if (gguf_cuda_set_context(drv, ctx) != 0) {
		return -10;
	}

	CUdeviceptr d_vec = 0;
	CUdeviceptr d_out = 0;

	size_t vec_bytes = (size_t)cols * sizeof(float);
	size_t out_bytes = (size_t)rows * sizeof(float);

	if (drv->cuMemAlloc(&d_vec, vec_bytes) != CUDA_SUCCESS) {
		return -1;
	}

	if (drv->cuMemAlloc(&d_out, out_bytes) != CUDA_SUCCESS) {
		drv->cuMemFree(d_vec);
		return -2;
	}

	if (drv->cuMemcpyHtoD(d_vec, vec, vec_bytes) != CUDA_SUCCESS) {
		goto fail;
	}

	void *params[5];
	params[0] = &d_matrix;
	params[1] = &d_vec;
	params[2] = &d_out;
	params[3] = &rows;
	params[4] = &cols;

	unsigned int block = 256;
	unsigned int grid = ((unsigned int)rows + block - 1) / block;

	if (drv->cuLaunchKernel(fn, grid, 1, 1, block, 1, 1, 0, NULL, params, NULL) != CUDA_SUCCESS) {
		goto fail;
	}

	if (drv->cuMemcpyDtoH(out, d_out, out_bytes) != CUDA_SUCCESS) {
		goto fail;
	}

	drv->cuMemFree(d_vec);
	drv->cuMemFree(d_out);
	return 0;

fail:
	drv->cuMemFree(d_vec);
	drv->cuMemFree(d_out);
	return -3;
}

int gguf_cuda_matmul_vec(cuda_driver_t *drv, CUcontext ctx, CUfunction fn, const float *matrix, const float *vec, float *out, int rows, int cols) {
	CUdeviceptr d_matrix = 0;

	int rc = gguf_cuda_upload_matrix(drv, ctx, &d_matrix, matrix, rows, cols);
	if (rc != 0) {
		return rc - 10;
	}

	rc = gguf_cuda_matmul_vec_device(drv, ctx, fn, d_matrix, vec, out, rows, cols);
	gguf_cuda_free(drv, d_matrix);
	return rc;
}

int gguf_cuda_upload_q8_0(cuda_driver_t *drv, CUcontext ctx, CUdeviceptr *d_matrix,
	const void *raw, size_t nbytes) {
	if (gguf_cuda_set_context(drv, ctx) != 0) {
		return -10;
	}

	if (drv->cuMemAlloc(d_matrix, nbytes) != CUDA_SUCCESS) {
		return -1;
	}

	if (drv->cuMemcpyHtoD(*d_matrix, raw, nbytes) != CUDA_SUCCESS) {
		drv->cuMemFree(*d_matrix);
		*d_matrix = 0;
		return -2;
	}

	return 0;
}

int gguf_cuda_matmul_vec_q8_0_device(cuda_driver_t *drv, CUcontext ctx, CUfunction fn,
	CUdeviceptr d_matrix, const float *vec, float *out, int rows, int cols) {
	if (gguf_cuda_set_context(drv, ctx) != 0) {
		return -10;
	}

	CUdeviceptr d_vec = 0;
	CUdeviceptr d_out = 0;

	size_t vec_bytes = (size_t)cols * sizeof(float);
	size_t out_bytes = (size_t)rows * sizeof(float);

	if (drv->cuMemAlloc(&d_vec, vec_bytes) != CUDA_SUCCESS) {
		return -1;
	}

	if (drv->cuMemAlloc(&d_out, out_bytes) != CUDA_SUCCESS) {
		drv->cuMemFree(d_vec);
		return -2;
	}

	if (drv->cuMemcpyHtoD(d_vec, vec, vec_bytes) != CUDA_SUCCESS) {
		goto fail;
	}

	void *params[5];
	params[0] = &d_matrix;
	params[1] = &d_vec;
	params[2] = &d_out;
	params[3] = &rows;
	params[4] = &cols;

	unsigned int block = 256;
	unsigned int grid = ((unsigned int)rows + block - 1) / block;

	if (drv->cuLaunchKernel(fn, grid, 1, 1, block, 1, 1, 0, NULL, params, NULL) != CUDA_SUCCESS) {
		goto fail;
	}

	if (drv->cuMemcpyDtoH(out, d_out, out_bytes) != CUDA_SUCCESS) {
		goto fail;
	}

	drv->cuMemFree(d_vec);
	drv->cuMemFree(d_out);
	return 0;

fail:
	drv->cuMemFree(d_vec);
	drv->cuMemFree(d_out);
	return -3;
}
