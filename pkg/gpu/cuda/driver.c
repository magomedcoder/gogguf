#include "driver.h"

#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
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
	drv->cuDeviceGetAttribute = (PFN_cuDeviceGetAttribute)load_sym(lib, "cuDeviceGetAttribute");
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

int gguf_cuda_init(cuda_driver_t *drv, void **lib_out, CUcontext *ctx, char *name, size_t name_len, char *errbuf, size_t errbuf_len, int *cc_out) {
	void *lib = NULL;
	memset(drv, 0, sizeof(*drv));

	if (errbuf && errbuf_len > 0) {
		errbuf[0] = '\0';
	}

	if (load_driver(drv, &lib) != 0) {
		if (errbuf && errbuf_len > 0) {
			snprintf(errbuf, errbuf_len, "libcuda.so не найден");
		}

		return -1;
	}

	if (drv->cuInit(0) != CUDA_SUCCESS) {
		dlclose(lib);
		if (errbuf && errbuf_len > 0) {
			snprintf(errbuf, errbuf_len, "cuInit failed");
		}

		return -3;
	}

	int count = 0;
	if (drv->cuDeviceGetCount(&count) != CUDA_SUCCESS || count <= 0) {
		dlclose(lib);
		if (errbuf && errbuf_len > 0) {
			snprintf(errbuf, errbuf_len, "GPU не найдена");
		}

		return -4;
	}

	CUdevice dev = 0;
	if (drv->cuDeviceGet(&dev, 0) != CUDA_SUCCESS) {
		dlclose(lib);
		return -5;
	}

	int cc_major = 0;
	int cc_minor = 0;
	if (drv->cuDeviceGetAttribute) {
		drv->cuDeviceGetAttribute(&cc_major, 75, dev);
		drv->cuDeviceGetAttribute(&cc_minor, 76, dev);
	}

	int cc = cc_major * 10 + cc_minor;
	if (cc > 0 && cc < GGUF_CUDA_MIN_CC) {
		dlclose(lib);
		if (name && name_len > 0) {
			snprintf(name, name_len, "GPU sm_%d.%d", cc_major, cc_minor);
		}

		if (errbuf && errbuf_len > 0) {
			snprintf(errbuf, errbuf_len, "compute capability %d.%d < %d.%d (нужен Pascal sm_60+); используйте -ngl 0", cc_major, cc_minor, GGUF_CUDA_MIN_CC / 10, GGUF_CUDA_MIN_CC % 10);
		}

		return -7;
	}

	if (name && name_len > 0) {
		char dev_name[256];
		dev_name[0] = '\0';
		drv->cuDeviceGetName(dev_name, (int)sizeof(dev_name), dev);
		if (cc > 0) {
			snprintf(name, name_len, "%s (sm_%d%d)", dev_name, cc_major, cc_minor);
		} else {
			snprintf(name, name_len, "%s", dev_name);
		}

		name[name_len - 1] = '\0';
	}

	if (drv->cuCtxCreate(ctx, 0, dev) != CUDA_SUCCESS) {
		dlclose(lib);
		if (errbuf && errbuf_len > 0) {
			snprintf(errbuf, errbuf_len, "cuCtxCreate failed");
		}

		return -6;
	}

	if (lib_out) {
		*lib_out = lib;
	}

	if (cc_out) {
		*cc_out = cc;
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
	CUmodule *module, CUfunction *fn, CUfunction *fn_q8, CUfunction *fn_rmsnorm, CUfunction *fn_rope, CUfunction *fn_swiglu, char *errbuf, size_t errbuf_len) {
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
				if (jit_log[0] != '\0') {
					snprintf(errbuf, errbuf_len, "cuModuleLoadDataEx: %s; jit: %s", gguf_cuda_last_error(drv, err), jit_log);
				} else {
					snprintf(errbuf, errbuf_len, "cuModuleLoadDataEx: %s (проверьте compute capability >= sm_60)", gguf_cuda_last_error(drv, err));
				}
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

	if (fn) {
		err = drv->cuModuleGetFunction(fn, *module, "matmul_vec");
		if (err != CUDA_SUCCESS) {
			if (errbuf && errbuf_len > 0) {
				snprintf(errbuf, errbuf_len, "cuModuleGetFunction matmul_vec: %s", gguf_cuda_last_error(drv, err));
			}
			return -2;
		}
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

	if (fn_rmsnorm) {
		err = drv->cuModuleGetFunction(fn_rmsnorm, *module, "rmsnorm");
		if (err != CUDA_SUCCESS) {
			if (errbuf && errbuf_len > 0) {
				snprintf(errbuf, errbuf_len, "cuModuleGetFunction rmsnorm: %s", gguf_cuda_last_error(drv, err));
			}
			return -4;
		}
	}

	if (fn_rope) {
		err = drv->cuModuleGetFunction(fn_rope, *module, "rope_heads");
		if (err != CUDA_SUCCESS) {
			if (errbuf && errbuf_len > 0) {
				snprintf(errbuf, errbuf_len, "cuModuleGetFunction rope_heads: %s", gguf_cuda_last_error(drv, err));
			}
			return -5;
		}
	}

	if (fn_swiglu) {
		err = drv->cuModuleGetFunction(fn_swiglu, *module, "swiglu");
		if (err != CUDA_SUCCESS) {
			if (errbuf && errbuf_len > 0) {
				snprintf(errbuf, errbuf_len, "cuModuleGetFunction swiglu: %s", gguf_cuda_last_error(drv, err));
			}
			return -6;
		}
	}

	return 0;
}

int gguf_cuda_module_function(cuda_driver_t *drv, CUmodule module, const char *name, CUfunction *fn_out) {
	if (!drv || !module || !name || !fn_out) {
		return -1;
	}

	CUresult err = drv->cuModuleGetFunction(fn_out, module, name);
	return err == CUDA_SUCCESS ? 0 : -2;
}

static void softmax_host(float *x, int n) {
	if (n <= 0) {
		return;
	}

	float maxv = x[0];
	for (int i = 1; i < n; i++) {
		if (x[i] > maxv) {
			maxv = x[i];
		}
	}

	double sum = 0.0;
	for (int i = 0; i < n; i++) {
		double e = exp((double)x[i] - (double)maxv);
		x[i] = (float)e;
		sum += e;
	}

	float inv = (float)(1.0 / sum);
	for (int i = 0; i < n; i++) {
		x[i] *= inv;
	}
}

static int gguf_cuda_attention_device(cuda_driver_t *drv, CUcontext ctx, CUfunction fn_qk, CUfunction fn_v, float *dst, const float *q, CUdeviceptr d_k, CUdeviceptr d_v,
	gguf_attn_pool_t *pool, int seq_len, int n_heads, int n_kv_heads, int head_dim) {
	if (gguf_cuda_set_context(drv, ctx) != 0) {
		return -10;
	}

	if (seq_len <= 0 || n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0 || n_heads % n_kv_heads != 0) {
		return -11;
	}

	int group = n_heads / n_kv_heads;
	int kv_stride = n_kv_heads * head_dim;
	float scale = 1.0f / sqrtf((float)head_dim);

	int q_elems = n_heads * head_dim;
	size_t q_bytes = (size_t)q_elems * sizeof(float);
	size_t dst_bytes = q_bytes;
	size_t scores_bytes = (size_t)seq_len * sizeof(float);

	CUdeviceptr d_q = 0;
	CUdeviceptr d_dst = 0;
	CUdeviceptr d_scores = 0;
	int pooled = 0;

	if (pool && pool->d_q && pool->d_dst && pool->d_scores && pool->q_elems >= q_elems && pool->max_seq >= seq_len) {
		d_q = pool->d_q;
		d_dst = pool->d_dst;
		d_scores = pool->d_scores;
		pooled = 1;
	}

	float *h_scores = (float *)malloc(scores_bytes);
	if (!h_scores) {
		return -12;
	}

	if (!pooled) {
		if (drv->cuMemAlloc(&d_q, q_bytes) != CUDA_SUCCESS) {
			free(h_scores);
			return -1;
		}

		if (drv->cuMemAlloc(&d_dst, dst_bytes) != CUDA_SUCCESS) {
			free(h_scores);
			drv->cuMemFree(d_q);
			return -1;
		}

		if (drv->cuMemAlloc(&d_scores, scores_bytes) != CUDA_SUCCESS) {
			free(h_scores);
			drv->cuMemFree(d_q);
			drv->cuMemFree(d_dst);
			return -1;
		}
	}

	if (drv->cuMemcpyHtoD(d_q, q, q_bytes) != CUDA_SUCCESS) {
		goto fail;
	}

	unsigned int block = 256;

	for (int h = 0; h < n_heads; h++) {
		int kv_head = h / group;
		int q_off = h * head_dim;
		int kv_off = kv_head * head_dim;
		CUdeviceptr d_out_head = d_dst + (CUdeviceptr)((size_t)q_off * sizeof(float));

		void *params_qk[9];
		params_qk[0] = &d_q;
		params_qk[1] = &d_k;
		params_qk[2] = &d_scores;
		params_qk[3] = &seq_len;
		params_qk[4] = &head_dim;
		params_qk[5] = &kv_stride;
		params_qk[6] = &kv_off;
		params_qk[7] = &q_off;
		params_qk[8] = &scale;

		unsigned int grid_qk = ((unsigned int)seq_len + block - 1) / block;
		if (drv->cuLaunchKernel(fn_qk, grid_qk, 1, 1, block, 1, 1, 0, NULL, params_qk, NULL) != CUDA_SUCCESS) {
			goto fail;
		}

		if (drv->cuMemcpyDtoH(h_scores, d_scores, scores_bytes) != CUDA_SUCCESS) {
			goto fail;
		}

		softmax_host(h_scores, seq_len);

		if (drv->cuMemcpyHtoD(d_scores, h_scores, scores_bytes) != CUDA_SUCCESS) {
			goto fail;
		}

		void *params_v[7];
		params_v[0] = &d_scores;
		params_v[1] = &d_v;
		params_v[2] = &d_out_head;
		params_v[3] = &seq_len;
		params_v[4] = &head_dim;
		params_v[5] = &kv_stride;
		params_v[6] = &kv_off;

		unsigned int grid_v = ((unsigned int)head_dim + block - 1) / block;
		if (drv->cuLaunchKernel(fn_v, grid_v, 1, 1, block, 1, 1, 0, NULL, params_v, NULL) != CUDA_SUCCESS) {
			goto fail;
		}
	}

	if (drv->cuMemcpyDtoH(dst, d_dst, dst_bytes) != CUDA_SUCCESS) {
		goto fail;
	}

	free(h_scores);
	if (!pooled) {
		drv->cuMemFree(d_q);
		drv->cuMemFree(d_dst);
		drv->cuMemFree(d_scores);
	}
	return 0;

fail:
	free(h_scores);
	if (!pooled) {
		drv->cuMemFree(d_q);
		drv->cuMemFree(d_dst);
		drv->cuMemFree(d_scores);
	}
	return -4;
}

int gguf_cuda_attention(cuda_driver_t *drv, CUcontext ctx, CUfunction fn_qk, CUfunction fn_v, float *dst, const float *q, const float *k, const float *v,
	int seq_len, int n_heads, int n_kv_heads, int head_dim) {
	if (gguf_cuda_set_context(drv, ctx) != 0) {
		return -10;
	}

	if (seq_len <= 0 || n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0 || n_heads % n_kv_heads != 0) {
		return -11;
	}

	int kv_stride = n_kv_heads * head_dim;
	size_t kv_bytes = (size_t)seq_len * (size_t)kv_stride * sizeof(float);

	CUdeviceptr d_k = 0;
	CUdeviceptr d_v = 0;

	if (drv->cuMemAlloc(&d_k, kv_bytes) != CUDA_SUCCESS) {
		return -1;
	}

	if (drv->cuMemAlloc(&d_v, kv_bytes) != CUDA_SUCCESS) {
		drv->cuMemFree(d_k);
		return -1;
	}

	if (drv->cuMemcpyHtoD(d_k, k, kv_bytes) != CUDA_SUCCESS) {
		drv->cuMemFree(d_k);
		drv->cuMemFree(d_v);
		return -4;
	}

	if (drv->cuMemcpyHtoD(d_v, v, kv_bytes) != CUDA_SUCCESS) {
		drv->cuMemFree(d_k);
		drv->cuMemFree(d_v);
		return -4;
	}

	int rc = gguf_cuda_attention_device(drv, ctx, fn_qk, fn_v, dst, q, d_k, d_v, NULL, seq_len, n_heads, n_kv_heads, head_dim);
	drv->cuMemFree(d_k);
	drv->cuMemFree(d_v);
	return rc;
}

int gguf_cuda_kv_init(cuda_driver_t *drv, CUcontext ctx, gguf_kv_cache_t *cache, int num_layers, int max_seq, int kv_dim) {
	if (!cache || num_layers <= 0 || max_seq <= 0 || kv_dim <= 0) {
		return -1;
	}

	if (gguf_cuda_set_context(drv, ctx) != 0) {
		return -10;
	}

	cache->num_layers = num_layers;
	cache->layers = (gguf_kv_layer_t *)calloc((size_t)num_layers, sizeof(gguf_kv_layer_t));
	if (!cache->layers) {
		return -2;
	}

	size_t bytes = (size_t)max_seq * (size_t)kv_dim * sizeof(float);
	for (int i = 0; i < num_layers; i++) {
		cache->layers[i].max_seq = max_seq;
		cache->layers[i].kv_dim = kv_dim;
		if (drv->cuMemAlloc(&cache->layers[i].d_k, bytes) != CUDA_SUCCESS) {
			gguf_cuda_kv_free(drv, cache);
			return -3;
		}

		if (drv->cuMemAlloc(&cache->layers[i].d_v, bytes) != CUDA_SUCCESS) {
			gguf_cuda_kv_free(drv, cache);
			return -3;
		}
	}

	return 0;
}

void gguf_cuda_kv_free(cuda_driver_t *drv, gguf_kv_cache_t *cache) {
	if (!cache) {
		return;
	}

	if (cache->layers) {
		for (int i = 0; i < cache->num_layers; i++) {
			if (cache->layers[i].d_k) {
				drv->cuMemFree(cache->layers[i].d_k);
			}

			if (cache->layers[i].d_v) {
				drv->cuMemFree(cache->layers[i].d_v);
			}
		}

		free(cache->layers);
	}

	cache->layers = NULL;
	cache->num_layers = 0;
}

int gguf_cuda_kv_append(cuda_driver_t *drv, CUcontext ctx, gguf_kv_cache_t *cache, int layer, int pos, const float *k, const float *v) {
	if (!cache || !cache->layers || layer < 0 || layer >= cache->num_layers || pos < 0) {
		return -1;
	}

	gguf_kv_layer_t *ly = &cache->layers[layer];
	if (pos >= ly->max_seq) {
		return -2;
	}

	if (gguf_cuda_set_context(drv, ctx) != 0) {
		return -10;
	}

	size_t off = (size_t)pos * (size_t)ly->kv_dim * sizeof(float);
	size_t bytes = (size_t)ly->kv_dim * sizeof(float);

	if (drv->cuMemcpyHtoD(ly->d_k + (CUdeviceptr)off, k, bytes) != CUDA_SUCCESS) {
		return -3;
	}

	if (drv->cuMemcpyHtoD(ly->d_v + (CUdeviceptr)off, v, bytes) != CUDA_SUCCESS) {
		return -3;
	}

	return 0;
}

int gguf_cuda_kv_attention(cuda_driver_t *drv, CUcontext ctx, CUfunction fn_qk, CUfunction fn_v, gguf_kv_cache_t *cache, gguf_attn_pool_t *pool, int layer, float *dst, const float *q, int seq_len, int n_heads, int n_kv_heads, int head_dim) {
	if (!cache || !cache->layers || layer < 0 || layer >= cache->num_layers) {
		return -1;
	}

	gguf_kv_layer_t *ly = &cache->layers[layer];
	if (seq_len > ly->max_seq) {
		return -2;
	}

	return gguf_cuda_attention_device(drv, ctx, fn_qk, fn_v, dst, q, ly->d_k, ly->d_v, pool, seq_len, n_heads, n_kv_heads, head_dim);
}

int gguf_cuda_attn_pool_init(cuda_driver_t *drv, CUcontext ctx, gguf_attn_pool_t *pool, int q_elems, int max_seq) {
	if (!pool || q_elems <= 0 || max_seq <= 0) {
		return -1;
	}

	if (gguf_cuda_set_context(drv, ctx) != 0) {
		return -10;
	}

	memset(pool, 0, sizeof(*pool));
	pool->q_elems = q_elems;
	pool->max_seq = max_seq;

	size_t q_bytes = (size_t)q_elems * sizeof(float);
	size_t score_bytes = (size_t)max_seq * sizeof(float);

	if (drv->cuMemAlloc(&pool->d_q, q_bytes) != CUDA_SUCCESS) {
		gguf_cuda_attn_pool_free(drv, pool);
		return -2;
	}

	if (drv->cuMemAlloc(&pool->d_dst, q_bytes) != CUDA_SUCCESS) {
		gguf_cuda_attn_pool_free(drv, pool);
		return -2;
	}

	if (drv->cuMemAlloc(&pool->d_scores, score_bytes) != CUDA_SUCCESS) {
		gguf_cuda_attn_pool_free(drv, pool);
		return -2;
	}

	return 0;
}

void gguf_cuda_attn_pool_free(cuda_driver_t *drv, gguf_attn_pool_t *pool) {
	if (!pool) {
		return;
	}

	if (pool->d_q) {
		drv->cuMemFree(pool->d_q);
	}

	if (pool->d_dst) {
		drv->cuMemFree(pool->d_dst);
	}

	if (pool->d_scores) {
		drv->cuMemFree(pool->d_scores);
	}

	memset(pool, 0, sizeof(*pool));
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

int gguf_cuda_rmsnorm(cuda_driver_t *drv, CUcontext ctx, CUfunction fn,
	const float *x, const float *weight, float *out, int n, float eps) {
	if (gguf_cuda_set_context(drv, ctx) != 0) {
		return -10;
	}

	if (n <= 0) {
		return -11;
	}

	CUdeviceptr d_x = 0;
	CUdeviceptr d_weight = 0;
	CUdeviceptr d_out = 0;

	size_t nbytes = (size_t)n * sizeof(float);

	if (drv->cuMemAlloc(&d_x, nbytes) != CUDA_SUCCESS) {
		return -1;
	}

	if (drv->cuMemAlloc(&d_weight, nbytes) != CUDA_SUCCESS) {
		drv->cuMemFree(d_x);
		return -2;
	}

	if (drv->cuMemAlloc(&d_out, nbytes) != CUDA_SUCCESS) {
		drv->cuMemFree(d_x);
		drv->cuMemFree(d_weight);
		return -3;
	}

	if (drv->cuMemcpyHtoD(d_x, x, nbytes) != CUDA_SUCCESS) {
		goto fail;
	}

	if (drv->cuMemcpyHtoD(d_weight, weight, nbytes) != CUDA_SUCCESS) {
		goto fail;
	}

	void *params[5];
	params[0] = &d_x;
	params[1] = &d_weight;
	params[2] = &d_out;
	params[3] = &n;
	params[4] = &eps;

	if (drv->cuLaunchKernel(fn, 1, 1, 1, 1, 1, 1, 0, NULL, params, NULL) != CUDA_SUCCESS) {
		goto fail;
	}

	if (drv->cuMemcpyDtoH(out, d_out, nbytes) != CUDA_SUCCESS) {
		goto fail;
	}

	drv->cuMemFree(d_x);
	drv->cuMemFree(d_weight);
	drv->cuMemFree(d_out);
	return 0;

fail:
	drv->cuMemFree(d_x);
	drv->cuMemFree(d_weight);
	drv->cuMemFree(d_out);
	return -4;
}

int gguf_cuda_rope_heads(cuda_driver_t *drv, CUcontext ctx, CUfunction fn, float *v, const float *cos_tbl, const float *sin_tbl, int nheads, int head_dim, int half) {
	if (gguf_cuda_set_context(drv, ctx) != 0) {
		return -10;
	}

	if (nheads <= 0 || head_dim <= 0 || half <= 0 || half*2 != head_dim) {
		return -11;
	}

	int n = nheads * head_dim;
	int tbl = half;

	CUdeviceptr d_v = 0;
	CUdeviceptr d_cos = 0;
	CUdeviceptr d_sin = 0;

	size_t v_bytes = (size_t)n * sizeof(float);
	size_t tbl_bytes = (size_t)tbl * sizeof(float);

	if (drv->cuMemAlloc(&d_v, v_bytes) != CUDA_SUCCESS) {
		return -1;
	}

	if (drv->cuMemAlloc(&d_cos, tbl_bytes) != CUDA_SUCCESS) {
		drv->cuMemFree(d_v);
		return -2;
	}

	if (drv->cuMemAlloc(&d_sin, tbl_bytes) != CUDA_SUCCESS) {
		drv->cuMemFree(d_v);
		drv->cuMemFree(d_cos);
		return -3;
	}

	if (drv->cuMemcpyHtoD(d_v, v, v_bytes) != CUDA_SUCCESS) {
		goto fail;
	}

	if (drv->cuMemcpyHtoD(d_cos, cos_tbl, tbl_bytes) != CUDA_SUCCESS) {
		goto fail;
	}

	if (drv->cuMemcpyHtoD(d_sin, sin_tbl, tbl_bytes) != CUDA_SUCCESS) {
		goto fail;
	}

	void *params[6];
	params[0] = &d_v;
	params[1] = &d_cos;
	params[2] = &d_sin;
	params[3] = &nheads;
	params[4] = &head_dim;
	params[5] = &half;

	unsigned int total = (unsigned int)(nheads * half);
	unsigned int block = 256;
	unsigned int grid = (total + block - 1) / block;

	if (drv->cuLaunchKernel(fn, grid, 1, 1, block, 1, 1, 0, NULL, params, NULL) != CUDA_SUCCESS) {
		goto fail;
	}

	if (drv->cuMemcpyDtoH(v, d_v, v_bytes) != CUDA_SUCCESS) {
		goto fail;
	}

	drv->cuMemFree(d_v);
	drv->cuMemFree(d_cos);
	drv->cuMemFree(d_sin);
	return 0;

fail:
	drv->cuMemFree(d_v);
	drv->cuMemFree(d_cos);
	drv->cuMemFree(d_sin);
	return -4;
}

int gguf_cuda_swiglu(cuda_driver_t *drv, CUcontext ctx, CUfunction fn, float *gate, const float *up, int n) {
	if (gguf_cuda_set_context(drv, ctx) != 0) {
		return -10;
	}

	if (n <= 0) {
		return -11;
	}

	CUdeviceptr d_gate = 0;
	CUdeviceptr d_up = 0;

	size_t nbytes = (size_t)n * sizeof(float);

	if (drv->cuMemAlloc(&d_gate, nbytes) != CUDA_SUCCESS) {
		return -1;
	}

	if (drv->cuMemAlloc(&d_up, nbytes) != CUDA_SUCCESS) {
		drv->cuMemFree(d_gate);
		return -2;
	}

	if (drv->cuMemcpyHtoD(d_gate, gate, nbytes) != CUDA_SUCCESS) {
		goto fail;
	}

	if (drv->cuMemcpyHtoD(d_up, up, nbytes) != CUDA_SUCCESS) {
		goto fail;
	}

	void *params[3];
	params[0] = &d_gate;
	params[1] = &d_up;
	params[2] = &n;

	unsigned int block = 256;
	unsigned int grid = ((unsigned int)n + block - 1) / block;

	if (drv->cuLaunchKernel(fn, grid, 1, 1, block, 1, 1, 0, NULL, params, NULL) != CUDA_SUCCESS) {
		goto fail;
	}

	if (drv->cuMemcpyDtoH(gate, d_gate, nbytes) != CUDA_SUCCESS) {
		goto fail;
	}

	drv->cuMemFree(d_gate);
	drv->cuMemFree(d_up);
	return 0;

fail:
	drv->cuMemFree(d_gate);
	drv->cuMemFree(d_up);
	return -4;
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

#define GGUF_GPU_Q8_BLOCK 36

static float fp16_to_fp32(uint16_t h) {
	uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
	uint32_t exp = (h >> 10) & 0x1fu;
	uint32_t mant = h & 0x3ffu;
	uint32_t f;

	if (exp == 0) {
		if (mant == 0) {
			f = sign;
		} else {
			exp = 127 - 15 + 1;
			while ((mant & 0x400u) == 0) {
				mant <<= 1;
				exp--;
			}
			mant &= 0x3ffu;
			f = sign | ((exp & 0xffu) << 23) | (mant << 13);
		}
	} else if (exp == 31) {
		f = sign | 0x7f800000u | (mant << 13);
	} else {
		f = sign | (((exp - 15 + 127) & 0xffu) << 23) | (mant << 13);
	}

	float out;
	memcpy(&out, &f, sizeof(out));

	return out;
}

int gguf_cuda_upload_q8_0(cuda_driver_t *drv, CUcontext ctx, CUdeviceptr *d_matrix,
	const void *raw, size_t nbytes) {
	if (gguf_cuda_set_context(drv, ctx) != 0) {
		return -10;
	}

	if (nbytes == 0 || nbytes % 34 != 0) {
		return -11;
	}

	size_t nblocks = nbytes / 34;
	size_t gpu_bytes = nblocks * GGUF_GPU_Q8_BLOCK;
	uint8_t *expanded = (uint8_t *)malloc(gpu_bytes);
	if (!expanded) {
		return -12;
	}

	const uint8_t *src = (const uint8_t *)raw;
	uint8_t *dst = expanded;
	for (size_t i = 0; i < nblocks; i++) {
		uint16_t scale_fp16 = (uint16_t)src[0] | ((uint16_t)src[1] << 8);
		float scale = fp16_to_fp32(scale_fp16);
		memcpy(dst, &scale, sizeof(float));
		memcpy(dst + 4, src + 2, 32);
		src += 34;
		dst += GGUF_GPU_Q8_BLOCK;
	}

	if (drv->cuMemAlloc(d_matrix, gpu_bytes) != CUDA_SUCCESS) {
		free(expanded);
		return -1;
	}

	if (drv->cuMemcpyHtoD(*d_matrix, expanded, gpu_bytes) != CUDA_SUCCESS) {
		drv->cuMemFree(*d_matrix);
		*d_matrix = 0;
		free(expanded);
		return -2;
	}

	free(expanded);

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
