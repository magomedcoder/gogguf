//go:build cuda

package cuda

/*
#include "driver.h"
*/
import "C"

import (
	"fmt"
	"unsafe"

	"github.com/magomedcoder/gogguf/pkg/dtype"
)

// matMulVecQuantCachedInto is the shared CachedInto path for all quant GEMVs (upload cache + PTX launch).
// Per-type MatMulVecQ*_CachedInto are thin wrappers.
func (b *Backend) matMulVecQuantCachedInto(t dtype.DType, name string, raw []byte, rows, cols int, vec, dst []float32) error {
	matrices, fn, err := b.quantCache(t)
	if err != nil {
		return err
	}

	if err := validateQuantMatMul(t, raw, rows, cols, vec); err != nil {
		return err
	}

	if len(dst) < rows {
		return fmt.Errorf("cuda: MatMulVecQuantCachedInto dst too short")
	}

	kind := quantKind(t)
	label := t.String()

	b.mu.Lock()
	defer b.mu.Unlock()

	gm, hit := matrices[name]
	if hit {
		b.touchWeight(kind, name)
	}

	if !hit || gm.rows != rows || gm.cols != cols || gm.bytes != len(raw) {
		if hit {
			b.freeWeight(kind, name)
		}

		var ptr C.CUdeviceptr
		rc := b.withEvict(func() C.int {
			return b.uploadQuant(t, &ptr, raw)
		})

		if rc != 0 {
			return uploadFail(label, name, rc)
		}

		gm = gpuQ8Matrix{ptr: ptr, rows: rows, cols: cols, bytes: len(raw)}
		matrices[name] = gm
		b.trackWeight(kind, name, gm.bytes)
	}

	b.prepareVecUpload(vec)
	rc := b.launchQuantVec(t, fn, gm.ptr, vec, dst, rows, cols)
	if rc != 0 {
		return fmt.Errorf("cuda: matmul_vec_%s %q: code %d", label, name, int(rc))
	}

	return nil
}

func (b *Backend) launchQuantVec(t dtype.DType, fn C.CUfunction, ptr C.CUdeviceptr, vec, dst []float32, rows, cols int) C.int {
	vp := (*C.float)(unsafe.Pointer(&vec[0]))
	dp := (*C.float)(unsafe.Pointer(&dst[0]))
	r, c := C.int(rows), C.int(cols)
	switch t {
	case dtype.Q8_0:
		return C.gguf_cuda_matmul_vec_q8_0_device(&b.drv, b.ctx, fn, &b.matmulPool, ptr, vp, dp, r, c)
	case dtype.Q4_0:
		return C.gguf_cuda_matmul_vec_q4_0_device(&b.drv, b.ctx, fn, &b.matmulPool, ptr, vp, dp, r, c)
	case dtype.Q4_K:
		return C.gguf_cuda_matmul_vec_q4_k_device(&b.drv, b.ctx, fn, &b.matmulPool, ptr, vp, dp, r, c)
	case dtype.Q5_K:
		return C.gguf_cuda_matmul_vec_q5_k_device(&b.drv, b.ctx, fn, &b.matmulPool, ptr, vp, dp, r, c)
	case dtype.Q6_K:
		return C.gguf_cuda_matmul_vec_q6_k_device(&b.drv, b.ctx, fn, &b.matmulPool, ptr, vp, dp, r, c)
	case dtype.Q4_1:
		return C.gguf_cuda_matmul_vec_q4_1_device(&b.drv, b.ctx, fn, &b.matmulPool, ptr, vp, dp, r, c)
	case dtype.Q5_0:
		return C.gguf_cuda_matmul_vec_q5_0_device(&b.drv, b.ctx, fn, &b.matmulPool, ptr, vp, dp, r, c)
	case dtype.Q5_1:
		return C.gguf_cuda_matmul_vec_q5_1_device(&b.drv, b.ctx, fn, &b.matmulPool, ptr, vp, dp, r, c)
	case dtype.Q2_K:
		return C.gguf_cuda_matmul_vec_q2_k_device(&b.drv, b.ctx, fn, &b.matmulPool, ptr, vp, dp, r, c)
	case dtype.Q3_K:
		return C.gguf_cuda_matmul_vec_q3_k_device(&b.drv, b.ctx, fn, &b.matmulPool, ptr, vp, dp, r, c)
	case dtype.Q8_K:
		return C.gguf_cuda_matmul_vec_q8_k_device(&b.drv, b.ctx, fn, &b.matmulPool, ptr, vp, dp, r, c)
	default:
		return -1
	}
}

func (b *Backend) MatMulVecQ8_0CachedInto(name string, raw []byte, rows, cols int, vec, dst []float32) error {
	return b.matMulVecQuantCachedInto(dtype.Q8_0, name, raw, rows, cols, vec, dst)
}

func (b *Backend) MatMulVecQ4_0CachedInto(name string, raw []byte, rows, cols int, vec, dst []float32) error {
	return b.matMulVecQuantCachedInto(dtype.Q4_0, name, raw, rows, cols, vec, dst)
}

func (b *Backend) MatMulVecQ4_KCachedInto(name string, raw []byte, rows, cols int, vec, dst []float32) error {
	return b.matMulVecQuantCachedInto(dtype.Q4_K, name, raw, rows, cols, vec, dst)
}

func (b *Backend) MatMulVecQ5_KCachedInto(name string, raw []byte, rows, cols int, vec, dst []float32) error {
	return b.matMulVecQuantCachedInto(dtype.Q5_K, name, raw, rows, cols, vec, dst)
}

func (b *Backend) MatMulVecQ6_KCachedInto(name string, raw []byte, rows, cols int, vec, dst []float32) error {
	return b.matMulVecQuantCachedInto(dtype.Q6_K, name, raw, rows, cols, vec, dst)
}

func (b *Backend) MatMulVecQ4_1CachedInto(name string, raw []byte, rows, cols int, vec, dst []float32) error {
	return b.matMulVecQuantCachedInto(dtype.Q4_1, name, raw, rows, cols, vec, dst)
}

func (b *Backend) MatMulVecQ5_0CachedInto(name string, raw []byte, rows, cols int, vec, dst []float32) error {
	return b.matMulVecQuantCachedInto(dtype.Q5_0, name, raw, rows, cols, vec, dst)
}

func (b *Backend) MatMulVecQ5_1CachedInto(name string, raw []byte, rows, cols int, vec, dst []float32) error {
	return b.matMulVecQuantCachedInto(dtype.Q5_1, name, raw, rows, cols, vec, dst)
}

func (b *Backend) MatMulVecQ2_KCachedInto(name string, raw []byte, rows, cols int, vec, dst []float32) error {
	return b.matMulVecQuantCachedInto(dtype.Q2_K, name, raw, rows, cols, vec, dst)
}

func (b *Backend) MatMulVecQ3_KCachedInto(name string, raw []byte, rows, cols int, vec, dst []float32) error {
	return b.matMulVecQuantCachedInto(dtype.Q3_K, name, raw, rows, cols, vec, dst)
}

func (b *Backend) MatMulVecQ8_KCachedInto(name string, raw []byte, rows, cols int, vec, dst []float32) error {
	return b.matMulVecQuantCachedInto(dtype.Q8_K, name, raw, rows, cols, vec, dst)
}
