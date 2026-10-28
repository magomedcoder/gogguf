//go:build cuda

package cuda

/*
#include "driver.h"
*/
import "C"

import (
	"fmt"
	"unsafe"
)

func (b *Backend) matMulMatValidate(rows, cols, batch int, x, out []float32) error {
	if batch < 1 {
		return fmt.Errorf("cuda: batch=%d", batch)
	}

	if len(x) < batch*cols {
		return fmt.Errorf("cuda: x too short for batch=%d", batch)
	}

	if len(out) < batch*rows {
		return fmt.Errorf("cuda: out too short for batch=%d", batch)
	}

	return nil
}

// MatMulMatCachedInto FP32 W * X[batch] via matmul_mat_f32 (single launch) when available.
func (b *Backend) MatMulMatCachedInto(name string, matrix []float32, rows, cols int, x []float32, batch int, out []float32) error {
	if err := b.matMulMatValidate(rows, cols, batch, x, out); err != nil {
		return err
	}
	b.mu.Lock()
	defer b.mu.Unlock()

	gm, err := b.ensureFP32Matrix(name, matrix, rows, cols)
	if err != nil {
		return err
	}

	b.lastVecAddr = 0
	b.lastVecLen = 0
	var rc C.int
	if b.hasMatF32 {
		rc = C.gguf_cuda_matmul_mat_f32_device(
			&b.drv,
			b.ctx,
			b.fnMatF32,
			b.fn,
			&b.matmulPool,
			gm.ptr,
			(*C.float)(unsafe.Pointer(&x[0])),
			(*C.float)(unsafe.Pointer(&out[0])),
			C.int(rows),
			C.int(cols),
			C.int(batch),
		)
	} else {
		rc = C.gguf_cuda_matmul_mat_device(
			&b.drv,
			b.ctx,
			b.fn,
			&b.matmulPool,
			gm.ptr,
			(*C.float)(unsafe.Pointer(&x[0])),
			(*C.float)(unsafe.Pointer(&out[0])),
			C.int(rows),
			C.int(cols),
			C.int(batch),
		)
	}

	if rc != 0 {
		return fmt.Errorf("cuda: MatMulMatCachedInto %q: code %d", name, int(rc))
	}

	return nil
}

// MatMulMatQ8_0CachedInto uses matmul_mat_q8_0 when available, else batched vec launches.
func (b *Backend) MatMulMatQ8_0CachedInto(name string, raw []byte, rows, cols int, x []float32, batch int, out []float32) error {
	if err := b.matMulMatValidate(rows, cols, batch, x, out); err != nil {
		return err
	}

	if err := validateQ8MatMul(raw, rows, cols, x[:cols]); err != nil {
		return err
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	gm, ok := b.matricesQ8[name]
	if ok {
		b.touchWeight(kindQ8, name)
	}

	if !ok || gm.rows != rows || gm.cols != cols || gm.bytes != len(raw) {
		if ok {
			b.freeWeight(kindQ8, name)
		}

		var ptr C.CUdeviceptr
		rc := b.withEvict(func() C.int {
			return C.gguf_cuda_upload_q8_0(&b.drv, b.ctx, &ptr, unsafe.Pointer(&raw[0]), C.size_t(len(raw)))
		})

		if rc != 0 {
			return uploadFail("q8_0", name, rc)
		}

		gm = gpuQ8Matrix{
			ptr:   ptr,
			rows:  rows,
			cols:  cols,
			bytes: len(raw),
		}
		b.matricesQ8[name] = gm
		b.trackWeight(kindQ8, name, gm.bytes)
	}

	b.lastVecAddr = 0
	b.lastVecLen = 0
	var rc C.int
	if b.hasMatQ8 {
		rc = C.gguf_cuda_matmul_mat_q8_0_device(
			&b.drv,
			b.ctx,
			b.fnMatQ8,
			b.fnQ8,
			&b.matmulPool,
			gm.ptr,
			(*C.float)(unsafe.Pointer(&x[0])),
			(*C.float)(unsafe.Pointer(&out[0])),
			C.int(rows),
			C.int(cols),
			C.int(batch),
		)
	} else {
		rc = C.gguf_cuda_matmul_mat_device(
			&b.drv,
			b.ctx,
			b.fnQ8,
			&b.matmulPool,
			gm.ptr,
			(*C.float)(unsafe.Pointer(&x[0])),
			(*C.float)(unsafe.Pointer(&out[0])),
			C.int(rows),
			C.int(cols),
			C.int(batch),
		)
	}
	if rc != 0 {
		return fmt.Errorf("cuda: MatMulMatQ8_0CachedInto %q: code %d", name, int(rc))
	}

	return nil
}

func (b *Backend) matMulMatQuantCachedInto(name string, raw []byte, rows, cols, batch int, x, out []float32, has bool, kind string, cache map[string]gpuQ8Matrix, fn C.CUfunction, upload func(*C.CUdeviceptr) C.int) error {
	if !has {
		return fmt.Errorf("cuda: %s matmul kernel unavailable", kind)
	}
	if err := b.matMulMatValidate(rows, cols, batch, x, out); err != nil {
		return err
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	gm, ok := cache[name]
	if ok {
		b.touchWeight(kind, name)
	}

	if !ok || gm.rows != rows || gm.cols != cols || gm.bytes != len(raw) {
		if ok {
			b.freeWeight(kind, name)
		}

		var ptr C.CUdeviceptr
		if rc := b.withEvict(func() C.int { return upload(&ptr) }); rc != 0 {
			return uploadFail(kind, name, rc)
		}

		gm = gpuQ8Matrix{
			ptr:   ptr,
			rows:  rows,
			cols:  cols,
			bytes: len(raw),
		}
		cache[name] = gm
		b.trackWeight(kind, name, gm.bytes)
	}

	b.lastVecAddr = 0
	b.lastVecLen = 0
	rc := C.gguf_cuda_matmul_mat_device(
		&b.drv,
		b.ctx,
		fn,
		&b.matmulPool,
		gm.ptr,
		(*C.float)(unsafe.Pointer(&x[0])),
		(*C.float)(unsafe.Pointer(&out[0])),
		C.int(rows),
		C.int(cols),
		C.int(batch),
	)

	if rc != 0 {
		return fmt.Errorf("cuda: MatMulMat %s %q: code %d", kind, name, int(rc))
	}

	return nil
}

func (b *Backend) MatMulMatQ4_0CachedInto(name string, raw []byte, rows, cols int, x []float32, batch int, out []float32) error {
	if err := validateQ4MatMul(raw, rows, cols, x[:cols]); err != nil {
		return err
	}

	return b.matMulMatQuantCachedInto(name, raw, rows, cols, batch, x, out, b.hasQ4, "q4_0", b.matricesQ4, b.fnQ4, func(ptr *C.CUdeviceptr) C.int {
		return C.gguf_cuda_upload_q4_0(&b.drv, b.ctx, ptr, unsafe.Pointer(&raw[0]), C.size_t(len(raw)))
	})
}

func (b *Backend) MatMulMatQ4_KCachedInto(name string, raw []byte, rows, cols int, x []float32, batch int, out []float32) error {
	if err := validateQ4KMatMul(raw, rows, cols, x[:cols]); err != nil {
		return err
	}

	return b.matMulMatQuantCachedInto(name, raw, rows, cols, batch, x, out, b.hasQ4K, "q4_k", b.matricesQ4K, b.fnQ4K, func(ptr *C.CUdeviceptr) C.int {
		return C.gguf_cuda_upload_q4_k(&b.drv, b.ctx, ptr, unsafe.Pointer(&raw[0]), C.size_t(len(raw)))
	})
}

func (b *Backend) MatMulMatQ5_KCachedInto(name string, raw []byte, rows, cols int, x []float32, batch int, out []float32) error {
	if err := validateQ5KMatMul(raw, rows, cols, x[:cols]); err != nil {
		return err
	}

	return b.matMulMatQuantCachedInto(name, raw, rows, cols, batch, x, out, b.hasQ5K, "q5_k", b.matricesQ5K, b.fnQ5K, func(ptr *C.CUdeviceptr) C.int {
		return C.gguf_cuda_upload_q5_k(&b.drv, b.ctx, ptr, unsafe.Pointer(&raw[0]), C.size_t(len(raw)))
	})
}

func (b *Backend) MatMulMatQ6_KCachedInto(name string, raw []byte, rows, cols int, x []float32, batch int, out []float32) error {
	if err := validateQ6KMatMul(raw, rows, cols, x[:cols]); err != nil {
		return err
	}

	return b.matMulMatQuantCachedInto(name, raw, rows, cols, batch, x, out, b.hasQ6K, "q6_k", b.matricesQ6K, b.fnQ6K, func(ptr *C.CUdeviceptr) C.int {
		return C.gguf_cuda_upload_q6_k(&b.drv, b.ctx, ptr, unsafe.Pointer(&raw[0]), C.size_t(len(raw)))
	})
}

func (b *Backend) MatMulMatQ4_1CachedInto(name string, raw []byte, rows, cols int, x []float32, batch int, out []float32) error {
	if err := validateQ41MatMul(raw, rows, cols, x[:cols]); err != nil {
		return err
	}

	return b.matMulMatQuantCachedInto(name, raw, rows, cols, batch, x, out, b.hasQ41, "q4_1", b.matricesQ41, b.fnQ41, func(ptr *C.CUdeviceptr) C.int {
		return C.gguf_cuda_upload_q4_1(&b.drv, b.ctx, ptr, unsafe.Pointer(&raw[0]), C.size_t(len(raw)))
	})
}

func (b *Backend) MatMulMatQ5_0CachedInto(name string, raw []byte, rows, cols int, x []float32, batch int, out []float32) error {
	if err := validateQ50MatMul(raw, rows, cols, x[:cols]); err != nil {
		return err
	}

	return b.matMulMatQuantCachedInto(name, raw, rows, cols, batch, x, out, b.hasQ50, "q5_0", b.matricesQ50, b.fnQ50, func(ptr *C.CUdeviceptr) C.int {
		return C.gguf_cuda_upload_q5_0(&b.drv, b.ctx, ptr, unsafe.Pointer(&raw[0]), C.size_t(len(raw)))
	})
}

func (b *Backend) MatMulMatQ5_1CachedInto(name string, raw []byte, rows, cols int, x []float32, batch int, out []float32) error {
	if err := validateQ51MatMul(raw, rows, cols, x[:cols]); err != nil {
		return err
	}

	return b.matMulMatQuantCachedInto(name, raw, rows, cols, batch, x, out, b.hasQ51, "q5_1", b.matricesQ51, b.fnQ51, func(ptr *C.CUdeviceptr) C.int {
		return C.gguf_cuda_upload_q5_1(&b.drv, b.ctx, ptr, unsafe.Pointer(&raw[0]), C.size_t(len(raw)))
	})
}

func (b *Backend) MatMulMatQ2_KCachedInto(name string, raw []byte, rows, cols int, x []float32, batch int, out []float32) error {
	if err := validateQ2KMatMul(raw, rows, cols, x[:cols]); err != nil {
		return err
	}

	return b.matMulMatQuantCachedInto(name, raw, rows, cols, batch, x, out, b.hasQ2K, "q2_k", b.matricesQ2K, b.fnQ2K, func(ptr *C.CUdeviceptr) C.int {
		return C.gguf_cuda_upload_q2_k(&b.drv, b.ctx, ptr, unsafe.Pointer(&raw[0]), C.size_t(len(raw)))
	})
}

func (b *Backend) MatMulMatQ3_KCachedInto(name string, raw []byte, rows, cols int, x []float32, batch int, out []float32) error {
	if err := validateQ3KMatMul(raw, rows, cols, x[:cols]); err != nil {
		return err
	}

	return b.matMulMatQuantCachedInto(name, raw, rows, cols, batch, x, out, b.hasQ3K, "q3_k", b.matricesQ3K, b.fnQ3K, func(ptr *C.CUdeviceptr) C.int {
		return C.gguf_cuda_upload_q3_k(&b.drv, b.ctx, ptr, unsafe.Pointer(&raw[0]), C.size_t(len(raw)))
	})
}

func (b *Backend) MatMulMatQ8_KCachedInto(name string, raw []byte, rows, cols int, x []float32, batch int, out []float32) error {
	if err := validateQ8KMatMul(raw, rows, cols, x[:cols]); err != nil {
		return err
	}

	return b.matMulMatQuantCachedInto(name, raw, rows, cols, batch, x, out, b.hasQ8K, "q8_k", b.matricesQ8K, b.fnQ8K, func(ptr *C.CUdeviceptr) C.int {
		return C.gguf_cuda_upload_q8_k(&b.drv, b.ctx, ptr, unsafe.Pointer(&raw[0]), C.size_t(len(raw)))
	})
}
