//go:build cuda

package cuda

/*
#cgo LDFLAGS: -ldl
#include "driver.h"
#include <stdlib.h>
*/
import "C"
import (
	"fmt"
	"sync"
	"unsafe"

	"github.com/magomedcoder/gguf.go/pkg/quant"
)

type gpuMatrix struct {
	ptr  C.CUdeviceptr
	rows int
	cols int
}

type gpuQ8Matrix struct {
	ptr   C.CUdeviceptr
	rows  int
	cols  int
	bytes int
}

// Backend - CUDA через Driver API (libcuda.so), без cublas/cudart
type Backend struct {
	name   string
	drv    C.cuda_driver_t
	lib    unsafe.Pointer
	ctx    C.CUcontext
	module C.CUmodule
	fn     C.CUfunction
	fnQ8   C.CUfunction

	mu         sync.Mutex
	matrices   map[string]gpuMatrix
	matricesQ8 map[string]gpuQ8Matrix
}

// Open инициализирует GPU 0 и загружает kernels
func Open() (*Backend, error) {
	b := &Backend{
		matrices:   make(map[string]gpuMatrix),
		matricesQ8: make(map[string]gpuQ8Matrix),
	}

	var nameBuf [256]C.char
	rc := C.gguf_cuda_init(&b.drv, &b.lib, &b.ctx, &nameBuf[0], C.size_t(len(nameBuf)))
	if rc != 0 {
		return nil, fmt.Errorf("cuda: init: код %d", int(rc))
	}

	b.name = "CUDA:0 " + C.GoString(&nameBuf[0])

	cptx := C.CString(kernelsPTX)
	defer C.free(unsafe.Pointer(cptx))

	var errBuf [1024]C.char
	rc = C.gguf_cuda_load_module(&b.drv, b.ctx, cptx, &b.module, &b.fn, &b.fnQ8, &errBuf[0], C.size_t(len(errBuf)))
	if rc != 0 {
		C.gguf_cuda_shutdown(&b.drv, b.ctx)
		return nil, fmt.Errorf("cuda: load module: код %d: %s", int(rc), C.GoString(&errBuf[0]))
	}

	return b, nil
}

func (b *Backend) Name() string { return b.name }

func (b *Backend) Close() error {
	b.mu.Lock()
	defer b.mu.Unlock()

	for _, m := range b.matrices {
		C.gguf_cuda_free(&b.drv, m.ptr)
	}
	b.matrices = nil

	for _, m := range b.matricesQ8 {
		C.gguf_cuda_free(&b.drv, m.ptr)
	}
	b.matricesQ8 = nil

	C.gguf_cuda_shutdown(&b.drv, b.ctx)
	return nil
}

func (b *Backend) MatMulVec(matrix []float32, rows, cols int, vec []float32) ([]float32, error) {
	if err := validateMatMul(matrix, rows, cols, vec); err != nil {
		return nil, err
	}

	out := make([]float32, rows)
	rc := C.gguf_cuda_matmul_vec(
		&b.drv, b.ctx, b.fn,
		(*C.float)(unsafe.Pointer(&matrix[0])),
		(*C.float)(unsafe.Pointer(&vec[0])),
		(*C.float)(unsafe.Pointer(&out[0])),
		C.int(rows), C.int(cols),
	)
	if rc != 0 {
		return nil, fmt.Errorf("cuda: matmul_vec: код %d", int(rc))
	}

	return out, nil
}

func (b *Backend) MatMulVecCached(name string, matrix []float32, rows, cols int, vec []float32) ([]float32, error) {
	if err := validateMatMul(matrix, rows, cols, vec); err != nil {
		return nil, err
	}

	b.mu.Lock()
	gm, ok := b.matrices[name]
	if !ok || gm.rows != rows || gm.cols != cols {
		if ok {
			C.gguf_cuda_free(&b.drv, gm.ptr)
		}

		var ptr C.CUdeviceptr
		rc := C.gguf_cuda_upload_matrix(
			&b.drv, b.ctx, &ptr,
			(*C.float)(unsafe.Pointer(&matrix[0])),
			C.int(rows), C.int(cols),
		)
		if rc != 0 {
			b.mu.Unlock()
			return nil, fmt.Errorf("cuda: upload matrix %q: код %d", name, int(rc))
		}

		gm = gpuMatrix{ptr: ptr, rows: rows, cols: cols}
		b.matrices[name] = gm
	}
	b.mu.Unlock()

	out := make([]float32, rows)
	rc := C.gguf_cuda_matmul_vec_device(
		&b.drv, b.ctx, b.fn, gm.ptr,
		(*C.float)(unsafe.Pointer(&vec[0])),
		(*C.float)(unsafe.Pointer(&out[0])),
		C.int(rows), C.int(cols),
	)
	if rc != 0 {
		return nil, fmt.Errorf("cuda: matmul_vec_device %q: код %d", name, int(rc))
	}

	return out, nil
}

func (b *Backend) MatMulVecQ8_0Cached(name string, raw []byte, rows, cols int, vec []float32) ([]float32, error) {
	if err := validateQ8MatMul(raw, rows, cols, vec); err != nil {
		return nil, err
	}

	b.mu.Lock()
	gm, ok := b.matricesQ8[name]
	if !ok || gm.rows != rows || gm.cols != cols || gm.bytes != len(raw) {
		if ok {
			C.gguf_cuda_free(&b.drv, gm.ptr)
		}

		var ptr C.CUdeviceptr
		rc := C.gguf_cuda_upload_q8_0(
			&b.drv, b.ctx, &ptr,
			unsafe.Pointer(&raw[0]),
			C.size_t(len(raw)),
		)
		if rc != 0 {
			b.mu.Unlock()
			return nil, fmt.Errorf("cuda: upload q8_0 %q: код %d", name, int(rc))
		}

		gm = gpuQ8Matrix{
			ptr:   ptr,
			rows:  rows,
			cols:  cols,
			bytes: len(raw),
		}
		b.matricesQ8[name] = gm
	}
	b.mu.Unlock()

	out := make([]float32, rows)
	rc := C.gguf_cuda_matmul_vec_q8_0_device(
		&b.drv, b.ctx, b.fnQ8, gm.ptr,
		(*C.float)(unsafe.Pointer(&vec[0])),
		(*C.float)(unsafe.Pointer(&out[0])),
		C.int(rows), C.int(cols),
	)
	if rc != 0 {
		return nil, fmt.Errorf("cuda: matmul_vec_q8_0 %q: код %d", name, int(rc))
	}

	return out, nil
}

func validateMatMul(matrix []float32, rows, cols int, vec []float32) error {
	if rows <= 0 || cols <= 0 {
		return fmt.Errorf("cuda: rows=%d cols=%d", rows, cols)
	}

	if len(vec) != cols {
		return fmt.Errorf("cuda: len(vec)=%d, cols=%d", len(vec), cols)
	}

	if len(matrix) < rows*cols {
		return fmt.Errorf("cuda: matrix слишком короткая")
	}

	return nil
}

func validateQ8MatMul(raw []byte, rows, cols int, vec []float32) error {
	if rows <= 0 || cols <= 0 {
		return fmt.Errorf("cuda: rows=%d cols=%d", rows, cols)
	}

	if cols%quant.QK8_0 != 0 {
		return fmt.Errorf("cuda: cols=%d не кратно %d", cols, quant.QK8_0)
	}

	if len(vec) != cols {
		return fmt.Errorf("cuda: len(vec)=%d, cols=%d", len(vec), cols)
	}

	blocksPerRow := cols / quant.QK8_0
	want := rows * blocksPerRow * quant.BlockQ8_0Size
	if len(raw) < want {
		return fmt.Errorf("cuda: Q8_0 matrix слишком короткая")
	}

	return nil
}
