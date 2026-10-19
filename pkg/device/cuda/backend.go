//go:build cuda

package cuda

/*
#cgo !windows LDFLAGS: -ldl -lm
#include "driver.h"
#include <stdlib.h>
*/
import "C"
import (
	"errors"
	"fmt"
	"sync"
	"unsafe"

	"github.com/magomedcoder/gogguf/pkg/compute"
	"github.com/magomedcoder/gogguf/pkg/dtype"
	"github.com/magomedcoder/gogguf/pkg/quant"
)

// errUploadOOM - cuMemAlloc returned code -1 when uploading weights to device
var errUploadOOM = errors.New("out of VRAM")

// uploadFail formats upload error; code -1 marks VRAM exhaustion
func uploadFail(kind, name string, rc C.int) error {
	if rc == -1 {
		return fmt.Errorf("cuda: upload %s %q: %w", kind, name, errUploadOOM)
	}

	return fmt.Errorf("cuda: upload %s %q: code %d", kind, name, int(rc))
}

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

// Backend - CUDA via Driver API (libcuda.so), without cublas/cudart
type Backend struct {
	name            string
	device          int
	drv             C.cuda_driver_t
	lib             unsafe.Pointer
	ctx             C.CUcontext
	module          C.CUmodule
	moduleOps       C.CUmodule
	fn              C.CUfunction
	fnQ8            C.CUfunction
	fnMatQ8         C.CUfunction // matmul_mat_q8_0 (batched GEMM)
	fnQ4            C.CUfunction
	fnQ4K           C.CUfunction
	fnQ5K           C.CUfunction
	fnQ6K           C.CUfunction
	fnQ41           C.CUfunction
	fnQ50           C.CUfunction
	fnQ51           C.CUfunction
	fnQ2K           C.CUfunction
	fnQ3K           C.CUfunction
	fnQ8K           C.CUfunction
	hasMatQ8        bool
	fnRMS           C.CUfunction
	fnRoPE          C.CUfunction
	fnRoPENorm      C.CUfunction
	fnSwiGLU        C.CUfunction
	fnGeGLU         C.CUfunction
	fnAttnQK        C.CUfunction
	fnAttnV         C.CUfunction
	fnSoftmax       C.CUfunction
	fnAttnFlash     C.CUfunction // attn_flash (fused QK+softmax+V, optional)
	fnMLA           C.CUfunction // mla_attn (absorbed MLA, optional)
	fnAdd           C.CUfunction
	fnEmbedGather   C.CUfunction
	fnLogitsArgmax  C.CUfunction
	hasRMS          bool
	hasRoPE         bool
	hasRoPENorm     bool
	hasSwiGLU       bool
	hasGeGLU        bool
	hasAttn         bool
	hasSoftmax      bool
	hasAttnFlash    bool
	hasMLA          bool
	hasAdd          bool
	hasGraphs       bool
	hasQ4           bool
	hasQ4K          bool
	hasQ5K          bool
	hasQ6K          bool
	hasQ41          bool
	hasQ50          bool
	hasQ51          bool
	hasQ2K          bool
	hasQ3K          bool
	hasQ8K          bool
	hasEmbedGather  bool
	hasLogitsArgmax bool

	mu          sync.Mutex
	matrices    map[string]gpuMatrix
	matricesQ8  map[string]gpuQ8Matrix
	matricesQ4  map[string]gpuQ8Matrix
	matricesQ4K map[string]gpuQ8Matrix
	matricesQ5K map[string]gpuQ8Matrix
	matricesQ6K map[string]gpuQ8Matrix
	matricesQ41 map[string]gpuQ8Matrix
	matricesQ50 map[string]gpuQ8Matrix
	matricesQ51 map[string]gpuQ8Matrix
	matricesQ2K map[string]gpuQ8Matrix
	matricesQ3K map[string]gpuQ8Matrix
	matricesQ8K map[string]gpuQ8Matrix
	kvCache     C.gguf_kv_cache_t
	kvReady     bool
	mlaCache    C.gguf_mla_cache_t
	mlaReady    bool
	attnPool    C.gguf_attn_pool_t
	matmulPool  C.gguf_matmul_pool_t
	wc          weightCache // LRU over uploaded weight matrices (see weight_cache.go)
	lastVecAddr uintptr
	lastVecLen  int

	// hiddenResident: hidden state lives in matmulPool.d_resid between layers
	hiddenResident bool

	// peers caches P2P probe results per other device (see peer.go); guarded by mu
	peers map[*Backend]bool
}

// Open initializes first visible device (GPU 0) and loads kernels
func Open() (*Backend, error) {
	return OpenDevice(0)
}

// OpenDevice initializes device with ordinal and loads kernels.
// Numbering - after CUDA_VISIBLE_DEVICES filter, as in llama.cpp
func OpenDevice(device int) (*Backend, error) {
	if device < 0 {
		return nil, fmt.Errorf("cuda: device=%d: ordinal must be >= 0", device)
	}

	b := &Backend{
		device:      device,
		matrices:    make(map[string]gpuMatrix),
		matricesQ8:  make(map[string]gpuQ8Matrix),
		matricesQ4:  make(map[string]gpuQ8Matrix),
		matricesQ4K: make(map[string]gpuQ8Matrix),
		matricesQ5K: make(map[string]gpuQ8Matrix),
		matricesQ6K: make(map[string]gpuQ8Matrix),
		matricesQ41: make(map[string]gpuQ8Matrix),
		matricesQ50: make(map[string]gpuQ8Matrix),
		matricesQ51: make(map[string]gpuQ8Matrix),
		matricesQ2K: make(map[string]gpuQ8Matrix),
		matricesQ3K: make(map[string]gpuQ8Matrix),
		matricesQ8K: make(map[string]gpuQ8Matrix),
	}

	b.loadWeightCacheEnv()

	var nameBuf [256]C.char
	var initErr [512]C.char
	var cc C.int
	rc := C.gguf_cuda_init(&b.drv, &b.lib, &b.ctx, C.int(device), &nameBuf[0], C.size_t(len(nameBuf)), &initErr[0], C.size_t(len(initErr)), &cc)
	if rc != 0 {
		msg := C.GoString(&initErr[0])
		if msg == "" {
			msg = C.GoString(&nameBuf[0])
		}

		return nil, fmt.Errorf("cuda: device init %d: code %d: %s", device, int(rc), msg)
	}

	b.name = fmt.Sprintf("CUDA:%d %s", device, C.GoString(&nameBuf[0]))
	b.hasGraphs = b.drv.has_graphs != 0
	gpuCC := int(cc)

	var errBuf [4096]C.char
	if err := b.loadMatmulModule(gpuCC, &errBuf); err != nil {
		C.gguf_cuda_shutdown(&b.drv, b.ctx)
		return nil, err
	}

	if err := b.loadOpsModule(gpuCC, &errBuf); err != nil {
		C.gguf_cuda_shutdown(&b.drv, b.ctx)
		return nil, err
	}

	if rc := C.gguf_cuda_matmul_pool_init(&b.drv, b.ctx, &b.matmulPool); rc != 0 {
		C.gguf_cuda_shutdown(&b.drv, b.ctx)
		return nil, fmt.Errorf("cuda: matmul pool init: code %d", int(rc))
	}

	return b, nil
}

func (b *Backend) loadMatmulModule(gpuCC int, errBuf *[4096]C.char) error {
	var lastErr string
	for _, target := range ptxTargets(gpuCC) {
		ptx := kernelsPTXForTarget(target)
		cptx := C.CString(ptx)
		rc := C.gguf_cuda_load_module(&b.drv, b.ctx, cptx, &b.module, &b.fn, &b.fnQ8, nil, nil, nil, &errBuf[0], C.size_t(len(errBuf)))
		C.free(unsafe.Pointer(cptx))
		if rc == 0 {
			cMatQ8 := C.CString("matmul_mat_q8_0")
			if C.gguf_cuda_module_function(&b.drv, b.module, cMatQ8, &b.fnMatQ8) == 0 {
				b.hasMatQ8 = true
			}
			C.free(unsafe.Pointer(cMatQ8))

			cQ4 := C.CString("matmul_vec_q4_0")
			if C.gguf_cuda_module_function(&b.drv, b.module, cQ4, &b.fnQ4) == 0 {
				b.hasQ4 = true
			}
			C.free(unsafe.Pointer(cQ4))

			cQ4K := C.CString("matmul_vec_q4_k")
			if C.gguf_cuda_module_function(&b.drv, b.module, cQ4K, &b.fnQ4K) == 0 {
				b.hasQ4K = true
			}
			C.free(unsafe.Pointer(cQ4K))

			cQ5K := C.CString("matmul_vec_q5_k")
			if C.gguf_cuda_module_function(&b.drv, b.module, cQ5K, &b.fnQ5K) == 0 {
				b.hasQ5K = true
			}
			C.free(unsafe.Pointer(cQ5K))

			cQ6K := C.CString("matmul_vec_q6_k")
			if C.gguf_cuda_module_function(&b.drv, b.module, cQ6K, &b.fnQ6K) == 0 {
				b.hasQ6K = true
			}
			C.free(unsafe.Pointer(cQ6K))

			cQ41 := C.CString("matmul_vec_q4_1")
			if C.gguf_cuda_module_function(&b.drv, b.module, cQ41, &b.fnQ41) == 0 {
				b.hasQ41 = true
			}
			C.free(unsafe.Pointer(cQ41))

			cQ50 := C.CString("matmul_vec_q5_0")
			if C.gguf_cuda_module_function(&b.drv, b.module, cQ50, &b.fnQ50) == 0 {
				b.hasQ50 = true
			}
			C.free(unsafe.Pointer(cQ50))

			cQ51 := C.CString("matmul_vec_q5_1")
			if C.gguf_cuda_module_function(&b.drv, b.module, cQ51, &b.fnQ51) == 0 {
				b.hasQ51 = true
			}
			C.free(unsafe.Pointer(cQ51))

			cQ2K := C.CString("matmul_vec_q2_k")
			if C.gguf_cuda_module_function(&b.drv, b.module, cQ2K, &b.fnQ2K) == 0 {
				b.hasQ2K = true
			}
			C.free(unsafe.Pointer(cQ2K))

			cQ3K := C.CString("matmul_vec_q3_k")
			if C.gguf_cuda_module_function(&b.drv, b.module, cQ3K, &b.fnQ3K) == 0 {
				b.hasQ3K = true
			}
			C.free(unsafe.Pointer(cQ3K))

			cQ8K := C.CString("matmul_vec_q8_k")
			if C.gguf_cuda_module_function(&b.drv, b.module, cQ8K, &b.fnQ8K) == 0 {
				b.hasQ8K = true
			}
			C.free(unsafe.Pointer(cQ8K))

			return nil
		}

		lastErr = C.GoString(&errBuf[0])
	}

	return fmt.Errorf("cuda: load matmul module (cc=%d): %s", gpuCC, lastErr)
}

func (b *Backend) loadOpsModule(gpuCC int, errBuf *[4096]C.char) error {
	var lastErr string
	for _, target := range ptxTargets(gpuCC) {
		ptx := opsPTXForTarget(target)
		cptx := C.CString(ptx)
		rc := C.gguf_cuda_load_module(&b.drv, b.ctx, cptx, &b.moduleOps, nil, nil, &b.fnRMS, &b.fnRoPE, &b.fnSwiGLU, &errBuf[0], C.size_t(len(errBuf)))
		C.free(unsafe.Pointer(cptx))
		if rc == 0 {
			b.hasRMS = true
			b.hasRoPE = true
			b.hasSwiGLU = true

			cRoPENorm := C.CString("rope_heads_norm")
			if C.gguf_cuda_module_function(&b.drv, b.moduleOps, cRoPENorm, &b.fnRoPENorm) == 0 {
				b.hasRoPENorm = true
			}
			C.free(unsafe.Pointer(cRoPENorm))

			cGeGLU := C.CString("geglu")
			if C.gguf_cuda_module_function(&b.drv, b.moduleOps, cGeGLU, &b.fnGeGLU) == 0 {
				b.hasGeGLU = true
			}
			C.free(unsafe.Pointer(cGeGLU))

			cAttnQK := C.CString("attn_qk")
			cAttnV := C.CString("attn_v")
			if C.gguf_cuda_module_function(&b.drv, b.moduleOps, cAttnQK, &b.fnAttnQK) == 0 && C.gguf_cuda_module_function(&b.drv, b.moduleOps, cAttnV, &b.fnAttnV) == 0 {
				b.hasAttn = true
			}
			C.free(unsafe.Pointer(cAttnQK))
			C.free(unsafe.Pointer(cAttnV))

			cFlash := C.CString("attn_flash")
			if C.gguf_cuda_module_function(&b.drv, b.moduleOps, cFlash, &b.fnAttnFlash) == 0 {
				b.hasAttnFlash = true
				b.drv.fn_attn_flash = b.fnAttnFlash
			}
			C.free(unsafe.Pointer(cFlash))

			cMLA := C.CString("mla_attn")
			if C.gguf_cuda_module_function(&b.drv, b.moduleOps, cMLA, &b.fnMLA) == 0 {
				b.hasMLA = true
			}
			C.free(unsafe.Pointer(cMLA))

			cSoftmax := C.CString("softmax")
			if C.gguf_cuda_module_function(&b.drv, b.moduleOps, cSoftmax, &b.fnSoftmax) == 0 {
				b.hasSoftmax = true
			}
			C.free(unsafe.Pointer(cSoftmax))

			cAdd := C.CString("add_inplace")
			if C.gguf_cuda_module_function(&b.drv, b.moduleOps, cAdd, &b.fnAdd) == 0 {
				b.hasAdd = true
			}
			C.free(unsafe.Pointer(cAdd))

			cEmb := C.CString("embed_gather_q8_0")
			if C.gguf_cuda_module_function(&b.drv, b.moduleOps, cEmb, &b.fnEmbedGather) == 0 {
				b.hasEmbedGather = true
			}
			C.free(unsafe.Pointer(cEmb))

			cArg := C.CString("logits_argmax")
			if C.gguf_cuda_module_function(&b.drv, b.moduleOps, cArg, &b.fnLogitsArgmax) == 0 {
				b.hasLogitsArgmax = true
			}
			C.free(unsafe.Pointer(cArg))

			return nil
		}

		lastErr = C.GoString(&errBuf[0])
	}

	return fmt.Errorf("cuda: load ops module (cc=%d): %s", gpuCC, lastErr)
}

func (b *Backend) Name() string { return b.name }

// Device returns device ordinal (after CUDA_VISIBLE_DEVICES)
func (b *Backend) Device() int { return b.device }

// VRAMInfo returns used and total device VRAM (cuMemGetInfo)
func (b *Backend) VRAMInfo() (used, total uint64, err error) {
	b.mu.Lock()
	defer b.mu.Unlock()

	var free, tot C.size_t
	if rc := C.gguf_cuda_mem_info(&b.drv, b.ctx, &free, &tot); rc != 0 {
		return 0, 0, fmt.Errorf("cuda: mem_info: code %d", int(rc))
	}

	return uint64(tot) - uint64(free), uint64(tot), nil
}

func (b *Backend) Close() error {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.kvReady {
		C.gguf_cuda_kv_free(&b.drv, &b.kvCache)
		b.kvReady = false
	}

	if b.mlaReady {
		C.gguf_cuda_mla_free(&b.drv, &b.mlaCache)
		b.mlaReady = false
	}

	C.gguf_cuda_attn_pool_free(&b.drv, &b.attnPool)
	C.gguf_cuda_matmul_pool_free(&b.drv, &b.matmulPool)

	for _, m := range b.matrices {
		C.gguf_cuda_free(&b.drv, m.ptr)
	}
	b.matrices = nil

	for _, m := range b.matricesQ8 {
		C.gguf_cuda_free(&b.drv, m.ptr)
	}
	b.matricesQ8 = nil

	for _, m := range b.matricesQ4 {
		C.gguf_cuda_free(&b.drv, m.ptr)
	}
	b.matricesQ4 = nil

	for _, m := range b.matricesQ4K {
		C.gguf_cuda_free(&b.drv, m.ptr)
	}
	b.matricesQ4K = nil

	for _, m := range b.matricesQ5K {
		C.gguf_cuda_free(&b.drv, m.ptr)
	}
	b.matricesQ5K = nil

	for _, m := range b.matricesQ6K {
		C.gguf_cuda_free(&b.drv, m.ptr)
	}
	b.matricesQ6K = nil

	for _, m := range b.matricesQ41 {
		C.gguf_cuda_free(&b.drv, m.ptr)
	}
	b.matricesQ41 = nil

	for _, m := range b.matricesQ50 {
		C.gguf_cuda_free(&b.drv, m.ptr)
	}
	b.matricesQ50 = nil

	for _, m := range b.matricesQ51 {
		C.gguf_cuda_free(&b.drv, m.ptr)
	}
	b.matricesQ51 = nil

	for _, m := range b.matricesQ2K {
		C.gguf_cuda_free(&b.drv, m.ptr)
	}
	b.matricesQ2K = nil

	for _, m := range b.matricesQ3K {
		C.gguf_cuda_free(&b.drv, m.ptr)
	}
	b.matricesQ3K = nil

	for _, m := range b.matricesQ8K {
		C.gguf_cuda_free(&b.drv, m.ptr)
	}
	b.matricesQ8K = nil

	C.gguf_cuda_shutdown(&b.drv, b.ctx)
	b.ctx = nil
	b.hiddenResident = false
	b.peers = nil

	return nil
}

func (b *Backend) MatMulVec(matrix []float32, rows, cols int, vec []float32) ([]float32, error) {
	if err := validateMatMul(matrix, rows, cols, vec); err != nil {
		return nil, err
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	out := make([]float32, rows)
	rc := C.gguf_cuda_matmul_vec(
		&b.drv,
		b.ctx,
		b.fn,
		&b.matmulPool,
		(*C.float)(unsafe.Pointer(&matrix[0])),
		(*C.float)(unsafe.Pointer(&vec[0])),
		(*C.float)(unsafe.Pointer(&out[0])),
		C.int(rows),
		C.int(cols),
	)
	if rc != 0 {
		return nil, fmt.Errorf("cuda: matmul_vec: code %d", int(rc))
	}

	return out, nil
}

func (b *Backend) MatMulVecCached(name string, matrix []float32, rows, cols int, vec []float32) ([]float32, error) {
	out := make([]float32, rows)
	if err := b.MatMulVecCachedInto(name, matrix, rows, cols, vec, out); err != nil {
		return nil, err
	}
	return out, nil
}

// MatMulVecCachedInto writes matmul result into dst [rows] (no extra host alloc for the result).
func (b *Backend) MatMulVecCachedInto(name string, matrix []float32, rows, cols int, vec, dst []float32) error {
	if err := validateMatMul(matrix, rows, cols, vec); err != nil {
		return err
	}
	if len(dst) < rows {
		return fmt.Errorf("cuda: MatMulVecCachedInto dst too short")
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	gm, ok := b.matrices[name]
	if ok {
		b.touchWeight(kindF32, name)
	}
	if !ok || gm.rows != rows || gm.cols != cols {
		if ok {
			b.freeWeight(kindF32, name)
		}

		var ptr C.CUdeviceptr
		rc := b.withEvict(func() C.int {
			return C.gguf_cuda_upload_matrix(
				&b.drv,
				b.ctx,
				&ptr,
				(*C.float)(unsafe.Pointer(&matrix[0])),
				C.int(rows),
				C.int(cols),
			)
		})
		if rc != 0 {
			return uploadFail("matrix", name, rc)
		}

		gm = gpuMatrix{ptr: ptr, rows: rows, cols: cols}
		b.matrices[name] = gm
		b.trackWeight(kindF32, name, int(gm.rows)*int(gm.cols)*4)
	}

	b.prepareVecUpload(vec)
	rc := C.gguf_cuda_matmul_vec_device(
		&b.drv,
		b.ctx,
		b.fn,
		&b.matmulPool,
		gm.ptr,
		(*C.float)(unsafe.Pointer(&vec[0])),
		(*C.float)(unsafe.Pointer(&dst[0])),
		C.int(rows),
		C.int(cols),
	)
	if rc != 0 {
		return fmt.Errorf("cuda: matmul_vec_device %q: code %d", name, int(rc))
	}
	return nil
}

func (b *Backend) MatMulVecQ8_0Cached(name string, raw []byte, rows, cols int, vec []float32) ([]float32, error) {
	out := make([]float32, rows)
	if err := b.MatMulVecQ8_0CachedInto(name, raw, rows, cols, vec, out); err != nil {
		return nil, err
	}

	return out, nil
}

func (b *Backend) MatMulVecQ8_0CachedInto(name string, raw []byte, rows, cols int, vec, dst []float32) error {
	if err := validateQ8MatMul(raw, rows, cols, vec); err != nil {
		return err
	}

	if len(dst) < rows {
		return fmt.Errorf("cuda: MatMulVecQ8_0CachedInto dst too short")
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
			return C.gguf_cuda_upload_q8_0(
				&b.drv,
				b.ctx,
				&ptr,
				unsafe.Pointer(&raw[0]),
				C.size_t(len(raw)),
			)
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

	b.prepareVecUpload(vec)
	rc := C.gguf_cuda_matmul_vec_q8_0_device(
		&b.drv,
		b.ctx,
		b.fnQ8,
		&b.matmulPool,
		gm.ptr,
		(*C.float)(unsafe.Pointer(&vec[0])),
		(*C.float)(unsafe.Pointer(&dst[0])),
		C.int(rows),
		C.int(cols),
	)
	if rc != 0 {
		return fmt.Errorf("cuda: matmul_vec_q8_0 %q: code %d", name, int(rc))
	}

	return nil
}

func (b *Backend) MatMulVecQ4_0Cached(name string, raw []byte, rows, cols int, vec []float32) ([]float32, error) {
	out := make([]float32, rows)
	if err := b.MatMulVecQ4_0CachedInto(name, raw, rows, cols, vec, out); err != nil {
		return nil, err
	}
	return out, nil
}

func (b *Backend) MatMulVecQ4_0CachedInto(name string, raw []byte, rows, cols int, vec, dst []float32) error {
	if !b.hasQ4 {
		return fmt.Errorf("cuda: q4_0 matmul kernel unavailable")
	}

	if err := validateQ4MatMul(raw, rows, cols, vec); err != nil {
		return err
	}

	if len(dst) < rows {
		return fmt.Errorf("cuda: MatMulVecQ4_0CachedInto dst too short")
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	gm, ok := b.matricesQ4[name]
	if ok {
		b.touchWeight(kindQ4, name)
	}

	if !ok || gm.rows != rows || gm.cols != cols || gm.bytes != len(raw) {
		if ok {
			b.freeWeight(kindQ4, name)
		}

		var ptr C.CUdeviceptr
		rc := b.withEvict(func() C.int {
			return C.gguf_cuda_upload_q4_0(
				&b.drv,
				b.ctx,
				&ptr,
				unsafe.Pointer(&raw[0]),
				C.size_t(len(raw)),
			)
		})
		if rc != 0 {
			return uploadFail("q4_0", name, rc)
		}

		gm = gpuQ8Matrix{
			ptr:   ptr,
			rows:  rows,
			cols:  cols,
			bytes: len(raw),
		}
		b.matricesQ4[name] = gm
		b.trackWeight(kindQ4, name, gm.bytes)
	}

	b.prepareVecUpload(vec)
	rc := C.gguf_cuda_matmul_vec_q4_0_device(
		&b.drv,
		b.ctx,
		b.fnQ4,
		&b.matmulPool,
		gm.ptr,
		(*C.float)(unsafe.Pointer(&vec[0])),
		(*C.float)(unsafe.Pointer(&dst[0])),
		C.int(rows),
		C.int(cols),
	)
	if rc != 0 {
		return fmt.Errorf("cuda: matmul_vec_q4_0 %q: code %d", name, int(rc))
	}

	return nil
}

func (b *Backend) MatMulVecQ4_KCached(name string, raw []byte, rows, cols int, vec []float32) ([]float32, error) {
	out := make([]float32, rows)
	if err := b.MatMulVecQ4_KCachedInto(name, raw, rows, cols, vec, out); err != nil {
		return nil, err
	}

	return out, nil
}

func (b *Backend) MatMulVecQ4_KCachedInto(name string, raw []byte, rows, cols int, vec, dst []float32) error {
	if !b.hasQ4K {
		return fmt.Errorf("cuda: q4_k matmul kernel unavailable")
	}

	if err := validateQ4KMatMul(raw, rows, cols, vec); err != nil {
		return err
	}

	if len(dst) < rows {
		return fmt.Errorf("cuda: MatMulVecQ4_KCachedInto dst too short")
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	gm, ok := b.matricesQ4K[name]
	if ok {
		b.touchWeight(kindQ4K, name)
	}

	if !ok || gm.rows != rows || gm.cols != cols || gm.bytes != len(raw) {
		if ok {
			b.freeWeight(kindQ4K, name)
		}

		var ptr C.CUdeviceptr
		rc := b.withEvict(func() C.int {
			return C.gguf_cuda_upload_q4_k(
				&b.drv,
				b.ctx,
				&ptr,
				unsafe.Pointer(&raw[0]),
				C.size_t(len(raw)),
			)
		})
		if rc != 0 {
			return uploadFail("q4_k", name, rc)
		}

		gm = gpuQ8Matrix{
			ptr:   ptr,
			rows:  rows,
			cols:  cols,
			bytes: len(raw),
		}
		b.matricesQ4K[name] = gm
		b.trackWeight(kindQ4K, name, gm.bytes)
	}

	b.prepareVecUpload(vec)
	rc := C.gguf_cuda_matmul_vec_q4_k_device(
		&b.drv,
		b.ctx,
		b.fnQ4K,
		&b.matmulPool,
		gm.ptr,
		(*C.float)(unsafe.Pointer(&vec[0])),
		(*C.float)(unsafe.Pointer(&dst[0])),
		C.int(rows),
		C.int(cols),
	)
	if rc != 0 {
		return fmt.Errorf("cuda: matmul_vec_q4_k %q: code %d", name, int(rc))
	}

	return nil
}

func (b *Backend) MatMulVecQ5_KCached(name string, raw []byte, rows, cols int, vec []float32) ([]float32, error) {
	out := make([]float32, rows)
	if err := b.MatMulVecQ5_KCachedInto(name, raw, rows, cols, vec, out); err != nil {
		return nil, err
	}

	return out, nil
}

func (b *Backend) MatMulVecQ5_KCachedInto(name string, raw []byte, rows, cols int, vec, dst []float32) error {
	if !b.hasQ5K {
		return fmt.Errorf("cuda: q5_k matmul kernel unavailable")
	}

	if err := validateQ5KMatMul(raw, rows, cols, vec); err != nil {
		return err
	}

	if len(dst) < rows {
		return fmt.Errorf("cuda: MatMulVecQ5_KCachedInto dst too short")
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	gm, ok := b.matricesQ5K[name]
	if ok {
		b.touchWeight(kindQ5K, name)
	}

	if !ok || gm.rows != rows || gm.cols != cols || gm.bytes != len(raw) {
		if ok {
			b.freeWeight(kindQ5K, name)
		}

		var ptr C.CUdeviceptr
		rc := b.withEvict(func() C.int {
			return C.gguf_cuda_upload_q5_k(
				&b.drv,
				b.ctx,
				&ptr,
				unsafe.Pointer(&raw[0]),
				C.size_t(len(raw)),
			)
		})
		if rc != 0 {
			return uploadFail("q5_k", name, rc)
		}

		gm = gpuQ8Matrix{
			ptr:   ptr,
			rows:  rows,
			cols:  cols,
			bytes: len(raw),
		}
		b.matricesQ5K[name] = gm
		b.trackWeight(kindQ5K, name, gm.bytes)
	}

	b.prepareVecUpload(vec)
	rc := C.gguf_cuda_matmul_vec_q5_k_device(
		&b.drv,
		b.ctx,
		b.fnQ5K,
		&b.matmulPool,
		gm.ptr,
		(*C.float)(unsafe.Pointer(&vec[0])),
		(*C.float)(unsafe.Pointer(&dst[0])),
		C.int(rows),
		C.int(cols),
	)
	if rc != 0 {
		return fmt.Errorf("cuda: matmul_vec_q5_k %q: code %d", name, int(rc))
	}

	return nil
}

func (b *Backend) MatMulVecQ6_KCached(name string, raw []byte, rows, cols int, vec []float32) ([]float32, error) {
	out := make([]float32, rows)
	if err := b.MatMulVecQ6_KCachedInto(name, raw, rows, cols, vec, out); err != nil {
		return nil, err
	}

	return out, nil
}

func (b *Backend) MatMulVecQ6_KCachedInto(name string, raw []byte, rows, cols int, vec, dst []float32) error {
	if !b.hasQ6K {
		return fmt.Errorf("cuda: q6_k matmul kernel unavailable")
	}

	if err := validateQ6KMatMul(raw, rows, cols, vec); err != nil {
		return err
	}

	if len(dst) < rows {
		return fmt.Errorf("cuda: MatMulVecQ6_KCachedInto dst too short")
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	gm, ok := b.matricesQ6K[name]
	if ok {
		b.touchWeight(kindQ6K, name)
	}

	if !ok || gm.rows != rows || gm.cols != cols || gm.bytes != len(raw) {
		if ok {
			b.freeWeight(kindQ6K, name)
		}

		var ptr C.CUdeviceptr
		rc := b.withEvict(func() C.int {
			return C.gguf_cuda_upload_q6_k(
				&b.drv,
				b.ctx,
				&ptr,
				unsafe.Pointer(&raw[0]),
				C.size_t(len(raw)),
			)
		})
		if rc != 0 {
			return uploadFail("q6_k", name, rc)
		}

		gm = gpuQ8Matrix{
			ptr:   ptr,
			rows:  rows,
			cols:  cols,
			bytes: len(raw),
		}

		b.matricesQ6K[name] = gm
		b.trackWeight(kindQ6K, name, gm.bytes)
	}

	b.prepareVecUpload(vec)
	rc := C.gguf_cuda_matmul_vec_q6_k_device(
		&b.drv,
		b.ctx,
		b.fnQ6K,
		&b.matmulPool,
		gm.ptr,
		(*C.float)(unsafe.Pointer(&vec[0])),
		(*C.float)(unsafe.Pointer(&dst[0])),
		C.int(rows),
		C.int(cols),
	)
	if rc != 0 {
		return fmt.Errorf("cuda: matmul_vec_q6_k %q: code %d", name, int(rc))
	}

	return nil
}

func (b *Backend) MatMulVecQ4_1Cached(name string, raw []byte, rows, cols int, vec []float32) ([]float32, error) {
	out := make([]float32, rows)
	if err := b.MatMulVecQ4_1CachedInto(name, raw, rows, cols, vec, out); err != nil {
		return nil, err
	}

	return out, nil
}

func (b *Backend) MatMulVecQ4_1CachedInto(name string, raw []byte, rows, cols int, vec, dst []float32) error {
	if !b.hasQ41 {
		return fmt.Errorf("cuda: q4_1 matmul kernel unavailable")
	}

	if err := validateQ41MatMul(raw, rows, cols, vec); err != nil {
		return err
	}

	if len(dst) < rows {
		return fmt.Errorf("cuda: MatMulVecQ4_1CachedInto dst too short")
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	gm, ok := b.matricesQ41[name]
	if ok {
		b.touchWeight(kindQ41, name)
	}

	if !ok || gm.rows != rows || gm.cols != cols || gm.bytes != len(raw) {
		if ok {
			b.freeWeight(kindQ41, name)
		}

		var ptr C.CUdeviceptr
		rc := b.withEvict(func() C.int {
			return C.gguf_cuda_upload_q4_1(
				&b.drv,
				b.ctx,
				&ptr,
				unsafe.Pointer(&raw[0]),
				C.size_t(len(raw)),
			)
		})
		if rc != 0 {
			return uploadFail("q4_1", name, rc)
		}

		gm = gpuQ8Matrix{
			ptr:   ptr,
			rows:  rows,
			cols:  cols,
			bytes: len(raw),
		}
		b.matricesQ41[name] = gm
		b.trackWeight(kindQ41, name, gm.bytes)
	}

	b.prepareVecUpload(vec)
	rc := C.gguf_cuda_matmul_vec_q4_1_device(
		&b.drv,
		b.ctx,
		b.fnQ41,
		&b.matmulPool,
		gm.ptr,
		(*C.float)(unsafe.Pointer(&vec[0])),
		(*C.float)(unsafe.Pointer(&dst[0])),
		C.int(rows),
		C.int(cols),
	)
	if rc != 0 {
		return fmt.Errorf("cuda: matmul_vec_q4_1 %q: code %d", name, int(rc))
	}

	return nil
}

func (b *Backend) MatMulVecQ5_0Cached(name string, raw []byte, rows, cols int, vec []float32) ([]float32, error) {
	out := make([]float32, rows)
	if err := b.MatMulVecQ5_0CachedInto(name, raw, rows, cols, vec, out); err != nil {
		return nil, err
	}

	return out, nil
}

func (b *Backend) MatMulVecQ5_0CachedInto(name string, raw []byte, rows, cols int, vec, dst []float32) error {
	if !b.hasQ50 {
		return fmt.Errorf("cuda: q5_0 matmul kernel unavailable")
	}

	if err := validateQ50MatMul(raw, rows, cols, vec); err != nil {
		return err
	}

	if len(dst) < rows {
		return fmt.Errorf("cuda: MatMulVecQ5_0CachedInto dst too short")
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	gm, ok := b.matricesQ50[name]
	if ok {
		b.touchWeight(kindQ50, name)
	}

	if !ok || gm.rows != rows || gm.cols != cols || gm.bytes != len(raw) {
		if ok {
			b.freeWeight(kindQ50, name)
		}

		var ptr C.CUdeviceptr
		rc := b.withEvict(func() C.int {
			return C.gguf_cuda_upload_q5_0(
				&b.drv,
				b.ctx,
				&ptr,
				unsafe.Pointer(&raw[0]),
				C.size_t(len(raw)),
			)
		})
		if rc != 0 {
			return uploadFail("q5_0", name, rc)
		}

		gm = gpuQ8Matrix{
			ptr:   ptr,
			rows:  rows,
			cols:  cols,
			bytes: len(raw),
		}
		b.matricesQ50[name] = gm
		b.trackWeight(kindQ50, name, gm.bytes)
	}

	b.prepareVecUpload(vec)
	rc := C.gguf_cuda_matmul_vec_q5_0_device(
		&b.drv,
		b.ctx,
		b.fnQ50,
		&b.matmulPool,
		gm.ptr,
		(*C.float)(unsafe.Pointer(&vec[0])),
		(*C.float)(unsafe.Pointer(&dst[0])),
		C.int(rows),
		C.int(cols),
	)
	if rc != 0 {
		return fmt.Errorf("cuda: matmul_vec_q5_0 %q: code %d", name, int(rc))
	}

	return nil
}

func (b *Backend) MatMulVecQ5_1Cached(name string, raw []byte, rows, cols int, vec []float32) ([]float32, error) {
	out := make([]float32, rows)
	if err := b.MatMulVecQ5_1CachedInto(name, raw, rows, cols, vec, out); err != nil {
		return nil, err
	}

	return out, nil
}

func (b *Backend) MatMulVecQ5_1CachedInto(name string, raw []byte, rows, cols int, vec, dst []float32) error {
	if !b.hasQ51 {
		return fmt.Errorf("cuda: q5_1 matmul kernel unavailable")
	}

	if err := validateQ51MatMul(raw, rows, cols, vec); err != nil {
		return err
	}

	if len(dst) < rows {
		return fmt.Errorf("cuda: MatMulVecQ5_1CachedInto dst too short")
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	gm, ok := b.matricesQ51[name]
	if ok {
		b.touchWeight(kindQ51, name)
	}

	if !ok || gm.rows != rows || gm.cols != cols || gm.bytes != len(raw) {
		if ok {
			b.freeWeight(kindQ51, name)
		}

		var ptr C.CUdeviceptr
		rc := b.withEvict(func() C.int {
			return C.gguf_cuda_upload_q5_1(
				&b.drv,
				b.ctx,
				&ptr,
				unsafe.Pointer(&raw[0]),
				C.size_t(len(raw)),
			)
		})
		if rc != 0 {
			return uploadFail("q5_1", name, rc)
		}

		gm = gpuQ8Matrix{
			ptr:   ptr,
			rows:  rows,
			cols:  cols,
			bytes: len(raw),
		}
		b.matricesQ51[name] = gm
		b.trackWeight(kindQ51, name, gm.bytes)
	}

	b.prepareVecUpload(vec)
	rc := C.gguf_cuda_matmul_vec_q5_1_device(
		&b.drv,
		b.ctx,
		b.fnQ51,
		&b.matmulPool,
		gm.ptr,
		(*C.float)(unsafe.Pointer(&vec[0])),
		(*C.float)(unsafe.Pointer(&dst[0])),
		C.int(rows),
		C.int(cols),
	)
	if rc != 0 {
		return fmt.Errorf("cuda: matmul_vec_q5_1 %q: code %d", name, int(rc))
	}

	return nil
}

func (b *Backend) MatMulVecQ2_KCached(name string, raw []byte, rows, cols int, vec []float32) ([]float32, error) {
	out := make([]float32, rows)
	if err := b.MatMulVecQ2_KCachedInto(name, raw, rows, cols, vec, out); err != nil {
		return nil, err
	}

	return out, nil
}

func (b *Backend) MatMulVecQ2_KCachedInto(name string, raw []byte, rows, cols int, vec, dst []float32) error {
	if !b.hasQ2K {
		return fmt.Errorf("cuda: q2_k matmul kernel unavailable")
	}

	if err := validateQ2KMatMul(raw, rows, cols, vec); err != nil {
		return err
	}

	if len(dst) < rows {
		return fmt.Errorf("cuda: MatMulVecQ2_KCachedInto dst too short")
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	gm, ok := b.matricesQ2K[name]
	if ok {
		b.touchWeight(kindQ2K, name)
	}

	if !ok || gm.rows != rows || gm.cols != cols || gm.bytes != len(raw) {
		if ok {
			b.freeWeight(kindQ2K, name)
		}

		var ptr C.CUdeviceptr
		rc := b.withEvict(func() C.int {
			return C.gguf_cuda_upload_q2_k(
				&b.drv,
				b.ctx,
				&ptr,
				unsafe.Pointer(&raw[0]),
				C.size_t(len(raw)),
			)
		})
		if rc != 0 {
			return uploadFail("q2_k", name, rc)
		}

		gm = gpuQ8Matrix{
			ptr:   ptr,
			rows:  rows,
			cols:  cols,
			bytes: len(raw),
		}
		b.matricesQ2K[name] = gm
		b.trackWeight(kindQ2K, name, gm.bytes)
	}

	b.prepareVecUpload(vec)
	rc := C.gguf_cuda_matmul_vec_q2_k_device(
		&b.drv,
		b.ctx,
		b.fnQ2K,
		&b.matmulPool,
		gm.ptr,
		(*C.float)(unsafe.Pointer(&vec[0])),
		(*C.float)(unsafe.Pointer(&dst[0])),
		C.int(rows),
		C.int(cols),
	)
	if rc != 0 {
		return fmt.Errorf("cuda: matmul_vec_q2_k %q: code %d", name, int(rc))
	}

	return nil
}

func (b *Backend) MatMulVecQ3_KCached(name string, raw []byte, rows, cols int, vec []float32) ([]float32, error) {
	out := make([]float32, rows)
	if err := b.MatMulVecQ3_KCachedInto(name, raw, rows, cols, vec, out); err != nil {
		return nil, err
	}

	return out, nil
}

func (b *Backend) MatMulVecQ3_KCachedInto(name string, raw []byte, rows, cols int, vec, dst []float32) error {
	if !b.hasQ3K {
		return fmt.Errorf("cuda: q3_k matmul kernel unavailable")
	}

	if err := validateQ3KMatMul(raw, rows, cols, vec); err != nil {
		return err
	}

	if len(dst) < rows {
		return fmt.Errorf("cuda: MatMulVecQ3_KCachedInto dst too short")
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	gm, ok := b.matricesQ3K[name]
	if ok {
		b.touchWeight(kindQ3K, name)
	}

	if !ok || gm.rows != rows || gm.cols != cols || gm.bytes != len(raw) {
		if ok {
			b.freeWeight(kindQ3K, name)
		}

		var ptr C.CUdeviceptr
		rc := b.withEvict(func() C.int {
			return C.gguf_cuda_upload_q3_k(
				&b.drv,
				b.ctx,
				&ptr,
				unsafe.Pointer(&raw[0]),
				C.size_t(len(raw)),
			)
		})
		if rc != 0 {
			return uploadFail("q3_k", name, rc)
		}

		gm = gpuQ8Matrix{
			ptr:   ptr,
			rows:  rows,
			cols:  cols,
			bytes: len(raw),
		}
		b.matricesQ3K[name] = gm
		b.trackWeight(kindQ3K, name, gm.bytes)
	}

	b.prepareVecUpload(vec)
	rc := C.gguf_cuda_matmul_vec_q3_k_device(
		&b.drv,
		b.ctx,
		b.fnQ3K,
		&b.matmulPool,
		gm.ptr,
		(*C.float)(unsafe.Pointer(&vec[0])),
		(*C.float)(unsafe.Pointer(&dst[0])),
		C.int(rows),
		C.int(cols),
	)
	if rc != 0 {
		return fmt.Errorf("cuda: matmul_vec_q3_k %q: code %d", name, int(rc))
	}

	return nil
}

func (b *Backend) MatMulVecQ8_KCached(name string, raw []byte, rows, cols int, vec []float32) ([]float32, error) {
	out := make([]float32, rows)
	if err := b.MatMulVecQ8_KCachedInto(name, raw, rows, cols, vec, out); err != nil {
		return nil, err
	}

	return out, nil
}

func (b *Backend) MatMulVecQ8_KCachedInto(name string, raw []byte, rows, cols int, vec, dst []float32) error {
	if !b.hasQ8K {
		return fmt.Errorf("cuda: q8_k matmul kernel unavailable")
	}

	if err := validateQ8KMatMul(raw, rows, cols, vec); err != nil {
		return err
	}

	if len(dst) < rows {
		return fmt.Errorf("cuda: MatMulVecQ8_KCachedInto dst too short")
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	gm, ok := b.matricesQ8K[name]
	if ok {
		b.touchWeight(kindQ8K, name)
	}

	if !ok || gm.rows != rows || gm.cols != cols || gm.bytes != len(raw) {
		if ok {
			b.freeWeight(kindQ8K, name)
		}

		var ptr C.CUdeviceptr
		rc := b.withEvict(func() C.int {
			return C.gguf_cuda_upload_q8_k(
				&b.drv,
				b.ctx,
				&ptr,
				unsafe.Pointer(&raw[0]),
				C.size_t(len(raw)),
			)
		})
		if rc != 0 {
			return uploadFail("q8_k", name, rc)
		}

		gm = gpuQ8Matrix{
			ptr:   ptr,
			rows:  rows,
			cols:  cols,
			bytes: len(raw),
		}
		b.matricesQ8K[name] = gm
		b.trackWeight(kindQ8K, name, gm.bytes)
	}

	b.prepareVecUpload(vec)
	rc := C.gguf_cuda_matmul_vec_q8_k_device(
		&b.drv,
		b.ctx,
		b.fnQ8K,
		&b.matmulPool,
		gm.ptr,
		(*C.float)(unsafe.Pointer(&vec[0])),
		(*C.float)(unsafe.Pointer(&dst[0])),
		C.int(rows),
		C.int(cols),
	)
	if rc != 0 {
		return fmt.Errorf("cuda: matmul_vec_q8_k %q: code %d", name, int(rc))
	}

	return nil
}

// ropeModeNeoX / ropeModeNorm - graph cache key for selected RoPE kernel
const (
	ropeModeNeoX = 0
	ropeModeNorm = 1
)

// ropeKernel returns RoPE kernel and its graph cache key
func (b *Backend) ropeKernel(mode compute.RoPEMode) (C.CUfunction, int, error) {
	switch mode {
	case compute.RoPENorm:
		if !b.hasRoPENorm {
			return nil, 0, fmt.Errorf("cuda: rope_heads_norm kernel unavailable")
		}

		return b.fnRoPENorm, ropeModeNorm, nil
	default:
		return b.fnRoPE, ropeModeNeoX, nil
	}
}

// ensureHeadNorm caches QK-norm weights; empty name - layer without QK-norm (null pointer)
func (b *Backend) ensureHeadNorm(name string, norm []float32, headDim int) (C.CUdeviceptr, error) {
	if name == "" || len(norm) == 0 {
		return 0, nil
	}

	if len(norm) < headDim {
		return 0, fmt.Errorf("cuda: %q: len(norm)=%d, head_dim=%d", name, len(norm), headDim)
	}

	m, err := b.ensureFP32Matrix(name, norm, headDim, 1)
	if err != nil {
		return 0, err
	}

	return m.ptr, nil
}

func (b *Backend) ensureFP32Matrix(name string, matrix []float32, rows, cols int) (gpuMatrix, error) {
	gm, ok := b.matrices[name]
	if ok {
		b.touchWeight(kindF32, name)
	}
	if ok && gm.rows == rows && gm.cols == cols {
		return gm, nil
	}

	if ok {
		b.freeWeight(kindF32, name)
	}

	var ptr C.CUdeviceptr
	rc := b.withEvict(func() C.int {
		return C.gguf_cuda_upload_matrix(
			&b.drv,
			b.ctx,
			&ptr,
			(*C.float)(unsafe.Pointer(&matrix[0])),
			C.int(rows),
			C.int(cols),
		)
	})
	if rc != 0 {
		return gpuMatrix{}, uploadFail("matrix", name, rc)
	}

	gm = gpuMatrix{
		ptr:  ptr,
		rows: rows,
		cols: cols,
	}
	b.matrices[name] = gm
	b.trackWeight(kindF32, name, int(gm.rows)*int(gm.cols)*4)

	return gm, nil
}

// quantMatrices returns weight cache and uploader for type t
func (b *Backend) quantCache(t dtype.DType) (map[string]gpuQ8Matrix, C.CUfunction, error) {
	switch t {
	case dtype.Q8_0:
		return b.matricesQ8, b.fnQ8, nil
	case dtype.Q4_0:
		if !b.hasQ4 {
			return nil, nil, fmt.Errorf("cuda: q4_0 kernel unavailable")
		}
		return b.matricesQ4, b.fnQ4, nil
	case dtype.Q4_K:
		if !b.hasQ4K {
			return nil, nil, fmt.Errorf("cuda: q4_k kernel unavailable")
		}
		return b.matricesQ4K, b.fnQ4K, nil
	case dtype.Q5_K:
		if !b.hasQ5K {
			return nil, nil, fmt.Errorf("cuda: q5_k kernel unavailable")
		}
		return b.matricesQ5K, b.fnQ5K, nil
	case dtype.Q6_K:
		if !b.hasQ6K {
			return nil, nil, fmt.Errorf("cuda: q6_k kernel unavailable")
		}
		return b.matricesQ6K, b.fnQ6K, nil
	case dtype.Q4_1:
		if !b.hasQ41 {
			return nil, nil, fmt.Errorf("cuda: q4_1 kernel unavailable")
		}
		return b.matricesQ41, b.fnQ41, nil
	case dtype.Q5_0:
		if !b.hasQ50 {
			return nil, nil, fmt.Errorf("cuda: q5_0 kernel unavailable")
		}
		return b.matricesQ50, b.fnQ50, nil
	case dtype.Q5_1:
		if !b.hasQ51 {
			return nil, nil, fmt.Errorf("cuda: q5_1 kernel unavailable")
		}
		return b.matricesQ51, b.fnQ51, nil
	case dtype.Q2_K:
		if !b.hasQ2K {
			return nil, nil, fmt.Errorf("cuda: q2_k kernel unavailable")
		}
		return b.matricesQ2K, b.fnQ2K, nil
	case dtype.Q3_K:
		if !b.hasQ3K {
			return nil, nil, fmt.Errorf("cuda: q3_k kernel unavailable")
		}
		return b.matricesQ3K, b.fnQ3K, nil
	case dtype.Q8_K:
		if !b.hasQ8K {
			return nil, nil, fmt.Errorf("cuda: q8_k kernel unavailable")
		}
		return b.matricesQ8K, b.fnQ8K, nil
	default:
		return nil, nil, fmt.Errorf("cuda: type %s not supported on fused path", t)
	}
}

func (b *Backend) uploadQuant(t dtype.DType, ptr *C.CUdeviceptr, raw []byte) C.int {
	switch t {
	case dtype.Q8_0:
		return C.gguf_cuda_upload_q8_0(&b.drv, b.ctx, ptr, unsafe.Pointer(&raw[0]), C.size_t(len(raw)))
	case dtype.Q4_0:
		return C.gguf_cuda_upload_q4_0(&b.drv, b.ctx, ptr, unsafe.Pointer(&raw[0]), C.size_t(len(raw)))
	case dtype.Q4_K:
		return C.gguf_cuda_upload_q4_k(&b.drv, b.ctx, ptr, unsafe.Pointer(&raw[0]), C.size_t(len(raw)))
	case dtype.Q5_K:
		return C.gguf_cuda_upload_q5_k(&b.drv, b.ctx, ptr, unsafe.Pointer(&raw[0]), C.size_t(len(raw)))
	case dtype.Q6_K:
		return C.gguf_cuda_upload_q6_k(&b.drv, b.ctx, ptr, unsafe.Pointer(&raw[0]), C.size_t(len(raw)))
	case dtype.Q4_1:
		return C.gguf_cuda_upload_q4_1(&b.drv, b.ctx, ptr, unsafe.Pointer(&raw[0]), C.size_t(len(raw)))
	case dtype.Q5_0:
		return C.gguf_cuda_upload_q5_0(&b.drv, b.ctx, ptr, unsafe.Pointer(&raw[0]), C.size_t(len(raw)))
	case dtype.Q5_1:
		return C.gguf_cuda_upload_q5_1(&b.drv, b.ctx, ptr, unsafe.Pointer(&raw[0]), C.size_t(len(raw)))
	case dtype.Q2_K:
		return C.gguf_cuda_upload_q2_k(&b.drv, b.ctx, ptr, unsafe.Pointer(&raw[0]), C.size_t(len(raw)))
	case dtype.Q3_K:
		return C.gguf_cuda_upload_q3_k(&b.drv, b.ctx, ptr, unsafe.Pointer(&raw[0]), C.size_t(len(raw)))
	case dtype.Q8_K:
		return C.gguf_cuda_upload_q8_k(&b.drv, b.ctx, ptr, unsafe.Pointer(&raw[0]), C.size_t(len(raw)))
	default:
		return -1
	}
}

func validateQuantMatMul(t dtype.DType, raw []byte, rows, cols int, vec []float32) error {
	switch t {
	case dtype.Q8_0:
		return validateQ8MatMul(raw, rows, cols, vec)
	case dtype.Q4_0:
		return validateQ4MatMul(raw, rows, cols, vec)
	case dtype.Q4_K:
		return validateQ4KMatMul(raw, rows, cols, vec)
	case dtype.Q5_K:
		return validateQ5KMatMul(raw, rows, cols, vec)
	case dtype.Q6_K:
		return validateQ6KMatMul(raw, rows, cols, vec)
	case dtype.Q4_1:
		return validateQ41MatMul(raw, rows, cols, vec)
	case dtype.Q5_0:
		return validateQ50MatMul(raw, rows, cols, vec)
	case dtype.Q5_1:
		return validateQ51MatMul(raw, rows, cols, vec)
	case dtype.Q2_K:
		return validateQ2KMatMul(raw, rows, cols, vec)
	case dtype.Q3_K:
		return validateQ3KMatMul(raw, rows, cols, vec)
	case dtype.Q8_K:
		return validateQ8KMatMul(raw, rows, cols, vec)
	default:
		return fmt.Errorf("cuda: type %s not supported on fused path", t)
	}
}

// ensureQuantMatrix caches quantized weights on device (Q8_0/Q4_0/Q4_K/Q5_K/Q6_K)
func (b *Backend) ensureQuantMatrix(t dtype.DType, name string, raw []byte, rows, cols int) (gpuQ8Matrix, error) {
	cache, _, err := b.quantCache(t)
	if err != nil {
		return gpuQ8Matrix{}, err
	}

	gm, ok := cache[name]
	if ok {
		b.touchWeight(quantKind(t), name)
	}
	if ok && gm.rows == rows && gm.cols == cols && gm.bytes == len(raw) {
		return gm, nil
	}

	if ok {
		b.freeWeight(quantKind(t), name)
	}

	if len(raw) == 0 {
		return gpuQ8Matrix{}, fmt.Errorf("cuda: %q: empty weights", name)
	}

	var ptr C.CUdeviceptr
	if rc := b.withEvict(func() C.int {
		return b.uploadQuant(t, &ptr, raw)
	}); rc != 0 {
		return gpuQ8Matrix{}, uploadFail(t.String(), name, rc)
	}

	gm = gpuQ8Matrix{
		ptr:   ptr,
		rows:  rows,
		cols:  cols,
		bytes: len(raw),
	}
	cache[name] = gm
	b.trackWeight(quantKind(t), name, gm.bytes)

	return gm, nil
}

func (b *Backend) ensureQ8Matrix(name string, raw []byte, rows, cols int) (gpuQ8Matrix, error) {
	gm, ok := b.matricesQ8[name]
	if ok {
		b.touchWeight(kindQ8, name)
	}
	if ok && gm.rows == rows && gm.cols == cols && gm.bytes == len(raw) {
		return gm, nil
	}

	if ok {
		b.freeWeight(kindQ8, name)
	}

	var ptr C.CUdeviceptr
	rc := b.withEvict(func() C.int {
		return C.gguf_cuda_upload_q8_0(
			&b.drv,
			b.ctx,
			&ptr,
			unsafe.Pointer(&raw[0]),
			C.size_t(len(raw)),
		)
	})
	if rc != 0 {
		return gpuQ8Matrix{}, uploadFail("q8_0", name, rc)
	}

	gm = gpuQ8Matrix{
		ptr:   ptr,
		rows:  rows,
		cols:  cols,
		bytes: len(raw),
	}
	b.matricesQ8[name] = gm
	b.trackWeight(kindQ8, name, gm.bytes)

	return gm, nil
}

func (b *Backend) FFNSwiGLUCached(gateName, upName, downName string, gateW, upW, downW, x, out []float32, embd, ffn int) error {
	if !b.hasSwiGLU {
		return fmt.Errorf("cuda: SwiGLU kernel unavailable")
	}

	if len(x) < embd || len(out) < embd {
		return fmt.Errorf("cuda: FFNSwiGLUCached: x/out too short")
	}

	if len(gateW) < ffn*embd || len(upW) < ffn*embd || len(downW) < embd*ffn {
		return fmt.Errorf("cuda: FFNSwiGLUCached: weights too short")
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	gateM, err := b.ensureFP32Matrix(gateName, gateW, ffn, embd)
	if err != nil {
		return err
	}

	upM, err := b.ensureFP32Matrix(upName, upW, ffn, embd)
	if err != nil {
		return err
	}

	downM, err := b.ensureFP32Matrix(downName, downW, embd, ffn)
	if err != nil {
		return err
	}

	b.prepareVecUpload(x)
	rc := C.gguf_cuda_ffn_swiglu_device(
		&b.drv,
		b.ctx,
		b.fn,
		b.fnSwiGLU,
		&b.matmulPool,
		gateM.ptr,
		upM.ptr,
		downM.ptr,
		(*C.float)(unsafe.Pointer(&x[0])),
		(*C.float)(unsafe.Pointer(&out[0])),
		C.int(embd),
		C.int(ffn),
	)
	b.lastVecAddr = 0
	b.lastVecLen = 0
	if rc != 0 {
		return fmt.Errorf("cuda: ffn_swiglu: code %d", int(rc))
	}

	return nil
}

func (b *Backend) FFNSwiGLUQ8_0Cached(gateName, upName, downName string, gateRaw, upRaw, downRaw []byte, x, out []float32, embd, ffn int) error {
	if !b.hasSwiGLU {
		return fmt.Errorf("cuda: SwiGLU kernel unavailable")
	}

	if len(x) < embd || len(out) < embd {
		return fmt.Errorf("cuda: FFNSwiGLUQ8_0Cached: x/out too short")
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	gateM, err := b.ensureQ8Matrix(gateName, gateRaw, ffn, embd)
	if err != nil {
		return err
	}

	upM, err := b.ensureQ8Matrix(upName, upRaw, ffn, embd)
	if err != nil {
		return err
	}

	downM, err := b.ensureQ8Matrix(downName, downRaw, embd, ffn)
	if err != nil {
		return err
	}

	b.prepareVecUpload(x)
	rc := C.gguf_cuda_ffn_swiglu_device(
		&b.drv,
		b.ctx,
		b.fnQ8,
		b.fnSwiGLU,
		&b.matmulPool,
		gateM.ptr,
		upM.ptr,
		downM.ptr,
		(*C.float)(unsafe.Pointer(&x[0])),
		(*C.float)(unsafe.Pointer(&out[0])),
		C.int(embd),
		C.int(ffn),
	)
	b.lastVecAddr = 0
	b.lastVecLen = 0
	if rc != 0 {
		return fmt.Errorf("cuda: ffn_swiglu q8: code %d", int(rc))
	}

	return nil
}

func (b *Backend) AttnFFNResidualCached(woName, ffnNormName, gateName, upName, downName string, woW, ffnNorm, gateW, upW, downW, x, attn []float32, embd, attnDim, ffn int, eps float32) error {
	if !b.hasSwiGLU || !b.hasRMS || !b.hasAdd {
		return fmt.Errorf("cuda: attn+ffn residual kernels unavailable")
	}

	if len(x) < embd || len(attn) < attnDim || len(ffnNorm) < embd {
		return fmt.Errorf("cuda: AttnFFNResidualCached: buffers too short")
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	woM, err := b.ensureFP32Matrix(woName, woW, embd, attnDim)
	if err != nil {
		return err
	}

	normM, err := b.ensureFP32Matrix(ffnNormName, ffnNorm, embd, 1)
	if err != nil {
		return err
	}

	gateM, err := b.ensureFP32Matrix(gateName, gateW, ffn, embd)
	if err != nil {
		return err
	}

	upM, err := b.ensureFP32Matrix(upName, upW, ffn, embd)
	if err != nil {
		return err
	}

	downM, err := b.ensureFP32Matrix(downName, downW, embd, ffn)
	if err != nil {
		return err
	}

	// Host buffer path: clear residency (if any), otherwise x would be ignored
	b.clearResidencyLocked()

	rc := C.gguf_cuda_attn_ffn_residual_device(
		&b.drv,
		b.ctx,
		b.fn,
		b.fnRMS,
		b.fnSwiGLU,
		b.fnAdd,
		&b.matmulPool,
		woM.ptr,
		normM.ptr,
		gateM.ptr,
		upM.ptr,
		downM.ptr,
		(*C.float)(unsafe.Pointer(&x[0])),
		(*C.float)(unsafe.Pointer(&attn[0])),
		(*C.float)(unsafe.Pointer(&x[0])),
		C.int(embd),
		C.int(attnDim),
		C.int(ffn),
		C.float(eps),
		nil,
	)
	b.lastVecAddr = 0
	b.lastVecLen = 0

	if rc != 0 {
		return fmt.Errorf("cuda: attn_ffn_residual: code %d", int(rc))
	}

	return nil
}

func (b *Backend) AttnFFNResidualQ8_0Cached(woName, ffnNormName, gateName, upName, downName string, woRaw, gateRaw, upRaw, downRaw []byte, ffnNorm, x, attn []float32, embd, attnDim, ffn int, eps float32) error {
	if !b.hasSwiGLU || !b.hasRMS || !b.hasAdd {
		return fmt.Errorf("cuda: attn+ffn residual kernels unavailable")
	}

	if len(x) < embd || len(attn) < attnDim || len(ffnNorm) < embd {
		return fmt.Errorf("cuda: AttnFFNResidualQ8_0Cached: buffers too short")
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	woM, err := b.ensureQ8Matrix(woName, woRaw, embd, attnDim)
	if err != nil {
		return err
	}

	normM, err := b.ensureFP32Matrix(ffnNormName, ffnNorm, embd, 1)
	if err != nil {
		return err
	}

	gateM, err := b.ensureQ8Matrix(gateName, gateRaw, ffn, embd)
	if err != nil {
		return err
	}

	upM, err := b.ensureQ8Matrix(upName, upRaw, ffn, embd)
	if err != nil {
		return err
	}

	downM, err := b.ensureQ8Matrix(downName, downRaw, embd, ffn)
	if err != nil {
		return err
	}

	// Host buffer path: clear residency (if any)
	b.clearResidencyLocked()

	rc := C.gguf_cuda_attn_ffn_residual_device(
		&b.drv,
		b.ctx,
		b.fnQ8,
		b.fnRMS,
		b.fnSwiGLU,
		b.fnAdd,
		&b.matmulPool,
		woM.ptr,
		normM.ptr,
		gateM.ptr,
		upM.ptr, downM.ptr,
		(*C.float)(unsafe.Pointer(&x[0])),
		(*C.float)(unsafe.Pointer(&attn[0])),
		(*C.float)(unsafe.Pointer(&x[0])),
		C.int(embd),
		C.int(attnDim),
		C.int(ffn),
		C.float(eps),
		nil,
	)
	b.lastVecAddr = 0
	b.lastVecLen = 0
	if rc != 0 {
		return fmt.Errorf("cuda: attn_ffn_residual q8: code %d", int(rc))
	}

	return nil
}

func (b *Backend) QKVRoPEAttentionCached(qName, kName, vName, qNormName, kNormName string, qW, kW, vW, qNorm, kNorm, h, cos, sin, attn, kOut, vOut []float32, embd, nHeads, nKVHeads, headDim, layer, kvPos, seqLen int, eps float32) error {
	if !b.hasAttn || !b.hasRoPE || !b.hasRMS {
		return fmt.Errorf("cuda: qkv+rope+attn kernels unavailable")
	}

	if len(h) < embd || len(cos) < headDim/2 || len(sin) < headDim/2 || len(attn) < nHeads*headDim || len(kOut) < nKVHeads*headDim || len(vOut) < nKVHeads*headDim {
		return fmt.Errorf("cuda: QKVRoPEAttentionCached: buffers too short")
	}

	b.mu.Lock()
	defer b.mu.Unlock()
	if !b.kvReady {
		return fmt.Errorf("cuda: kv cache not initialized")
	}

	qM, err := b.ensureFP32Matrix(qName, qW, nHeads*headDim, embd)
	if err != nil {
		return err
	}

	kM, err := b.ensureFP32Matrix(kName, kW, nKVHeads*headDim, embd)
	if err != nil {
		return err
	}

	vM, err := b.ensureFP32Matrix(vName, vW, nKVHeads*headDim, embd)
	if err != nil {
		return err
	}

	qNormM, err := b.ensureFP32Matrix(qNormName, qNorm, headDim, 1)
	if err != nil {
		return err
	}

	kNormM, err := b.ensureFP32Matrix(kNormName, kNorm, headDim, 1)
	if err != nil {
		return err
	}

	fnSM := b.fnSoftmax
	if !b.hasSoftmax {
		fnSM = nil
	}

	rc := C.gguf_cuda_qkv_rope_attn_device(
		&b.drv,
		b.ctx,
		b.fn,
		b.fnRMS,
		b.fnRoPE,
		b.fnAttnQK,
		b.fnAttnV,
		fnSM,
		&b.matmulPool,
		&b.attnPool,
		&b.kvCache,
		qM.ptr,
		kM.ptr,
		vM.ptr,
		qNormM.ptr,
		kNormM.ptr,
		0,
		(*C.float)(unsafe.Pointer(&h[0])),
		(*C.float)(unsafe.Pointer(&cos[0])),
		(*C.float)(unsafe.Pointer(&sin[0])),
		(*C.float)(unsafe.Pointer(&attn[0])),
		(*C.float)(unsafe.Pointer(&kOut[0])),
		(*C.float)(unsafe.Pointer(&vOut[0])),
		C.int(embd),
		C.int(nHeads),
		C.int(nKVHeads),
		C.int(headDim),
		C.int(layer),
		C.int(kvPos),
		C.int(seqLen),
		C.int(ropeModeNeoX),
		C.float(eps),
	)
	b.lastVecAddr = 0
	b.lastVecLen = 0
	if rc != 0 {
		return fmt.Errorf("cuda: qkv_rope_attn: code %d", int(rc))
	}

	return nil
}

func (b *Backend) QKVRoPEAttentionQ8_0Cached(qName, kName, vName, qNormName, kNormName string, qRaw, kRaw, vRaw []byte, qNorm, kNorm, h, cos, sin, attn, kOut, vOut []float32, embd, nHeads, nKVHeads, headDim, layer, kvPos, seqLen int, eps float32) error {
	if !b.hasAttn || !b.hasRoPE || !b.hasRMS {
		return fmt.Errorf("cuda: qkv+rope+attn kernels unavailable")
	}

	if len(h) < embd || len(cos) < headDim/2 || len(sin) < headDim/2 || len(attn) < nHeads*headDim || len(kOut) < nKVHeads*headDim || len(vOut) < nKVHeads*headDim {
		return fmt.Errorf("cuda: QKVRoPEAttentionQ8_0Cached: buffers too short")
	}

	b.mu.Lock()
	defer b.mu.Unlock()
	if !b.kvReady {
		return fmt.Errorf("cuda: kv cache not initialized")
	}

	qM, err := b.ensureQ8Matrix(qName, qRaw, nHeads*headDim, embd)
	if err != nil {
		return err
	}

	kM, err := b.ensureQ8Matrix(kName, kRaw, nKVHeads*headDim, embd)
	if err != nil {
		return err
	}

	vM, err := b.ensureQ8Matrix(vName, vRaw, nKVHeads*headDim, embd)
	if err != nil {
		return err
	}

	qNormM, err := b.ensureFP32Matrix(qNormName, qNorm, headDim, 1)
	if err != nil {
		return err
	}

	kNormM, err := b.ensureFP32Matrix(kNormName, kNorm, headDim, 1)
	if err != nil {
		return err
	}

	fnSM := b.fnSoftmax
	if !b.hasSoftmax {
		fnSM = nil
	}

	rc := C.gguf_cuda_qkv_rope_attn_device(
		&b.drv,
		b.ctx,
		b.fnQ8,
		b.fnRMS,
		b.fnRoPE,
		b.fnAttnQK,
		b.fnAttnV,
		fnSM,
		&b.matmulPool,
		&b.attnPool,
		&b.kvCache,
		qM.ptr,
		kM.ptr,
		vM.ptr,
		qNormM.ptr,
		kNormM.ptr,
		0,
		(*C.float)(unsafe.Pointer(&h[0])),
		(*C.float)(unsafe.Pointer(&cos[0])),
		(*C.float)(unsafe.Pointer(&sin[0])),
		(*C.float)(unsafe.Pointer(&attn[0])),
		(*C.float)(unsafe.Pointer(&kOut[0])),
		(*C.float)(unsafe.Pointer(&vOut[0])),
		C.int(embd),
		C.int(nHeads),
		C.int(nKVHeads),
		C.int(headDim),
		C.int(layer),
		C.int(kvPos),
		C.int(seqLen),
		C.int(ropeModeNeoX),
		C.float(eps),
	)
	b.lastVecAddr = 0
	b.lastVecLen = 0
	if rc != 0 {
		return fmt.Errorf("cuda: qkv_rope_attn q8: code %d", int(rc))
	}

	return nil
}

// FFNSwiGLUQuantCached: FFN SwiGLU natively for Q8_0/Q4_0/Q4_K/Q5_K/Q6_K (no host Floats)
func (b *Backend) FFNSwiGLUQuantCached(t dtype.DType, gateName, upName, downName string, gateRaw, upRaw, downRaw []byte, x, out []float32, embd, ffn int) error {
	if !b.hasSwiGLU {
		return fmt.Errorf("cuda: SwiGLU kernel unavailable")
	}

	return b.ffnGatedQuantCached(b.fnSwiGLU, "swiglu", t, gateName, upName, downName, gateRaw, upRaw, downRaw, x, out, embd, ffn)
}

// FFNGeGLUQuantCached: FFN GeGLU (Gemma) natively for the same quants
func (b *Backend) FFNGeGLUQuantCached(t dtype.DType, gateName, upName, downName string, gateRaw, upRaw, downRaw []byte, x, out []float32, embd, ffn int) error {
	if !b.hasGeGLU {
		return fmt.Errorf("cuda: GeGLU kernel unavailable")
	}

	return b.ffnGatedQuantCached(b.fnGeGLU, "geglu", t, gateName, upName, downName, gateRaw, upRaw, downRaw, x, out, embd, ffn)
}

// ffnGatedQuantCached - gate/up matmul + fnAct activation + down on device
func (b *Backend) ffnGatedQuantCached(fnAct C.CUfunction, act string, t dtype.DType, gateName, upName, downName string, gateRaw, upRaw, downRaw []byte, x, out []float32, embd, ffn int) error {
	if len(x) < embd || len(out) < embd {
		return fmt.Errorf("cuda: ffn %s: x/out too short", act)
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	_, fnMatmul, err := b.quantCache(t)
	if err != nil {
		return err
	}

	gateM, err := b.ensureQuantMatrix(t, gateName, gateRaw, ffn, embd)
	if err != nil {
		return err
	}

	upM, err := b.ensureQuantMatrix(t, upName, upRaw, ffn, embd)
	if err != nil {
		return err
	}

	downM, err := b.ensureQuantMatrix(t, downName, downRaw, embd, ffn)
	if err != nil {
		return err
	}

	b.prepareVecUpload(x)
	rc := C.gguf_cuda_ffn_swiglu_device(
		&b.drv,
		b.ctx,
		fnMatmul,
		fnAct,
		&b.matmulPool,
		gateM.ptr,
		upM.ptr,
		downM.ptr,
		(*C.float)(unsafe.Pointer(&x[0])),
		(*C.float)(unsafe.Pointer(&out[0])),
		C.int(embd),
		C.int(ffn),
	)
	b.lastVecAddr = 0
	b.lastVecLen = 0
	if rc != 0 {
		return fmt.Errorf("cuda: ffn_%s %s: code %d", act, t, int(rc))
	}

	return nil
}

// AttnFFNResidualQuantCached: WO+residual+RMSNorm+FFN+residual natively for quants.
// With active residency x not uploaded to GPU and result stays in d_resid.
func (b *Backend) AttnFFNResidualQuantCached(t dtype.DType, woName, ffnNormName, gateName, upName, downName string, woRaw, gateRaw, upRaw, downRaw []byte, ffnNorm, x, attn []float32, embd, attnDim, ffn int, eps float32) error {
	if !b.hasSwiGLU || !b.hasRMS || !b.hasAdd {
		return fmt.Errorf("cuda: attn+ffn residual kernels unavailable")
	}

	if len(ffnNorm) < embd {
		return fmt.Errorf("cuda: AttnFFNResidualQuantCached: ffn_norm too short")
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	// x == nil - explicit residency request: read and keep residual on device.
	// Otherwise source of truth is host buffer x, and residency is cleared.
	resident := b.hiddenResident && len(x) == 0
	if !resident {
		if len(x) < embd {
			return fmt.Errorf("cuda: AttnFFNResidualQuantCached: x too short")
		}

		b.clearResidencyLocked()
	}

	_, fnMatmul, err := b.quantCache(t)
	if err != nil {
		return err
	}

	woM, err := b.ensureQuantMatrix(t, woName, woRaw, embd, attnDim)
	if err != nil {
		return err
	}

	normM, err := b.ensureFP32Matrix(ffnNormName, ffnNorm, embd, 1)
	if err != nil {
		return err
	}

	gateM, err := b.ensureQuantMatrix(t, gateName, gateRaw, ffn, embd)
	if err != nil {
		return err
	}

	upM, err := b.ensureQuantMatrix(t, upName, upRaw, ffn, embd)
	if err != nil {
		return err
	}

	downM, err := b.ensureQuantMatrix(t, downName, downRaw, embd, ffn)
	if err != nil {
		return err
	}

	var xPtr *C.float
	if len(x) >= embd {
		xPtr = (*C.float)(unsafe.Pointer(&x[0]))
	}

	var attnPtr *C.float
	if len(attn) > 0 {
		attnPtr = (*C.float)(unsafe.Pointer(&attn[0]))
	}

	var dirty C.int
	rc := C.gguf_cuda_attn_ffn_residual_device(
		&b.drv,
		b.ctx,
		fnMatmul,
		b.fnRMS,
		b.fnSwiGLU,
		b.fnAdd,
		&b.matmulPool,
		woM.ptr,
		normM.ptr,
		gateM.ptr,
		upM.ptr,
		downM.ptr,
		xPtr,
		attnPtr,
		xPtr,
		C.int(embd),
		C.int(attnDim),
		C.int(ffn),
		C.float(eps),
		&dirty,
	)
	b.lastVecAddr = 0
	b.lastVecLen = 0
	if rc != 0 {
		// d_resid was modified: no valid hidden on host or GPU.
		// HiddenActive()=false tells the caller rollback to host is impossible.
		if resident && dirty != 0 {
			b.clearResidencyLocked()
		}

		return fmt.Errorf("cuda: attn_ffn_residual %s: code %d", t, int(rc))
	}

	return nil
}

// QKVRoPEAttentionQuantCached: QKV+RoPE+attn natively for quants.
// With active residency h computed on GPU as RMSNorm(d_resid, attnNorm).
// mode selects RoPE kernel (NeoX for Qwen/Mistral, NORM for Llama); empty qNormName/kNormName - layer without QK-norm.
func (b *Backend) QKVRoPEAttentionQuantCached(t dtype.DType, mode compute.RoPEMode, qName, kName, vName, qNormName, kNormName, attnNormName string, qRaw, kRaw, vRaw []byte, qNorm, kNorm, attnNorm, h, cos, sin, attn, kOut, vOut []float32, embd, nHeads, nKVHeads, headDim, layer, kvPos, seqLen int, eps float32) error {
	if !b.hasAttn || !b.hasRoPE || !b.hasRMS {
		return fmt.Errorf("cuda: qkv+rope+attn kernels unavailable")
	}

	fnRoPE, ropeMode, err := b.ropeKernel(mode)
	if err != nil {
		return err
	}

	if len(cos) < headDim/2 || len(sin) < headDim/2 || len(kOut) < nKVHeads*headDim || len(vOut) < nKVHeads*headDim {
		return fmt.Errorf("cuda: QKVRoPEAttentionQuantCached: buffers too short")
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	if !b.kvReady {
		return fmt.Errorf("cuda: kv cache not initialized")
	}

	// h == nil - hidden from resident residual via RMSNorm on device
	resident := b.hiddenResident && len(h) == 0
	if !resident && len(h) < embd {
		return fmt.Errorf("cuda: QKVRoPEAttentionQuantCached: h too short")
	}

	if resident && len(attnNorm) < embd {
		return fmt.Errorf("cuda: QKVRoPEAttentionQuantCached: attn_norm too short")
	}

	_, fnMatmul, err := b.quantCache(t)
	if err != nil {
		return err
	}

	qM, err := b.ensureQuantMatrix(t, qName, qRaw, nHeads*headDim, embd)
	if err != nil {
		return err
	}

	kM, err := b.ensureQuantMatrix(t, kName, kRaw, nKVHeads*headDim, embd)
	if err != nil {
		return err
	}

	vM, err := b.ensureQuantMatrix(t, vName, vRaw, nKVHeads*headDim, embd)
	if err != nil {
		return err
	}

	// Not all architectures have QK-norm: empty name means null pointer, kernel skips norm
	qNormPtr, err := b.ensureHeadNorm(qNormName, qNorm, headDim)
	if err != nil {
		return err
	}

	kNormPtr, err := b.ensureHeadNorm(kNormName, kNorm, headDim)
	if err != nil {
		return err
	}

	var attnNormPtr C.CUdeviceptr
	if resident {
		attnNormM, err := b.ensureFP32Matrix(attnNormName, attnNorm, embd, 1)
		if err != nil {
			return err
		}
		attnNormPtr = attnNormM.ptr
	}

	fnSM := b.fnSoftmax
	if !b.hasSoftmax {
		fnSM = nil
	}

	var hPtr *C.float
	if len(h) >= embd {
		hPtr = (*C.float)(unsafe.Pointer(&h[0]))
	}
	// Residency: skip attn DtoH (AttnFFN uses d_vec); still mirror k/v for CPU cache recovery.
	var attnPtr *C.float
	if !resident && len(attn) >= nHeads*headDim {
		attnPtr = (*C.float)(unsafe.Pointer(&attn[0]))
	}

	rc := C.gguf_cuda_qkv_rope_attn_device(
		&b.drv,
		b.ctx,
		fnMatmul,
		b.fnRMS,
		fnRoPE,
		b.fnAttnQK,
		b.fnAttnV,
		fnSM,
		&b.matmulPool,
		&b.attnPool,
		&b.kvCache,
		qM.ptr,
		kM.ptr,
		vM.ptr,
		qNormPtr,
		kNormPtr,
		attnNormPtr,
		hPtr,
		(*C.float)(unsafe.Pointer(&cos[0])),
		(*C.float)(unsafe.Pointer(&sin[0])),
		attnPtr,
		(*C.float)(unsafe.Pointer(&kOut[0])),
		(*C.float)(unsafe.Pointer(&vOut[0])),
		C.int(embd),
		C.int(nHeads),
		C.int(nKVHeads),
		C.int(headDim),
		C.int(layer),
		C.int(kvPos),
		C.int(seqLen),
		C.int(ropeMode),
		C.float(eps),
	)
	b.lastVecAddr = 0
	b.lastVecLen = 0
	if rc != 0 {
		return fmt.Errorf("cuda: qkv_rope_attn %s: code %d", t, int(rc))
	}

	return nil
}

// HiddenResident: CUDA keeps residual in d_resid between layers
func (b *Backend) HiddenResident() bool {
	return b.hasRMS && b.hasAdd && b.hasSwiGLU && b.hasAttn
}

// HiddenActive: d_resid holds the current hidden state
func (b *Backend) HiddenActive() bool {
	b.mu.Lock()
	defer b.mu.Unlock()

	return b.hiddenResident
}

// HiddenUpload enables residency: one HtoD hidden state per token
func (b *Backend) HiddenUpload(x []float32) error {
	if len(x) == 0 {
		return fmt.Errorf("cuda: HiddenUpload: empty x")
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	rc := C.gguf_cuda_hidden_upload(&b.drv, b.ctx, &b.matmulPool, (*C.float)(unsafe.Pointer(&x[0])), C.int(len(x)))
	if rc != 0 {
		b.hiddenResident = false
		return fmt.Errorf("cuda: hidden_upload: code %d", int(rc))
	}

	b.matmulPool.keep_resid_device = 1
	b.hiddenResident = true
	b.lastVecAddr = 0
	b.lastVecLen = 0

	return nil
}

// HiddenDownload fetches residual from device and disables residency
func (b *Backend) HiddenDownload(dst []float32) error {
	if len(dst) == 0 {
		return fmt.Errorf("cuda: HiddenDownload: empty dst")
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	if !b.hiddenResident {
		return fmt.Errorf("cuda: HiddenDownload: residency not active")
	}

	rc := C.gguf_cuda_hidden_download(&b.drv, b.ctx, &b.matmulPool, (*C.float)(unsafe.Pointer(&dst[0])), C.int(len(dst)))
	b.clearResidencyLocked()
	if rc != 0 {
		return fmt.Errorf("cuda: hidden_download: code %d", int(rc))
	}

	return nil
}

// LogitsFromDevice: out_norm + lm_head on GPU over resident hidden; DtoH only logits
func (b *Backend) LogitsFromDevice(t dtype.DType, normName string, norm []float32, headName string, headRaw []byte, headF32, logits []float32, vocab, embd int, eps float32) error {
	if !b.hasRMS {
		return fmt.Errorf("cuda: rmsnorm kernel unavailable")
	}

	if len(norm) < embd || len(logits) < vocab {
		return fmt.Errorf("cuda: LogitsFromDevice: buffers too short")
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	if !b.hiddenResident {
		return fmt.Errorf("cuda: LogitsFromDevice: residency not active")
	}

	normM, err := b.ensureFP32Matrix(normName, norm, embd, 1)
	if err != nil {
		return err
	}

	var headPtr C.CUdeviceptr
	fnMatmul := b.fn
	if t == dtype.F32 || t == dtype.F16 {
		headM, err := b.ensureFP32Matrix(headName, headF32, vocab, embd)
		if err != nil {
			return err
		}
		headPtr = headM.ptr
	} else {
		var fn C.CUfunction
		if _, fn, err = b.quantCache(t); err != nil {
			return err
		}

		headM, err := b.ensureQuantMatrix(t, headName, headRaw, vocab, embd)
		if err != nil {
			return err
		}
		headPtr = headM.ptr
		fnMatmul = fn
	}

	rc := C.gguf_cuda_logits_from_device(
		&b.drv,
		b.ctx,
		fnMatmul,
		b.fnRMS,
		&b.matmulPool,
		normM.ptr,
		headPtr,
		(*C.float)(unsafe.Pointer(&logits[0])),
		C.int(vocab),
		C.int(embd),
		C.float(eps),
	)
	b.lastVecAddr = 0
	b.lastVecLen = 0
	if rc != 0 {
		return fmt.Errorf("cuda: logits_from_device: code %d", int(rc))
	}

	return nil
}

// clearResidencyLocked disables device residency (call under b.mu)
func (b *Backend) clearResidencyLocked() {
	b.hiddenResident = false
	b.matmulPool.resid_on_device = 0
	b.matmulPool.keep_resid_device = 0
	b.matmulPool.resid_len = 0
}

// prepareVecUpload marks HtoD skip if same host vec already on GPU (Q/K/V from one h)
func (b *Backend) prepareVecUpload(vec []float32) {
	addr := uintptr(unsafe.Pointer(&vec[0]))
	if addr == b.lastVecAddr && len(vec) == b.lastVecLen {
		b.matmulPool.skip_vec_htod = 1
		return
	}

	b.matmulPool.skip_vec_htod = 0
	b.lastVecAddr = addr
	b.lastVecLen = len(vec)
}

func (b *Backend) RMSNormInto(dst, x, weight []float32, eps float32) error {
	if !b.hasRMS {
		return fmt.Errorf("cuda: rmsnorm kernel unavailable")
	}

	if len(dst) != len(x) || len(x) != len(weight) || len(x) == 0 {
		return fmt.Errorf("cuda: RMSNormInto: length mismatch")
	}

	rc := C.gguf_cuda_rmsnorm(
		&b.drv,
		b.ctx,
		b.fnRMS,
		(*C.float)(unsafe.Pointer(&x[0])),
		(*C.float)(unsafe.Pointer(&weight[0])),
		(*C.float)(unsafe.Pointer(&dst[0])),
		C.int(len(x)),
		C.float(eps),
	)
	if rc != 0 {
		return fmt.Errorf("cuda: rmsnorm: code %d", int(rc))
	}

	return nil
}

func (b *Backend) ApplyRoPEHeads(v []float32, nHeads, headDim, pos int, freqBase float32) error {
	if !b.hasRoPE {
		return fmt.Errorf("cuda: rope kernel unavailable")
	}

	half := headDim / 2
	if nHeads <= 0 || headDim <= 0 || half*2 != headDim {
		return fmt.Errorf("cuda: ApplyRoPEHeads: invalid dimensions")
	}

	if len(v) < nHeads*headDim {
		return fmt.Errorf("cuda: ApplyRoPEHeads: v too short")
	}

	if half > compute.MaxRoPEPairs() {
		return fmt.Errorf("cuda: head_dim=%d too large for GPU RoPE", headDim)
	}

	cos := make([]float32, half)
	sin := make([]float32, half)
	compute.RoPECosSin(cos, sin, headDim, pos, freqBase)

	rc := C.gguf_cuda_rope_heads(&b.drv, b.ctx, b.fnRoPE, (*C.float)(unsafe.Pointer(&v[0])), (*C.float)(unsafe.Pointer(&cos[0])), (*C.float)(unsafe.Pointer(&sin[0])), C.int(nHeads), C.int(headDim), C.int(half))
	if rc != 0 {
		return fmt.Errorf("cuda: rope_heads: code %d", int(rc))
	}

	return nil
}

func (b *Backend) ApplyRoPEHeadsNorm(v []float32, nHeads, headDim, pos int, freqBase float32) error {
	if !b.hasRoPENorm {
		return fmt.Errorf("cuda: rope_heads_norm kernel unavailable")
	}

	half := headDim / 2
	if nHeads <= 0 || headDim <= 0 || half*2 != headDim {
		return fmt.Errorf("cuda: ApplyRoPEHeadsNorm: invalid dimensions")
	}

	if len(v) < nHeads*headDim {
		return fmt.Errorf("cuda: ApplyRoPEHeadsNorm: v too short")
	}

	if half > compute.MaxRoPEPairs() {
		return fmt.Errorf("cuda: head_dim=%d too large for GPU RoPE Norm", headDim)
	}

	cos := make([]float32, half)
	sin := make([]float32, half)
	compute.RoPECosSin(cos, sin, headDim, pos, freqBase)

	rc := C.gguf_cuda_rope_heads(&b.drv, b.ctx, b.fnRoPENorm, (*C.float)(unsafe.Pointer(&v[0])), (*C.float)(unsafe.Pointer(&cos[0])), (*C.float)(unsafe.Pointer(&sin[0])), C.int(nHeads), C.int(headDim), C.int(half))
	if rc != 0 {
		return fmt.Errorf("cuda: rope_heads_norm: code %d", int(rc))
	}

	return nil
}

func (b *Backend) SwiGLUInPlace(gate, up []float32) error {
	if !b.hasSwiGLU {
		return fmt.Errorf("cuda: swiglu kernel unavailable")
	}

	if len(gate) != len(up) {
		return fmt.Errorf("cuda: SwiGLUInPlace: len(gate)=%d len(up)=%d", len(gate), len(up))
	}

	if len(gate) == 0 {
		return nil
	}

	rc := C.gguf_cuda_swiglu(&b.drv, b.ctx, b.fnSwiGLU, (*C.float)(unsafe.Pointer(&gate[0])), (*C.float)(unsafe.Pointer(&up[0])), C.int(len(gate)))
	if rc != 0 {
		return fmt.Errorf("cuda: swiglu: code %d", int(rc))
	}

	return nil
}

func (b *Backend) AttentionScoresInto(dst, q, k, v, scores []float32, seqLen, nHeads, nKVHeads, headDim int) error {
	_ = scores
	if !b.hasAttn {
		return fmt.Errorf("cuda: attention kernels unavailable")
	}

	if len(dst) < nHeads*headDim {
		return fmt.Errorf("cuda: AttentionScoresInto: dst too short")
	}

	if seqLen <= 0 || nHeads <= 0 || nKVHeads <= 0 || headDim <= 0 || nHeads%nKVHeads != 0 {
		return fmt.Errorf("cuda: AttentionScoresInto: invalid dimensions")
	}

	kvStride := nKVHeads * headDim
	if len(q) < nHeads*headDim {
		return fmt.Errorf("cuda: AttentionScoresInto: q too short")
	}

	if len(k) < seqLen*kvStride || len(v) < seqLen*kvStride {
		return fmt.Errorf("cuda: AttentionScoresInto: k/v too short")
	}

	fnSM := b.fnSoftmax
	if !b.hasSoftmax {
		fnSM = nil
	}
	rc := C.gguf_cuda_attention(
		&b.drv,
		b.ctx,
		b.fnAttnQK,
		b.fnAttnV,
		fnSM,
		(*C.float)(unsafe.Pointer(&dst[0])),
		(*C.float)(unsafe.Pointer(&q[0])),
		(*C.float)(unsafe.Pointer(&k[0])),
		(*C.float)(unsafe.Pointer(&v[0])),
		C.int(seqLen),
		C.int(nHeads),
		C.int(nKVHeads),
		C.int(headDim))
	if rc != 0 {
		return fmt.Errorf("cuda: attention: code %d", int(rc))
	}

	return nil
}

func (b *Backend) KVCacheInit(layers, maxSeq, kvDim, nHeads, headDim int) error {
	if !b.hasAttn {
		return fmt.Errorf("cuda: attention kernels unavailable")
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	if b.kvReady {
		C.gguf_cuda_kv_free(&b.drv, &b.kvCache)
		b.kvReady = false
	}

	C.gguf_cuda_attn_pool_free(&b.drv, &b.attnPool)
	C.gguf_cuda_matmul_pool_clear_graphs(&b.drv, &b.matmulPool)

	rc := C.gguf_cuda_kv_init(&b.drv, b.ctx, &b.kvCache, C.int(layers), C.int(maxSeq), C.int(kvDim))
	if rc != 0 {
		return fmt.Errorf("cuda: kv_init: code %d", int(rc))
	}

	qBytes := nHeads * headDim
	rc = C.gguf_cuda_attn_pool_init(&b.drv, b.ctx, &b.attnPool, C.int(qBytes), C.int(maxSeq), C.int(kvDim), C.int(headDim/2))
	if rc != 0 {
		C.gguf_cuda_kv_free(&b.drv, &b.kvCache)
		return fmt.Errorf("cuda: attn_pool_init: code %d", int(rc))
	}

	b.kvReady = true
	return nil
}

func (b *Backend) KVCacheReset() {
	// Logical reset: new Append starts at pos=0, old data overwritten.
}

func (b *Backend) KVCacheAppend(layer, pos int, k, v []float32) error {
	if len(k) == 0 || len(v) == 0 {
		return fmt.Errorf("cuda: KVCacheAppend: empty k/v")
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	if !b.kvReady {
		return fmt.Errorf("cuda: kv cache not initialized")
	}

	rc := C.gguf_cuda_kv_append(
		&b.drv,
		b.ctx,
		&b.kvCache,
		C.int(layer),
		C.int(pos),
		(*C.float)(unsafe.Pointer(&k[0])),
		(*C.float)(unsafe.Pointer(&v[0])),
	)
	if rc != 0 {
		return fmt.Errorf("cuda: kv_append layer=%d pos=%d: code %d", layer, pos, int(rc))
	}

	return nil
}

func (b *Backend) KVCacheAppendN(layer, pos int, k, v []float32, n int) error {
	if n <= 0 {
		return fmt.Errorf("cuda: KVCacheAppendN: n=%d", n)
	}

	if len(k) == 0 || len(v) == 0 {
		return fmt.Errorf("cuda: KVCacheAppendN: empty k/v")
	}

	if len(k)%n != 0 || len(v)%n != 0 {
		return fmt.Errorf("cuda: KVCacheAppendN: len(k)=%d len(v)=%d not divisible by n=%d", len(k), len(v), n)
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	if !b.kvReady {
		return fmt.Errorf("cuda: kv cache not initialized")
	}

	rc := C.gguf_cuda_kv_append_n(
		&b.drv,
		b.ctx,
		&b.kvCache,
		C.int(layer),
		C.int(pos),
		(*C.float)(unsafe.Pointer(&k[0])),
		(*C.float)(unsafe.Pointer(&v[0])),
		C.int(n),
	)
	if rc != 0 {
		return fmt.Errorf("cuda: kv_append_n layer=%d pos=%d n=%d: code %d", layer, pos, n, int(rc))
	}

	return nil
}

func (b *Backend) AttentionScoresKV(layer int, dst, q []float32, seqLen, nHeads, nKVHeads, headDim int) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	return b.attentionScoresKVLocked(layer, dst, q, seqLen, nHeads, nKVHeads, headDim)
}

// attentionScoresKVLocked - single-query attention over KV cache (caller holds b.mu).
// Prefers fused attn_flash (one launch for all heads); falls back to per-head attn_qk/softmax/attn_v.
func (b *Backend) attentionScoresKVLocked(layer int, dst, q []float32, seqLen, nHeads, nKVHeads, headDim int) error {
	if !b.hasAttn {
		return fmt.Errorf("cuda: attention kernels unavailable")
	}

	if len(dst) < nHeads*headDim {
		return fmt.Errorf("cuda: AttentionScoresKV: dst too short")
	}

	if len(q) < nHeads*headDim {
		return fmt.Errorf("cuda: AttentionScoresKV: q too short")
	}

	if !b.kvReady {
		return fmt.Errorf("cuda: kv cache not initialized")
	}

	fnSM := b.fnSoftmax
	if !b.hasSoftmax {
		fnSM = nil
	}
	rc := C.gguf_cuda_kv_attention(
		&b.drv,
		b.ctx,
		b.fnAttnQK,
		b.fnAttnV,
		fnSM,
		&b.kvCache,
		&b.attnPool,
		C.int(layer),
		(*C.float)(unsafe.Pointer(&dst[0])),
		(*C.float)(unsafe.Pointer(&q[0])),
		C.int(seqLen),
		C.int(nHeads),
		C.int(nKVHeads),
		C.int(headDim),
	)
	if rc != 0 {
		return fmt.Errorf("cuda: kv_attention layer=%d: code %d", layer, int(rc))
	}

	return nil
}

// HasFlashAttention reports whether the fused attn_flash kernel is loaded.
func (b *Backend) HasFlashAttention() bool { return b.hasAttnFlash }

// AttentionScoresKVBatch - causal attention for n queries over the GPU KV cache.
// q/dst are [n][nHeads*headDim]; query i attends to pastLen+i+1 positions (K/V of all n tokens must already be in the cache, e.g. via KVCacheAppendN).
//
// With attn_flash: ONE Q HtoD, ONE launch (grid = nHeads x n), ONE DtoH.
// Limits for the fused path: headDim <= 256, pastLen+n <= maxSeq.
// Temporary q/out buffers (n*nHeads*headDim floats each) are allocated per call.
// Otherwise falls back to a loop of single-query attention (per-query HtoD/DtoH, three launches per head).
func (b *Backend) AttentionScoresKVBatch(layer int, dst, q []float32, n, pastLen, nHeads, nKVHeads, headDim int) error {
	if !b.hasAttn {
		return fmt.Errorf("cuda: attention kernels unavailable")
	}

	if n <= 0 || pastLen < 0 || nHeads <= 0 || nKVHeads <= 0 || headDim <= 0 || nHeads%nKVHeads != 0 {
		return fmt.Errorf("cuda: AttentionScoresKVBatch: invalid dimensions")
	}

	qDim := nHeads * headDim
	if len(dst) < n*qDim || len(q) < n*qDim {
		return fmt.Errorf("cuda: AttentionScoresKVBatch: q/dst too short")
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	if !b.kvReady {
		return fmt.Errorf("cuda: kv cache not initialized")
	}

	if b.hasAttnFlash && headDim <= 256 {
		rc := C.gguf_cuda_kv_attention_batch(
			&b.drv,
			b.ctx,
			&b.kvCache,
			C.int(layer),
			(*C.float)(unsafe.Pointer(&dst[0])),
			(*C.float)(unsafe.Pointer(&q[0])),
			C.int(n),
			C.int(pastLen),
			C.int(nHeads),
			C.int(nKVHeads),
			C.int(headDim),
		)
		if rc == 0 {
			return nil
		}
	}

	for i := range n {
		off := i * qDim
		if err := b.attentionScoresKVLocked(layer, dst[off:off+qDim], q[off:off+qDim], pastLen+i+1, nHeads, nKVHeads, headDim); err != nil {
			return err
		}
	}

	return nil
}

func validateMatMul(matrix []float32, rows, cols int, vec []float32) error {
	if rows <= 0 || cols <= 0 {
		return fmt.Errorf("cuda: rows=%d cols=%d", rows, cols)
	}

	if len(vec) != cols {
		return fmt.Errorf("cuda: len(vec)=%d, cols=%d", len(vec), cols)
	}

	if len(matrix) < rows*cols {
		return fmt.Errorf("cuda: matrix too short")
	}

	return nil
}

func validateQ8MatMul(raw []byte, rows, cols int, vec []float32) error {
	if rows <= 0 || cols <= 0 {
		return fmt.Errorf("cuda: rows=%d cols=%d", rows, cols)
	}

	if cols%quant.QK8_0 != 0 {
		return fmt.Errorf("cuda: cols=%d not divisible by %d", cols, quant.QK8_0)
	}

	if len(vec) != cols {
		return fmt.Errorf("cuda: len(vec)=%d, cols=%d", len(vec), cols)
	}

	blocksPerRow := cols / quant.QK8_0
	want := rows * blocksPerRow * quant.BlockQ8_0Size
	if len(raw) < want {
		return fmt.Errorf("cuda: Q8_0 matrix too short")
	}

	return nil
}

func validateQ4MatMul(raw []byte, rows, cols int, vec []float32) error {
	if rows <= 0 || cols <= 0 {
		return fmt.Errorf("cuda: rows=%d cols=%d", rows, cols)
	}

	if cols%quant.QK4_0 != 0 {
		return fmt.Errorf("cuda: cols=%d not divisible by %d", cols, quant.QK4_0)
	}

	if len(vec) != cols {
		return fmt.Errorf("cuda: len(vec)=%d, cols=%d", len(vec), cols)
	}

	blocksPerRow := cols / quant.QK4_0
	want := rows * blocksPerRow * quant.BlockQ4_0Size
	if len(raw) < want {
		return fmt.Errorf("cuda: Q4_0 matrix too short")
	}

	return nil
}

func validateQ4KMatMul(raw []byte, rows, cols int, vec []float32) error {
	if rows <= 0 || cols <= 0 {
		return fmt.Errorf("cuda: rows=%d cols=%d", rows, cols)
	}

	if cols%quant.QK_K != 0 {
		return fmt.Errorf("cuda: cols=%d not divisible by %d", cols, quant.QK_K)
	}

	if len(vec) != cols {
		return fmt.Errorf("cuda: len(vec)=%d, cols=%d", len(vec), cols)
	}

	blocksPerRow := cols / quant.QK_K
	want := rows * blocksPerRow * quant.BlockQ4_KSize
	if len(raw) < want {
		return fmt.Errorf("cuda: Q4_K matrix too short")
	}

	return nil
}

func validateQ5KMatMul(raw []byte, rows, cols int, vec []float32) error {
	if rows <= 0 || cols <= 0 {
		return fmt.Errorf("cuda: rows=%d cols=%d", rows, cols)
	}

	if cols%quant.QK_K != 0 {
		return fmt.Errorf("cuda: cols=%d not divisible by %d", cols, quant.QK_K)
	}

	if len(vec) != cols {
		return fmt.Errorf("cuda: len(vec)=%d, cols=%d", len(vec), cols)
	}

	blocksPerRow := cols / quant.QK_K
	want := rows * blocksPerRow * quant.BlockQ5_KSize
	if len(raw) < want {
		return fmt.Errorf("cuda: Q5_K matrix too short")
	}

	return nil
}

func validateQ6KMatMul(raw []byte, rows, cols int, vec []float32) error {
	if rows <= 0 || cols <= 0 {
		return fmt.Errorf("cuda: rows=%d cols=%d", rows, cols)
	}

	if cols%quant.QK_K != 0 {
		return fmt.Errorf("cuda: cols=%d not divisible by %d", cols, quant.QK_K)
	}

	if len(vec) != cols {
		return fmt.Errorf("cuda: len(vec)=%d, cols=%d", len(vec), cols)
	}

	blocksPerRow := cols / quant.QK_K
	want := rows * blocksPerRow * quant.BlockQ6_KSize
	if len(raw) < want {
		return fmt.Errorf("cuda: Q6_K matrix too short")
	}

	return nil
}

func validateQ41MatMul(raw []byte, rows, cols int, vec []float32) error {
	if rows <= 0 || cols <= 0 {
		return fmt.Errorf("cuda: rows=%d cols=%d", rows, cols)
	}

	if cols%quant.QK4_1 != 0 {
		return fmt.Errorf("cuda: cols=%d not divisible by %d", cols, quant.QK4_1)
	}

	if len(vec) != cols {
		return fmt.Errorf("cuda: len(vec)=%d, cols=%d", len(vec), cols)
	}

	blocksPerRow := cols / quant.QK4_1
	want := rows * blocksPerRow * quant.BlockQ4_1Size
	if len(raw) < want {
		return fmt.Errorf("cuda: Q4_1 matrix too short")
	}

	return nil
}

func validateQ50MatMul(raw []byte, rows, cols int, vec []float32) error {
	if rows <= 0 || cols <= 0 {
		return fmt.Errorf("cuda: rows=%d cols=%d", rows, cols)
	}

	if cols%quant.QK5_0 != 0 {
		return fmt.Errorf("cuda: cols=%d not divisible by %d", cols, quant.QK5_0)
	}

	if len(vec) != cols {
		return fmt.Errorf("cuda: len(vec)=%d, cols=%d", len(vec), cols)
	}

	blocksPerRow := cols / quant.QK5_0
	want := rows * blocksPerRow * quant.BlockQ5_0Size
	if len(raw) < want {
		return fmt.Errorf("cuda: Q5_0 matrix too short")
	}

	return nil
}

func validateQ51MatMul(raw []byte, rows, cols int, vec []float32) error {
	if rows <= 0 || cols <= 0 {
		return fmt.Errorf("cuda: rows=%d cols=%d", rows, cols)
	}

	if cols%quant.QK5_1 != 0 {
		return fmt.Errorf("cuda: cols=%d not divisible by %d", cols, quant.QK5_1)
	}

	if len(vec) != cols {
		return fmt.Errorf("cuda: len(vec)=%d, cols=%d", len(vec), cols)
	}

	blocksPerRow := cols / quant.QK5_1
	want := rows * blocksPerRow * quant.BlockQ5_1Size
	if len(raw) < want {
		return fmt.Errorf("cuda: Q5_1 matrix too short")
	}

	return nil
}

func validateQ2KMatMul(raw []byte, rows, cols int, vec []float32) error {
	if rows <= 0 || cols <= 0 {
		return fmt.Errorf("cuda: rows=%d cols=%d", rows, cols)
	}

	if cols%quant.QK_K != 0 {
		return fmt.Errorf("cuda: cols=%d not divisible by %d", cols, quant.QK_K)
	}

	if len(vec) != cols {
		return fmt.Errorf("cuda: len(vec)=%d, cols=%d", len(vec), cols)
	}

	blocksPerRow := cols / quant.QK_K
	want := rows * blocksPerRow * quant.BlockQ2_KSize
	if len(raw) < want {
		return fmt.Errorf("cuda: Q2_K matrix too short")
	}

	return nil
}

func validateQ3KMatMul(raw []byte, rows, cols int, vec []float32) error {
	if rows <= 0 || cols <= 0 {
		return fmt.Errorf("cuda: rows=%d cols=%d", rows, cols)
	}

	if cols%quant.QK_K != 0 {
		return fmt.Errorf("cuda: cols=%d not divisible by %d", cols, quant.QK_K)
	}

	if len(vec) != cols {
		return fmt.Errorf("cuda: len(vec)=%d, cols=%d", len(vec), cols)
	}

	blocksPerRow := cols / quant.QK_K
	want := rows * blocksPerRow * quant.BlockQ3_KSize
	if len(raw) < want {
		return fmt.Errorf("cuda: Q3_K matrix too short")
	}

	return nil
}

func validateQ8KMatMul(raw []byte, rows, cols int, vec []float32) error {
	if rows <= 0 || cols <= 0 {
		return fmt.Errorf("cuda: rows=%d cols=%d", rows, cols)
	}

	if cols%quant.QK_K != 0 {
		return fmt.Errorf("cuda: cols=%d not divisible by %d", cols, quant.QK_K)
	}

	if len(vec) != cols {
		return fmt.Errorf("cuda: len(vec)=%d, cols=%d", len(vec), cols)
	}

	blocksPerRow := cols / quant.QK_K
	want := rows * blocksPerRow * quant.BlockQ8_KSize
	if len(raw) < want {
		return fmt.Errorf("cuda: Q8_K matrix too short")
	}

	return nil
}
