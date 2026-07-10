//go:build cuda

package cuda

/*
#cgo LDFLAGS: -ldl -lm
#include "driver.h"
#include <stdlib.h>
*/
import "C"
import (
	"fmt"
	"sync"
	"unsafe"

	"github.com/magomedcoder/gogguf/pkg/ops"
	"github.com/magomedcoder/gogguf/pkg/quant"
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
	name      string
	drv       C.cuda_driver_t
	lib       unsafe.Pointer
	ctx       C.CUcontext
	module    C.CUmodule
	moduleOps C.CUmodule
	fn        C.CUfunction
	fnQ8      C.CUfunction
	fnRMS     C.CUfunction
	fnRoPE    C.CUfunction
	fnSwiGLU  C.CUfunction
	fnAttnQK  C.CUfunction
	fnAttnV   C.CUfunction
	hasRMS    bool
	hasRoPE   bool
	hasSwiGLU bool
	hasAttn   bool

	mu         sync.Mutex
	matrices   map[string]gpuMatrix
	matricesQ8 map[string]gpuQ8Matrix
	kvCache    C.gguf_kv_cache_t
	kvReady    bool
	attnPool   C.gguf_attn_pool_t
}

// Open инициализирует GPU 0 и загружает kernels
func Open() (*Backend, error) {
	b := &Backend{
		matrices:   make(map[string]gpuMatrix),
		matricesQ8: make(map[string]gpuQ8Matrix),
	}

	var nameBuf [256]C.char
	var initErr [512]C.char
	var cc C.int
	rc := C.gguf_cuda_init(&b.drv, &b.lib, &b.ctx, &nameBuf[0], C.size_t(len(nameBuf)), &initErr[0], C.size_t(len(initErr)), &cc)
	if rc != 0 {
		msg := C.GoString(&initErr[0])
		if msg == "" {
			msg = C.GoString(&nameBuf[0])
		}

		return nil, fmt.Errorf("cuda: init: код %d: %s", int(rc), msg)
	}

	b.name = "CUDA:0 " + C.GoString(&nameBuf[0])
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

			cAttnQK := C.CString("attn_qk")
			cAttnV := C.CString("attn_v")
			if C.gguf_cuda_module_function(&b.drv, b.moduleOps, cAttnQK, &b.fnAttnQK) == 0 && C.gguf_cuda_module_function(&b.drv, b.moduleOps, cAttnV, &b.fnAttnV) == 0 {
				b.hasAttn = true
			}
			C.free(unsafe.Pointer(cAttnQK))
			C.free(unsafe.Pointer(cAttnV))

			return nil
		}

		lastErr = C.GoString(&errBuf[0])
	}

	return fmt.Errorf("cuda: load ops module (cc=%d): %s", gpuCC, lastErr)
}

func (b *Backend) Name() string { return b.name }

func (b *Backend) Close() error {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.kvReady {
		C.gguf_cuda_kv_free(&b.drv, &b.kvCache)
		b.kvReady = false
	}

	C.gguf_cuda_attn_pool_free(&b.drv, &b.attnPool)

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

func (b *Backend) RMSNormInto(dst, x, weight []float32, eps float32) error {
	if !b.hasRMS {
		return fmt.Errorf("cuda: rmsnorm kernel недоступен")
	}

	if len(dst) != len(x) || len(x) != len(weight) || len(x) == 0 {
		return fmt.Errorf("cuda: RMSNormInto: несовпадение длин")
	}

	rc := C.gguf_cuda_rmsnorm(
		&b.drv, b.ctx, b.fnRMS,
		(*C.float)(unsafe.Pointer(&x[0])),
		(*C.float)(unsafe.Pointer(&weight[0])),
		(*C.float)(unsafe.Pointer(&dst[0])),
		C.int(len(x)), C.float(eps),
	)
	if rc != 0 {
		return fmt.Errorf("cuda: rmsnorm: код %d", int(rc))
	}

	return nil
}

func (b *Backend) ApplyRoPEHeads(v []float32, nHeads, headDim, pos int, freqBase float32) error {
	if !b.hasRoPE {
		return fmt.Errorf("cuda: rope kernel недоступен")
	}

	half := headDim / 2
	if nHeads <= 0 || headDim <= 0 || half*2 != headDim {
		return fmt.Errorf("cuda: ApplyRoPEHeads: неверные размеры")
	}

	if len(v) < nHeads*headDim {
		return fmt.Errorf("cuda: ApplyRoPEHeads: v слишком короткий")
	}

	if half > ops.MaxRoPEPairs() {
		return fmt.Errorf("cuda: head_dim=%d слишком велик для GPU RoPE", headDim)
	}

	cos := make([]float32, half)
	sin := make([]float32, half)
	ops.RoPECosSin(cos, sin, headDim, pos, freqBase)

	rc := C.gguf_cuda_rope_heads(&b.drv, b.ctx, b.fnRoPE, (*C.float)(unsafe.Pointer(&v[0])), (*C.float)(unsafe.Pointer(&cos[0])), (*C.float)(unsafe.Pointer(&sin[0])), C.int(nHeads), C.int(headDim), C.int(half))
	if rc != 0 {
		return fmt.Errorf("cuda: rope_heads: код %d", int(rc))
	}

	return nil
}

func (b *Backend) SwiGLUInPlace(gate, up []float32) error {
	if !b.hasSwiGLU {
		return fmt.Errorf("cuda: swiglu kernel недоступен")
	}

	if len(gate) != len(up) {
		return fmt.Errorf("cuda: SwiGLUInPlace: len(gate)=%d len(up)=%d", len(gate), len(up))
	}

	if len(gate) == 0 {
		return nil
	}

	rc := C.gguf_cuda_swiglu(&b.drv, b.ctx, b.fnSwiGLU, (*C.float)(unsafe.Pointer(&gate[0])), (*C.float)(unsafe.Pointer(&up[0])), C.int(len(gate)))
	if rc != 0 {
		return fmt.Errorf("cuda: swiglu: код %d", int(rc))
	}

	return nil
}

func (b *Backend) AttentionScoresInto(dst, q, k, v, scores []float32, seqLen, nHeads, nKVHeads, headDim int) error {
	_ = scores
	if !b.hasAttn {
		return fmt.Errorf("cuda: attention kernels недоступны")
	}

	if len(dst) < nHeads*headDim {
		return fmt.Errorf("cuda: AttentionScoresInto: dst слишком короткий")
	}

	if seqLen <= 0 || nHeads <= 0 || nKVHeads <= 0 || headDim <= 0 || nHeads%nKVHeads != 0 {
		return fmt.Errorf("cuda: AttentionScoresInto: неверные размеры")
	}

	kvStride := nKVHeads * headDim
	if len(q) < nHeads*headDim {
		return fmt.Errorf("cuda: AttentionScoresInto: q слишком короткий")
	}

	if len(k) < seqLen*kvStride || len(v) < seqLen*kvStride {
		return fmt.Errorf("cuda: AttentionScoresInto: k/v слишком короткие")
	}

	rc := C.gguf_cuda_attention(
		&b.drv,
		b.ctx,
		b.fnAttnQK,
		b.fnAttnV,
		(*C.float)(unsafe.Pointer(&dst[0])),
		(*C.float)(unsafe.Pointer(&q[0])),
		(*C.float)(unsafe.Pointer(&k[0])),
		(*C.float)(unsafe.Pointer(&v[0])),
		C.int(seqLen),
		C.int(nHeads),
		C.int(nKVHeads),
		C.int(headDim))
	if rc != 0 {
		return fmt.Errorf("cuda: attention: код %d", int(rc))
	}

	return nil
}

func (b *Backend) KVCacheInit(layers, maxSeq, kvDim, nHeads, headDim int) error {
	if !b.hasAttn {
		return fmt.Errorf("cuda: attention kernels недоступны")
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	if b.kvReady {
		C.gguf_cuda_kv_free(&b.drv, &b.kvCache)
		b.kvReady = false
	}

	C.gguf_cuda_attn_pool_free(&b.drv, &b.attnPool)

	rc := C.gguf_cuda_kv_init(&b.drv, b.ctx, &b.kvCache, C.int(layers), C.int(maxSeq), C.int(kvDim))
	if rc != 0 {
		return fmt.Errorf("cuda: kv_init: код %d", int(rc))
	}

	qBytes := nHeads * headDim
	rc = C.gguf_cuda_attn_pool_init(&b.drv, b.ctx, &b.attnPool, C.int(qBytes), C.int(maxSeq))
	if rc != 0 {
		C.gguf_cuda_kv_free(&b.drv, &b.kvCache)
		return fmt.Errorf("cuda: attn_pool_init: код %d", int(rc))
	}

	b.kvReady = true
	return nil
}

func (b *Backend) KVCacheReset() {
	// Логический сброс: новые Append начнутся с pos=0, старые данные перезаписываются.
}

func (b *Backend) KVCacheAppend(layer, pos int, k, v []float32) error {
	if len(k) == 0 || len(v) == 0 {
		return fmt.Errorf("cuda: KVCacheAppend: пустой k/v")
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	if !b.kvReady {
		return fmt.Errorf("cuda: kv cache не инициализирован")
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
		return fmt.Errorf("cuda: kv_append layer=%d pos=%d: код %d", layer, pos, int(rc))
	}

	return nil
}

func (b *Backend) AttentionScoresKV(layer int, dst, q []float32, seqLen, nHeads, nKVHeads, headDim int) error {
	if !b.hasAttn {
		return fmt.Errorf("cuda: attention kernels недоступны")
	}

	if len(dst) < nHeads*headDim {
		return fmt.Errorf("cuda: AttentionScoresKV: dst слишком короткий")
	}

	if len(q) < nHeads*headDim {
		return fmt.Errorf("cuda: AttentionScoresKV: q слишком короткий")
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	if !b.kvReady {
		return fmt.Errorf("cuda: kv cache не инициализирован")
	}

	rc := C.gguf_cuda_kv_attention(
		&b.drv,
		b.ctx,
		b.fnAttnQK,
		b.fnAttnV,
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
		return fmt.Errorf("cuda: kv_attention layer=%d: код %d", layer, int(rc))
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
