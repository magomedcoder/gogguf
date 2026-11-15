//go:build metal && darwin && cgo

package metal

/*
#cgo CFLAGS: -fobjc-arc -Wno-deprecated-declarations
#cgo LDFLAGS: -framework Metal -framework Foundation
#include <stdlib.h>
#include "driver.h"
*/
import "C"

import (
	"fmt"
	"math/bits"
	"sync"
	"unsafe"

	"github.com/magomedcoder/gogguf/pkg/compute"
	"github.com/magomedcoder/gogguf/pkg/dtype"
	"github.com/magomedcoder/gogguf/pkg/quant"
)

// devBuf is a shared-storage MTLBuffer: p is the CPU-visible pointer to the same unified memory the GPU reads
type devBuf struct {
	h C.gguf_metal_buf_t
	p unsafe.Pointer
	n int // size in bytes
}

func (d *devBuf) floats(n int) []float32 {
	return unsafe.Slice((*float32)(d.p), n)
}

// devWeight is a cached weight matrix (F32 or Q8_0 blocks) uploaded once by tensor name
type devWeight struct {
	buf   devBuf
	rows  int
	cols  int
	bytes int
}

// mtl is the darwin runtime behind Backend: one MTLDevice/queue, weight cache, scratch buffers and per-layer KV buffers.
// Every operation is encoded, committed and waited on inside one call (driver.h), serialized by mu.
type mtl struct {
	mu   sync.Mutex
	ctx  C.gguf_metal_t
	name string

	f32  map[string]*devWeight
	q8   map[string]*devWeight
	q4_0 map[string]*devWeight
	q4_k map[string]*devWeight

	// scratch: grown on demand (power-of-two) and reused across calls
	vec, out, tmpW  devBuf // matvec input / output / uncached weight upload
	nx, nw, nd      devBuf // rmsnorm x / weight / dst
	rope            devBuf // cos[half] followed by sin[half], then the vector being rotated lives in ropeV
	ropeV           devBuf
	gate, up        devBuf // swiglu
	aq, ak, av, ao  devBuf // attention q / host k / host v / out
	scores          devBuf // [nHeads][stride] softmax weights
	kvScores        devBuf
	kvLayers        int
	kvMaxSeq, kvDim int
	kvK, kvV        []devBuf
	kvLen           []int // cached positions per layer

	// B2 residency / fused decode (fused_darwin.go): resid is the device-resident residual stream (Backend owns the "active" flag); the other buffers are per-layer scratch reused across fused calls.
	resid              devBuf
	hn, fq, fo, fg, fu devBuf // normed hidden / Q / WO-and-down output / gate / up
	fattn              devBuf // attention output, left on the device between QKV and AttnFFN
	fattnLen           int    // floats valid in fattn (0 = none)

	// B3 prefill / batch (batch_darwin.go): scratch separate from the B2 buffers so a prefill never disturbs a resident decode state.
	// mx stages the host input X[batch][cols] (reused while xPtr/xLen/content still match = skip_x), mo the GEMM output, bgate/bup/bout the batch FFN activations.
	mx, mo, bgate, bup, bout devBuf
	xPtr                     uintptr // host slice staged in mx (0 = nothing reusable)
	xLen                     int
}

// openGPU creates the Metal device and compiles the B1/B2 MSL kernels
func openGPU() (gpu, string, error) {
	src := C.CString(mslSource)
	defer C.free(unsafe.Pointer(src))

	errBuf := make([]byte, 2048)
	ctx := C.gguf_metal_open(src, (*C.char)(unsafe.Pointer(&errBuf[0])), C.size_t(len(errBuf)))
	if ctx == nil {
		return nil, "", fmt.Errorf("%s", C.GoString((*C.char)(unsafe.Pointer(&errBuf[0]))))
	}

	r := &mtl{
		ctx:  ctx,
		f32:  make(map[string]*devWeight),
		q8:   make(map[string]*devWeight),
		q4_0: make(map[string]*devWeight),
		q4_k: make(map[string]*devWeight),
		name: C.GoString(C.gguf_metal_device_name(ctx)),
	}

	return r, r.name, nil
}

func (r *mtl) deviceName() string {
	return r.name
}

func (r *mtl) vram() (used, total uint64) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.ctx == nil {
		return 0, 0
	}

	return uint64(C.gguf_metal_allocated(r.ctx)), uint64(C.gguf_metal_recommended(r.ctx))
}

func (r *mtl) lastErr() string {
	return C.GoString(C.gguf_metal_last_error(r.ctx))
}

func (r *mtl) opened() error {
	if r.ctx == nil {
		return fmt.Errorf("metal: backend closed")
	}

	return nil
}

func (r *mtl) free(d *devBuf) {
	if d.h != nil {
		C.gguf_metal_buf_free(d.h)
	}

	*d = devBuf{}
}

// ensure grows d to at least n bytes (power-of-two) without keeping old contents
func (r *mtl) ensure(d *devBuf, n int) error {
	if d.n >= n && d.h != nil {
		return nil
	}

	size := 4096
	if n > size {
		size = 1 << bits.Len(uint(n-1))
	}

	r.free(d)

	h := C.gguf_metal_buf_new(r.ctx, C.size_t(size))
	if h == nil {
		return fmt.Errorf("metal: alloc %d bytes: %s", size, r.lastErr())
	}

	*d = devBuf{h: h, p: C.gguf_metal_buf_ptr(h), n: size}

	return nil
}

// ensureExact allocates exactly n bytes (KV layers)
func (r *mtl) ensureExact(d *devBuf, n int) error {
	r.free(d)

	h := C.gguf_metal_buf_new(r.ctx, C.size_t(n))
	if h == nil {
		return fmt.Errorf("metal: alloc %d bytes: %s", n, r.lastErr())
	}

	*d = devBuf{h: h, p: C.gguf_metal_buf_ptr(h), n: n}

	return nil
}

func (r *mtl) putFloats(d *devBuf, src []float32) error {
	if err := r.ensure(d, len(src)*4); err != nil {
		return err
	}

	copy(d.floats(len(src)), src)

	return nil
}

func (r *mtl) check(op string, rc C.int) error {
	if rc != 0 {
		return fmt.Errorf("metal: %s: code %d: %s", op, int(rc), r.lastErr())
	}

	return nil
}

// weight returns the device buffer for a matrix; name != "" caches by name (re-upload when shape changes), "" uses the shared temp upload buffer.
func (r *mtl) weight(cache map[string]*devWeight, name string, src unsafe.Pointer, bytes, rows, cols int) (C.gguf_metal_buf_t, error) {
	if name == "" {
		if err := r.ensure(&r.tmpW, bytes); err != nil {
			return nil, err
		}

		copy(unsafe.Slice((*byte)(r.tmpW.p), bytes), unsafe.Slice((*byte)(src), bytes))

		return r.tmpW.h, nil
	}

	if w, ok := cache[name]; ok {
		if w.rows == rows && w.cols == cols && w.bytes == bytes {
			return w.buf.h, nil
		}

		r.free(&w.buf)
		delete(cache, name)
	}

	h := C.gguf_metal_buf_from(r.ctx, src, C.size_t(bytes))
	if h == nil {
		return nil, fmt.Errorf("metal: upload weight %q (%d bytes): %s", name, bytes, r.lastErr())
	}

	cache[name] = &devWeight{buf: devBuf{h: h, p: C.gguf_metal_buf_ptr(h), n: bytes}, rows: rows, cols: cols, bytes: bytes}

	return h, nil
}

func (r *mtl) matMulF32(name string, matrix []float32, rows, cols int, vec, dst []float32) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if err := r.opened(); err != nil {
		return err
	}

	w, err := r.weight(r.f32, name, unsafe.Pointer(&matrix[0]), rows*cols*4, rows, cols)
	if err != nil {
		return err
	}

	if err := r.putFloats(&r.vec, vec); err != nil {
		return err
	}

	if err := r.ensure(&r.out, rows*4); err != nil {
		return err
	}

	if err := r.check("matmul_vec_f32", C.gguf_metal_matmul_f32(r.ctx, w, r.vec.h, r.out.h, C.uint32_t(rows), C.uint32_t(cols))); err != nil {
		return err
	}

	copy(dst, r.out.floats(rows))

	return nil
}

func (r *mtl) matMulQ8_0(name string, raw []byte, rows, cols int, vec, dst []float32) error {
	return r.matMulQuant(dtype.Q8_0, name, raw, rows, cols, vec, dst)
}

// quantKind maps a quant type to its matvec kernel kind, block geometry and weight cache
func (r *mtl) quantKind(t dtype.DType) (kind C.int, qk, blockBytes int, cache map[string]*devWeight, err error) {
	switch t {
	case dtype.Q8_0:
		return C.GGUF_METAL_MAT_Q8_0, quant.QK8_0, quant.BlockQ8_0Size, r.q8, nil
	case dtype.Q4_0:
		return C.GGUF_METAL_MAT_Q4_0, quant.QK4_0, quant.BlockQ4_0Size, r.q4_0, nil
	case dtype.Q4_K:
		return C.GGUF_METAL_MAT_Q4_K, quant.QK_K, quant.BlockQ4_KSize, r.q4_k, nil
	default:
		return 0, 0, 0, nil, unavailable("matmul " + t.String())
	}
}

func (r *mtl) matMulQuant(t dtype.DType, name string, raw []byte, rows, cols int, vec, dst []float32) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if err := r.opened(); err != nil {
		return err
	}

	kind, qk, bs, cache, err := r.quantKind(t)
	if err != nil {
		return err
	}

	bytes := rows * (cols / qk) * bs
	w, err := r.weight(cache, name, unsafe.Pointer(&raw[0]), bytes, rows, cols)
	if err != nil {
		return err
	}

	if err := r.putFloats(&r.vec, vec); err != nil {
		return err
	}

	if err := r.ensure(&r.out, rows*4); err != nil {
		return err
	}

	if err := r.check("matmul_vec_"+t.String(), C.gguf_metal_matmul_off(r.ctx, C.int(kind), w, r.vec.h, 0, r.out.h, 0, C.uint32_t(rows), C.uint32_t(cols))); err != nil {
		return err
	}

	copy(dst, r.out.floats(rows))

	return nil
}

func (r *mtl) rmsNorm(dst, x, weight []float32, eps float32) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if err := r.opened(); err != nil {
		return err
	}

	if err := r.putFloats(&r.nx, x); err != nil {
		return err
	}

	if err := r.putFloats(&r.nw, weight); err != nil {
		return err
	}

	if err := r.ensure(&r.nd, len(x)*4); err != nil {
		return err
	}

	if err := r.check("rmsnorm", C.gguf_metal_rmsnorm(r.ctx, r.nx.h, r.nw.h, r.nd.h, C.uint32_t(len(x)), C.float(eps))); err != nil {
		return err
	}

	copy(dst, r.nd.floats(len(x)))

	return nil
}

func (r *mtl) ropeHeads(v []float32, nHeads, headDim, pos int, freqBase float32) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if err := r.opened(); err != nil {
		return err
	}

	// cos/sin from the float64 host reference: float32 cos/sin of large angles on the GPU would drift from the CPU path
	half := headDim / 2
	tab := make([]float32, 2*half)
	compute.RoPECosSin(tab[:half], tab[half:], headDim, pos, freqBase)

	if err := r.putFloats(&r.rope, tab); err != nil {
		return err
	}

	if err := r.putFloats(&r.ropeV, v); err != nil {
		return err
	}

	if err := r.check("rope_heads", C.gguf_metal_rope_heads(r.ctx, r.ropeV.h, r.rope.h, C.uint32_t(nHeads), C.uint32_t(headDim))); err != nil {
		return err
	}

	copy(v, r.ropeV.floats(len(v)))

	return nil
}

func (r *mtl) swiglu(gate, up []float32) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if err := r.opened(); err != nil {
		return err
	}

	if err := r.putFloats(&r.gate, gate); err != nil {
		return err
	}

	if err := r.putFloats(&r.up, up); err != nil {
		return err
	}

	if err := r.check("swiglu", C.gguf_metal_swiglu(r.ctx, r.gate.h, r.up.h, C.uint32_t(len(gate)))); err != nil {
		return err
	}

	copy(gate, r.gate.floats(len(gate)))

	return nil
}

func (r *mtl) attention(dst, q, k, v []float32, seqLen, nHeads, nKVHeads, headDim int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if err := r.opened(); err != nil {
		return err
	}

	if err := r.putFloats(&r.aq, q); err != nil {
		return err
	}

	if err := r.putFloats(&r.ak, k); err != nil {
		return err
	}

	if err := r.putFloats(&r.av, v); err != nil {
		return err
	}

	if err := r.ensure(&r.scores, nHeads*seqLen*4); err != nil {
		return err
	}

	if err := r.ensure(&r.ao, nHeads*headDim*4); err != nil {
		return err
	}

	rc := C.gguf_metal_attention(r.ctx, r.aq.h, r.ak.h, r.av.h, r.scores.h, r.ao.h, C.uint32_t(seqLen), C.uint32_t(nHeads), C.uint32_t(nKVHeads), C.uint32_t(headDim), C.uint32_t(seqLen))
	if err := r.check("attn_qk_softmax_v", rc); err != nil {
		return err
	}

	copy(dst, r.ao.floats(nHeads*headDim))

	return nil
}

func (r *mtl) kvFree() {
	for i := range r.kvK {
		r.free(&r.kvK[i])
		r.free(&r.kvV[i])
	}

	r.kvK, r.kvV, r.kvLen = nil, nil, nil
	r.kvLayers, r.kvMaxSeq, r.kvDim = 0, 0, 0
}

func (r *mtl) kvInit(layers, maxSeq, kvDim, _, _ int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if err := r.opened(); err != nil {
		return err
	}

	r.kvFree()

	size := maxSeq * kvDim * 4
	r.kvK = make([]devBuf, layers)
	r.kvV = make([]devBuf, layers)
	r.kvLen = make([]int, layers)
	r.kvLayers, r.kvMaxSeq, r.kvDim = layers, maxSeq, kvDim

	for i := range layers {
		if err := r.ensureExact(&r.kvK[i], size); err != nil {
			r.kvFree()
			return fmt.Errorf("metal: kv_init layer %d: %w", i, err)
		}

		if err := r.ensureExact(&r.kvV[i], size); err != nil {
			r.kvFree()
			return fmt.Errorf("metal: kv_init layer %d: %w", i, err)
		}
	}

	return nil
}

func (r *mtl) kvReset() {
	r.mu.Lock()
	defer r.mu.Unlock()

	for i := range r.kvLen {
		r.kvLen[i] = 0
	}
}

func (r *mtl) kvReady(layer int) error {
	if err := r.opened(); err != nil {
		return err
	}

	if r.kvLayers == 0 {
		return fmt.Errorf("metal: kv cache not initialized")
	}

	if layer < 0 || layer >= r.kvLayers {
		return fmt.Errorf("metal: kv layer %d out of range [0,%d)", layer, r.kvLayers)
	}

	return nil
}

func (r *mtl) kvAppend(layer, pos int, k, v []float32, n int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if err := r.kvReady(layer); err != nil {
		return err
	}

	if len(k) != n*r.kvDim || len(v) != n*r.kvDim {
		return fmt.Errorf("metal: kv_append layer=%d: len(k)=%d len(v)=%d, want n*kvDim=%d", layer, len(k), len(v), n*r.kvDim)
	}

	if pos < 0 || pos+n > r.kvMaxSeq {
		return fmt.Errorf("metal: kv_append layer=%d pos=%d n=%d beyond maxSeq=%d", layer, pos, n, r.kvMaxSeq)
	}

	// unified memory: kv_append is a plain copy into the MTLBuffer at the position offset (no kernel needed)
	off := pos * r.kvDim
	copy(r.kvK[layer].floats(r.kvMaxSeq * r.kvDim)[off:], k)
	copy(r.kvV[layer].floats(r.kvMaxSeq * r.kvDim)[off:], v)

	r.kvLen[layer] = max(r.kvLen[layer], pos+n)

	return nil
}

func (r *mtl) kvDropOldest(layer, n int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if err := r.kvReady(layer); err != nil {
		return err
	}

	have := r.kvLen[layer]
	if have == 0 {
		return nil
	}

	drop := min(n, have)
	keep := have - drop
	if keep > 0 {
		kv := r.kvDim
		ks := r.kvK[layer].floats(r.kvMaxSeq * kv)
		vs := r.kvV[layer].floats(r.kvMaxSeq * kv)
		copy(ks[:keep*kv], ks[drop*kv:have*kv])
		copy(vs[:keep*kv], vs[drop*kv:have*kv])
	}

	r.kvLen[layer] = keep

	return nil
}

func (r *mtl) kvRebuild(layer int, k, v []float32, n int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if err := r.kvReady(layer); err != nil {
		return err
	}

	if n > r.kvMaxSeq || len(k) < n*r.kvDim || len(v) < n*r.kvDim {
		return fmt.Errorf("metal: kv_rebuild layer=%d n=%d: k/v too short or beyond maxSeq=%d", layer, n, r.kvMaxSeq)
	}

	copy(r.kvK[layer].floats(r.kvMaxSeq*r.kvDim), k[:n*r.kvDim])
	copy(r.kvV[layer].floats(r.kvMaxSeq*r.kvDim), v[:n*r.kvDim])
	r.kvLen[layer] = n

	return nil
}

func (r *mtl) attentionKV(layer int, dst, q []float32, seqLen, nHeads, nKVHeads, headDim int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if err := r.kvReady(layer); err != nil {
		return err
	}

	if nKVHeads*headDim != r.kvDim {
		return fmt.Errorf("metal: AttentionScoresKV: nKVHeads*headDim=%d, kv cache kvDim=%d", nKVHeads*headDim, r.kvDim)
	}

	if seqLen > r.kvLen[layer] {
		return fmt.Errorf("metal: AttentionScoresKV layer=%d: seqLen=%d but only %d positions cached", layer, seqLen, r.kvLen[layer])
	}

	if err := r.putFloats(&r.aq, q); err != nil {
		return err
	}

	if err := r.ensure(&r.kvScores, nHeads*r.kvMaxSeq*4); err != nil {
		return err
	}

	if err := r.ensure(&r.ao, nHeads*headDim*4); err != nil {
		return err
	}

	rc := C.gguf_metal_attention(r.ctx, r.aq.h, r.kvK[layer].h, r.kvV[layer].h, r.kvScores.h, r.ao.h, C.uint32_t(seqLen), C.uint32_t(nHeads), C.uint32_t(nKVHeads), C.uint32_t(headDim), C.uint32_t(r.kvMaxSeq))
	if err := r.check("attn_qk_softmax_v", rc); err != nil {
		return err
	}

	copy(dst, r.ao.floats(nHeads*headDim))

	return nil
}

func (r *mtl) close() error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.ctx == nil {
		return nil
	}

	for _, cache := range []map[string]*devWeight{r.f32, r.q8, r.q4_0, r.q4_k} {
		for name, w := range cache {
			r.free(&w.buf)
			delete(cache, name)
		}
	}

	r.kvFree()
	r.xPtr, r.xLen = 0, 0

	for _, d := range []*devBuf{&r.vec, &r.out, &r.tmpW, &r.nx, &r.nw, &r.nd, &r.rope, &r.ropeV, &r.gate, &r.up, &r.aq, &r.ak, &r.av, &r.ao, &r.scores, &r.kvScores, &r.resid, &r.hn, &r.fq, &r.fo, &r.fg, &r.fu, &r.fattn, &r.mx, &r.mo, &r.bgate, &r.bup, &r.bout} {
		r.free(d)
	}

	C.gguf_metal_close(r.ctx)
	r.ctx = nil

	return nil
}
