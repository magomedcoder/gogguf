//go:build metal && darwin && cgo

package metal

/*
#include "driver.h"
*/
import "C"

import (
	"fmt"
	"math"
	"unsafe"

	"github.com/magomedcoder/gogguf/pkg/quant"
)

// B3 runtime: batched Q8_0 GEMM, batch FFN and batch attention (contracts in batch.go).
// Each call stages its host inputs (unified memory: plain copies), encodes every launch into ONE command buffer (gguf_metal_batch_*), commits once and copies the result back.

// stageX makes mx hold x.
// While the same host slice (pointer + length) feeds consecutive calls (Q -> K -> V, gate -> up) the copy is skipped, like CUDA skip_vec_htod.
// Unlike the CUDA flag this also compares a sample of elements (first, last and a stride in between) against the staged copy: callers such as the prefill loop rewrite the same scratch slice in place (RMSNorm into h) between layers, and a stale reuse would be silent.
// A rewritten vector differs in the samples, so it is re-staged.
func (r *mtl) stageX(x []float32) error {
	p := uintptr(unsafe.Pointer(unsafe.SliceData(x)))
	if p == r.xPtr && len(x) == r.xLen && r.mx.h != nil && r.mx.n >= len(x)*4 && r.sameStaged(x) {
		return nil
	}

	r.xPtr, r.xLen = 0, 0

	if err := r.putFloats(&r.mx, x); err != nil {
		return err
	}

	r.xPtr, r.xLen = p, len(x)

	return nil
}

// sameStaged compares up to ~64 evenly spread elements of x (always including the first and the last) with the staged copy in mx
func (r *mtl) sameStaged(x []float32) bool {
	if len(x) == 0 {
		return true
	}

	dev := r.mx.floats(len(x))
	step := max(1, len(x)/64)
	for i := 0; i < len(x); i += step {
		if math.Float32bits(x[i]) != math.Float32bits(dev[i]) {
			return false
		}
	}

	last := len(x) - 1

	return math.Float32bits(x[last]) == math.Float32bits(dev[last])
}

// dropStagedX forgets the staged host slice (its content changed or may have: the output aliased the input)
func (r *mtl) dropStagedX() {
	r.xPtr, r.xLen = 0, 0
}

func (r *mtl) encMatMat(w C.gguf_metal_buf_t, x *devBuf, out *devBuf, rows, cols, batch int) error {
	return r.check("matmul_mat_q8_0", C.gguf_metal_matmul_mat_q8_0(r.ctx, w, x.h, 0, out.h, 0, u32(rows), u32(cols), u32(batch)))
}

func (r *mtl) matMulMatQ8_0(name string, raw []byte, rows, cols int, x []float32, batch int, out []float32) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if err := r.opened(); err != nil {
		return err
	}

	w, err := r.weight(r.q8, name, unsafe.Pointer(&raw[0]), rows*(cols/quant.QK8_0)*quant.BlockQ8_0Size, rows, cols)
	if err != nil {
		return err
	}

	if err := r.stageX(x[:batch*cols]); err != nil {
		return err
	}

	if err := r.ensure(&r.mo, batch*rows*4); err != nil {
		return err
	}

	if err := r.encMatMat(w, &r.mx, &r.mo, rows, cols, batch); err != nil {
		return err
	}

	copy(out[:batch*rows], r.mo.floats(batch*rows))

	// the output slice may share memory with x: the staged copy is then stale
	if len(out) > 0 && uintptr(unsafe.Pointer(unsafe.SliceData(out))) == r.xPtr {
		r.dropStagedX()
	}

	return nil
}

func (r *mtl) ffnSwiGLUMatQ8_0(a *ffnMatArgs) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if err := r.opened(); err != nil {
		return err
	}

	gate, err := r.weightQ8(a.gateName, a.gateRaw, a.ffn, a.embd)
	if err != nil {
		return err
	}

	up, err := r.weightQ8(a.upName, a.upRaw, a.ffn, a.embd)
	if err != nil {
		return err
	}

	down, err := r.weightQ8(a.downName, a.downRaw, a.embd, a.ffn)
	if err != nil {
		return err
	}

	if err := r.stageX(a.x[:a.batch*a.embd]); err != nil {
		return err
	}

	for _, e := range []struct {
		d *devBuf
		n int
	}{{&r.bgate, a.batch * a.ffn * 4}, {&r.bup, a.batch * a.ffn * 4}, {&r.bout, a.batch * a.embd * 4}} {
		if err := r.ensure(e.d, e.n); err != nil {
			return err
		}
	}

	// gate GEMM -> up GEMM -> SwiGLU over the whole [batch][ffn] -> down GEMM, one command buffer
	if _, err := r.batch(func() error {
		if err := r.encMatMat(gate, &r.mx, &r.bgate, a.ffn, a.embd, a.batch); err != nil {
			return err
		}

		if err := r.encMatMat(up, &r.mx, &r.bup, a.ffn, a.embd, a.batch); err != nil {
			return err
		}

		if err := r.check("swiglu", C.gguf_metal_swiglu(r.ctx, r.bgate.h, r.bup.h, u32(a.batch*a.ffn))); err != nil {
			return err
		}

		return r.encMatMat(down, &r.bgate, &r.bout, a.embd, a.ffn, a.batch)
	}); err != nil {
		return err
	}

	copy(a.out[:a.batch*a.embd], r.bout.floats(a.batch*a.embd))

	// x and out may alias (in-place FFN): the staged input is stale after the write-back
	if len(a.out) > 0 && uintptr(unsafe.Pointer(unsafe.SliceData(a.out))) == r.xPtr {
		r.dropStagedX()
	}

	return nil
}

func (r *mtl) attentionKVBatch(layer int, dst, q []float32, n, pastLen, nHeads, nKVHeads, headDim int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if err := r.kvReady(layer); err != nil {
		return err
	}

	if nKVHeads*headDim != r.kvDim {
		return fmt.Errorf("metal: AttentionScoresKVBatch: nKVHeads*headDim=%d, kv cache kvDim=%d", nKVHeads*headDim, r.kvDim)
	}

	if pastLen+n > r.kvLen[layer] {
		return fmt.Errorf("metal: AttentionScoresKVBatch layer=%d: pastLen=%d n=%d but only %d positions cached", layer, pastLen, n, r.kvLen[layer])
	}

	qDim := nHeads * headDim
	if err := r.putFloats(&r.aq, q[:n*qDim]); err != nil {
		return err
	}

	if err := r.ensure(&r.kvScores, nHeads*r.kvMaxSeq*4); err != nil {
		return err
	}

	if err := r.ensure(&r.ao, n*qDim*4); err != nil {
		return err
	}

	// one launch per query row (causal seqLen = pastLen+i+1), all in one command buffer; the serial encoder orders the launches that share the scores scratch
	kbuf, vbuf := &r.kvK[layer], &r.kvV[layer]
	if _, err := r.batch(func() error {
		for i := range n {
			off := C.size_t(i * qDim * 4)
			rc := C.gguf_metal_attention_off(r.ctx, r.aq.h, off, kbuf.h, vbuf.h, r.kvScores.h, r.ao.h, off, u32(pastLen+i+1), u32(nHeads), u32(nKVHeads), u32(headDim), u32(r.kvMaxSeq))
			if err := r.check("attn_qk_softmax_v", rc); err != nil {
				return err
			}
		}

		return nil
	}); err != nil {
		return err
	}

	copy(dst[:n*qDim], r.ao.floats(n*qDim))

	return nil
}
