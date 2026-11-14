//go:build metal && darwin && cgo

package metal

/*
#include "driver.h"
*/
import "C"

import (
	"fmt"
	"unsafe"

	"github.com/magomedcoder/gogguf/pkg/dtype"
	"github.com/magomedcoder/gogguf/pkg/quant"
)

// B2 runtime: device-resident residual (mtl.resid) and the fused decode path.
// Each fused call stages its host-side inputs (unified memory: plain copies), encodes the whole chain of B1 kernels into ONE command buffer (gguf_metal_batch_*), commits once and reads the results back, instead of one commit + wait per op.
// The residency flag lives in Backend (fused.go); this file only runs the kernels and reports whether a failed call may have modified resid.

func u32(n int) C.uint32_t {
	return C.uint32_t(n)
}

// hiddenReady: the library compiled with every kernel residency needs (open fails otherwise), so an opened runtime is always ready
func (r *mtl) hiddenReady() bool {
	r.mu.Lock()
	defer r.mu.Unlock()

	return r.ctx != nil
}

func (r *mtl) hiddenUpload(x []float32) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if err := r.opened(); err != nil {
		return err
	}

	if err := r.putFloats(&r.resid, x); err != nil {
		return err
	}

	r.fattnLen = 0

	return nil
}

func (r *mtl) hiddenDownload(dst []float32) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if err := r.opened(); err != nil {
		return err
	}

	if r.resid.h == nil || r.resid.n < len(dst)*4 {
		return fmt.Errorf("metal: hidden_download: no resident hidden of %d floats", len(dst))
	}

	copy(dst, r.resid.floats(len(dst)))
	r.fattnLen = 0

	return nil
}

// batch encodes one command buffer.
// committed reports that the GPU work was submitted: a returned error then may have left the written buffers modified; when committed is false nothing ran.
func (r *mtl) batch(encode func() error) (committed bool, err error) {
	if err := r.check("batch_begin", C.gguf_metal_batch_begin(r.ctx)); err != nil {
		return false, err
	}

	if err := encode(); err != nil {
		C.gguf_metal_batch_abort(r.ctx)
		return false, err
	}

	return true, r.check("batch_end", C.gguf_metal_batch_end(r.ctx))
}

func (r *mtl) encMat(kind C.int, w C.gguf_metal_buf_t, x *devBuf, xOff int, out *devBuf, outOff, rows, cols int) error {
	return r.check("matmul", C.gguf_metal_matmul_off(r.ctx, kind, w, x.h, C.size_t(xOff), out.h, C.size_t(outOff), u32(rows), u32(cols)))
}

func (r *mtl) encRMS(x *devBuf, xOff int, w C.gguf_metal_buf_t, dst *devBuf, dstOff, dim, rows int, eps float32) error {
	return r.check("rmsnorm", C.gguf_metal_rmsnorm_rows(r.ctx, x.h, C.size_t(xOff), w, dst.h, C.size_t(dstOff), u32(dim), u32(rows), C.float(eps)))
}

func (r *mtl) encRoPE(v *devBuf, vOff int, nHeads, headDim int) error {
	return r.check("rope_heads", C.gguf_metal_rope_heads_off(r.ctx, v.h, C.size_t(vOff), r.rope.h, u32(nHeads), u32(headDim)))
}

func (r *mtl) encAdd(a, b *devBuf, n int) error {
	return r.check("add", C.gguf_metal_add(r.ctx, a.h, b.h, u32(n)))
}

// weightQ8 returns the cached device Q8_0 matrix (fused paths always pass a name: the temp upload buffer would be clobbered inside one batch)
func (r *mtl) weightQ8(name string, raw []byte, rows, cols int) (C.gguf_metal_buf_t, error) {
	if name == "" {
		return nil, fmt.Errorf("metal: fused weight needs a cache name")
	}

	return r.weight(r.q8, name, unsafe.Pointer(&raw[0]), rows*(cols/quant.QK8_0)*quant.BlockQ8_0Size, rows, cols)
}

// weightVec returns a cached device FP32 vector (norm weights)
func (r *mtl) weightVec(name string, v []float32, n int) (C.gguf_metal_buf_t, error) {
	if name == "" {
		return nil, fmt.Errorf("metal: fused weight needs a cache name")
	}

	return r.weight(r.f32, name, unsafe.Pointer(&v[0]), n*4, n, 1)
}

func (r *mtl) qkvRoPEAttention(a *qkvArgs, resident bool) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if err := r.kvReady(a.layer); err != nil {
		return err
	}

	// the attention output of a previous call is no longer valid whatever happens below
	r.fattnLen = 0

	qDim, kvDim := a.nHeads*a.headDim, a.nKVHeads*a.headDim
	if kvDim != r.kvDim {
		return fmt.Errorf("metal: qkv_rope_attn: nKVHeads*headDim=%d, kv cache kvDim=%d", kvDim, r.kvDim)
	}

	if a.kvPos+1 > r.kvMaxSeq || a.seqLen > max(r.kvLen[a.layer], a.kvPos+1) {
		return fmt.Errorf("metal: qkv_rope_attn layer=%d: kvPos=%d seqLen=%d beyond cached %d / maxSeq %d", a.layer, a.kvPos, a.seqLen, r.kvLen[a.layer], r.kvMaxSeq)
	}

	if resident && r.resid.n < a.embd*4 {
		return fmt.Errorf("metal: qkv_rope_attn: resident hidden buffer holds < %d floats", a.embd)
	}

	qw, err := r.weightQ8(a.qName, a.qRaw, qDim, a.embd)
	if err != nil {
		return err
	}

	kw, err := r.weightQ8(a.kName, a.kRaw, kvDim, a.embd)
	if err != nil {
		return err
	}

	vw, err := r.weightQ8(a.vName, a.vRaw, kvDim, a.embd)
	if err != nil {
		return err
	}

	var qn, kn, an C.gguf_metal_buf_t
	if a.qNormName != "" {
		if qn, err = r.weightVec(a.qNormName, a.qNorm, a.headDim); err != nil {
			return err
		}
	}

	if a.kNormName != "" {
		if kn, err = r.weightVec(a.kNormName, a.kNorm, a.headDim); err != nil {
			return err
		}
	}

	if resident {
		if an, err = r.weightVec(a.attnNormName, a.attnNorm, a.embd); err != nil {
			return err
		}
	} else if err := r.putFloats(&r.hn, a.h[:a.embd]); err != nil {
		return err
	}

	// cos/sin tables from the host (float64 reference, optionally LongRoPE-scaled): the kernel only rotates pairs
	half := a.headDim / 2
	tab := make([]float32, 2*half)
	copy(tab[:half], a.cos[:half])
	copy(tab[half:], a.sin[:half])

	if err := r.putFloats(&r.rope, tab); err != nil {
		return err
	}

	for _, e := range []struct {
		d *devBuf
		n int
	}{{&r.hn, a.embd * 4}, {&r.fq, qDim * 4}, {&r.fattn, qDim * 4}, {&r.kvScores, a.nHeads * r.kvMaxSeq * 4}} {
		if err := r.ensure(e.d, e.n); err != nil {
			return err
		}
	}

	// K and V are produced straight into the KV buffers at the append position (kv_append fused into the matvec output offset)
	kbuf, vbuf := &r.kvK[a.layer], &r.kvV[a.layer]
	off := a.kvPos * kvDim * 4

	_, err = r.batch(func() error {
		if resident {
			if err := r.encRMS(&r.resid, 0, an, &r.hn, 0, a.embd, 1, a.eps); err != nil {
				return err
			}
		}

		if err := r.encMat(C.GGUF_METAL_MAT_Q8_0, qw, &r.hn, 0, &r.fq, 0, qDim, a.embd); err != nil {
			return err
		}

		if err := r.encMat(C.GGUF_METAL_MAT_Q8_0, kw, &r.hn, 0, kbuf, off, kvDim, a.embd); err != nil {
			return err
		}

		if err := r.encMat(C.GGUF_METAL_MAT_Q8_0, vw, &r.hn, 0, vbuf, off, kvDim, a.embd); err != nil {
			return err
		}

		if qn != nil {
			if err := r.encRMS(&r.fq, 0, qn, &r.fq, 0, a.headDim, a.nHeads, a.eps); err != nil {
				return err
			}
		}

		if kn != nil {
			if err := r.encRMS(kbuf, off, kn, kbuf, off, a.headDim, a.nKVHeads, a.eps); err != nil {
				return err
			}
		}

		if err := r.encRoPE(&r.fq, 0, a.nHeads, a.headDim); err != nil {
			return err
		}

		if err := r.encRoPE(kbuf, off, a.nKVHeads, a.headDim); err != nil {
			return err
		}

		return r.check("attn_qk_softmax_v", C.gguf_metal_attention(r.ctx, r.fq.h, kbuf.h, vbuf.h, r.kvScores.h, r.fattn.h, u32(a.seqLen), u32(a.nHeads), u32(a.nKVHeads), u32(a.headDim), u32(r.kvMaxSeq)))
	})
	if err != nil {
		return err
	}

	r.kvLen[a.layer] = max(r.kvLen[a.layer], a.kvPos+1)
	r.fattnLen = qDim

	// optional host mirrors
	if len(a.kOut) >= kvDim {
		copy(a.kOut[:kvDim], kbuf.floats(r.kvMaxSeq * kvDim)[a.kvPos*kvDim:][:kvDim])
	}

	if len(a.vOut) >= kvDim {
		copy(a.vOut[:kvDim], vbuf.floats(r.kvMaxSeq * kvDim)[a.kvPos*kvDim:][:kvDim])
	}

	if len(a.attn) >= qDim {
		copy(a.attn[:qDim], r.fattn.floats(qDim))
	}

	return nil
}

func (r *mtl) attnFFNResidual(a *attnFFNArgs, resident bool) (dirty bool, err error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if err := r.opened(); err != nil {
		return false, err
	}

	// the device attention output is consumed by this call (success or not)
	haveAttn := r.fattnLen == a.attnDim
	r.fattnLen = 0

	if len(a.attn) >= a.attnDim {
		if err := r.putFloats(&r.fattn, a.attn[:a.attnDim]); err != nil {
			return false, err
		}
	} else if !haveAttn {
		return false, fmt.Errorf("metal: attn_ffn_residual: no attention output on the device (attn == nil without a preceding QKVRoPEAttentionQuantCached)")
	}

	if resident {
		if r.resid.n < a.embd*4 {
			return false, fmt.Errorf("metal: attn_ffn_residual: resident hidden buffer holds < %d floats", a.embd)
		}
	} else {
		// host x is the residual: stage it in resid (residency is not active, so nothing is overwritten)
		if err := r.putFloats(&r.resid, a.x[:a.embd]); err != nil {
			return false, err
		}
	}

	wo, err := r.weightQ8(a.woName, a.woRaw, a.embd, a.attnDim)
	if err != nil {
		return false, err
	}

	nw, err := r.weightVec(a.ffnNormName, a.ffnNorm, a.embd)
	if err != nil {
		return false, err
	}

	gate, err := r.weightQ8(a.gateName, a.gateRaw, a.ffn, a.embd)
	if err != nil {
		return false, err
	}

	up, err := r.weightQ8(a.upName, a.upRaw, a.ffn, a.embd)
	if err != nil {
		return false, err
	}

	down, err := r.weightQ8(a.downName, a.downRaw, a.embd, a.ffn)
	if err != nil {
		return false, err
	}

	for _, e := range []struct {
		d *devBuf
		n int
	}{{&r.hn, a.embd * 4}, {&r.fo, a.embd * 4}, {&r.fg, a.ffn * 4}, {&r.fu, a.ffn * 4}} {
		if err := r.ensure(e.d, e.n); err != nil {
			return false, err
		}
	}

	committed, err := r.batch(func() error {
		if err := r.encMat(C.GGUF_METAL_MAT_Q8_0, wo, &r.fattn, 0, &r.fo, 0, a.embd, a.attnDim); err != nil {
			return err
		}

		if err := r.encAdd(&r.resid, &r.fo, a.embd); err != nil {
			return err
		}

		if err := r.encRMS(&r.resid, 0, nw, &r.hn, 0, a.embd, 1, a.eps); err != nil {
			return err
		}

		if err := r.encMat(C.GGUF_METAL_MAT_Q8_0, gate, &r.hn, 0, &r.fg, 0, a.ffn, a.embd); err != nil {
			return err
		}

		if err := r.encMat(C.GGUF_METAL_MAT_Q8_0, up, &r.hn, 0, &r.fu, 0, a.ffn, a.embd); err != nil {
			return err
		}

		if err := r.check("swiglu", C.gguf_metal_swiglu(r.ctx, r.fg.h, r.fu.h, u32(a.ffn))); err != nil {
			return err
		}

		if err := r.encMat(C.GGUF_METAL_MAT_Q8_0, down, &r.fg, 0, &r.fo, 0, a.embd, a.ffn); err != nil {
			return err
		}

		return r.encAdd(&r.resid, &r.fo, a.embd)
	})
	if err != nil {
		// committed: the GPU may have applied the first residual add; not committed: resid is untouched
		return committed, err
	}

	if !resident {
		copy(a.x[:a.embd], r.resid.floats(a.embd))
	}

	return false, nil
}

func (r *mtl) ffnSwiGLU(a *ffnArgs) error {
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

	if err := r.putFloats(&r.hn, a.x); err != nil {
		return err
	}

	for _, e := range []struct {
		d *devBuf
		n int
	}{{&r.fo, a.embd * 4}, {&r.fg, a.ffn * 4}, {&r.fu, a.ffn * 4}} {
		if err := r.ensure(e.d, e.n); err != nil {
			return err
		}
	}

	if _, err := r.batch(func() error {
		if err := r.encMat(C.GGUF_METAL_MAT_Q8_0, gate, &r.hn, 0, &r.fg, 0, a.ffn, a.embd); err != nil {
			return err
		}

		if err := r.encMat(C.GGUF_METAL_MAT_Q8_0, up, &r.hn, 0, &r.fu, 0, a.ffn, a.embd); err != nil {
			return err
		}

		if err := r.check("swiglu", C.gguf_metal_swiglu(r.ctx, r.fg.h, r.fu.h, u32(a.ffn))); err != nil {
			return err
		}

		return r.encMat(C.GGUF_METAL_MAT_Q8_0, down, &r.fg, 0, &r.fo, 0, a.embd, a.ffn)
	}); err != nil {
		return err
	}

	copy(a.out, r.fo.floats(a.embd))

	return nil
}

func (r *mtl) logitsFromResident(a *logitsArgs) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if err := r.opened(); err != nil {
		return err
	}

	if r.resid.n < a.embd*4 {
		return fmt.Errorf("metal: logits: resident hidden buffer holds < %d floats", a.embd)
	}

	nw, err := r.weightVec(a.normName, a.norm, a.embd)
	if err != nil {
		return err
	}

	var head C.gguf_metal_buf_t
	var kind C.int
	if a.t == dtype.F32 || a.t == dtype.F16 {
		kind = C.GGUF_METAL_MAT_F32
		head, err = r.weight(r.f32, a.headName, unsafe.Pointer(&a.headF32[0]), a.vocab*a.embd*4, a.vocab, a.embd)
	} else {
		var qk, bs int
		var cache map[string]*devWeight
		if kind, qk, bs, cache, err = r.quantKind(a.t); err != nil {
			return err
		}

		head, err = r.weight(cache, a.headName, unsafe.Pointer(&a.headRaw[0]), a.vocab*(a.embd/qk)*bs, a.vocab, a.embd)
	}

	if err != nil {
		return err
	}

	if err := r.ensure(&r.hn, a.embd*4); err != nil {
		return err
	}

	if err := r.ensure(&r.out, a.vocab*4); err != nil {
		return err
	}

	if _, err := r.batch(func() error {
		if err := r.encRMS(&r.resid, 0, nw, &r.hn, 0, a.embd, 1, a.eps); err != nil {
			return err
		}

		return r.encMat(kind, head, &r.hn, 0, &r.out, 0, a.vocab, a.embd)
	}); err != nil {
		return err
	}

	copy(a.logits, r.out.floats(a.vocab))

	return nil
}
