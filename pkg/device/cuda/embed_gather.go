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

// EmbedGatherQ8_0 dequants token_embd row into d_resid (no host embedding HtoD).
func (b *Backend) EmbedGatherQ8_0(name string, raw []byte, tokenID, vocab, embd int) error {
	if !b.hasEmbedGather {
		return fmt.Errorf("cuda: embed_gather_q8_0 unavailable")
	}

	if tokenID < 0 || tokenID >= vocab || embd <= 0 || embd%32 != 0 {
		return fmt.Errorf("cuda: EmbedGatherQ8_0: bad token/embd")
	}

	want := vocab * (embd / 32) * 36
	if len(raw) < want {
		return fmt.Errorf("cuda: EmbedGatherQ8_0: raw too short")
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	gm, ok := b.matricesQ8[name]
	if !ok || gm.rows != vocab || gm.cols != embd || gm.bytes != len(raw) {
		if ok {
			C.gguf_cuda_matmul_pool_clear_graphs(&b.drv, &b.matmulPool)
			C.gguf_cuda_free(&b.drv, gm.ptr)
		}

		var ptr C.CUdeviceptr
		rc := C.gguf_cuda_upload_q8_0(&b.drv, b.ctx, &ptr, unsafe.Pointer(&raw[0]), C.size_t(len(raw)))
		if rc != 0 {
			return uploadFail("q8_0 emb", name, rc)
		}

		gm = gpuQ8Matrix{ptr: ptr, rows: vocab, cols: embd, bytes: len(raw)}
		b.matricesQ8[name] = gm
	}

	rc := C.gguf_cuda_embed_gather_q8_0(&b.drv, b.ctx, b.fnEmbedGather, &b.matmulPool, gm.ptr, C.int(tokenID), C.int(embd))
	if rc != 0 {
		b.hiddenResident = false
		return fmt.Errorf("cuda: embed_gather_q8_0: code %d", int(rc))
	}

	b.matmulPool.keep_resid_device = 1
	b.hiddenResident = true
	b.lastVecAddr = 0
	b.lastVecLen = 0

	return nil
}

// TryEmbedGather uploads token embedding on device when quant is supported.
func (b *Backend) TryEmbedGather(name string, raw []byte, t dtype.DType, tokenID, vocab, embd int) error {
	switch t {
	case dtype.Q8_0:
		return b.EmbedGatherQ8_0(name, raw, tokenID, vocab, embd)
	default:
		return fmt.Errorf("cuda: embed gather unsupported for %v", t)
	}
}
