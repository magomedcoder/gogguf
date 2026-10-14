//go:build cuda

package cuda

/*
#include "driver.h"
*/
import "C"

import (
	"fmt"
	"unsafe"

	"github.com/magomedcoder/gogguf/pkg/device/mlaspec"
)

// Device MLA (DeepSeek-V2/V3 absorbed attention)
//
//   - Latent KV cache (gguf_mla_cache_t): per layer K [maxSeq][kvLora+rope], V [maxSeq][kvLora] (same widths as the host MLACache; separate from the GQA KVCache).
//   - mla_attn kernel (one block per head): q_nope absorb through k_b, QK over latent K with online softmax, weighted latent V, projection through v_b - one launch for all heads.
//   - k_b / v_b are uploaded once as FP32 device matrices (weight cache + LRU, see weight_cache.go) via a loader that is only invoked on a cache miss, so the hot path does not call Store.Floats.
//
// Host keeps: MLA projections (matmul on GPU through MatMulInto), YaRN/partial RoPE of q_pe and k_pe (CPU, applied before the call), RMSNorms. Limits: kvLora <= 1024, kvLora+rope <= 1280, seqLen <= maxSeq.

// HasMLA reports whether the mla_attn kernel is loaded.
func (b *Backend) HasMLA() bool { return b.hasMLA }

func (b *Backend) MLACacheInit(layers, maxSeq int, p mlaspec.Params) error {
	if !b.hasMLA {
		return fmt.Errorf("cuda: mla_attn kernel unavailable")
	}

	if layers <= 0 || maxSeq <= 0 || p.NHeads <= 0 || p.Nope <= 0 || p.KVLora <= 0 || p.Rope < 0 || p.VHead <= 0 {
		return fmt.Errorf("cuda: MLACacheInit: invalid sizes")
	}

	if p.KVLora > 1024 || p.KDim() > 1280 {
		return fmt.Errorf("cuda: MLACacheInit: kvLora=%d rope=%d exceed kernel limits", p.KVLora, p.Rope)
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	if b.mlaReady {
		C.gguf_cuda_mla_free(&b.drv, &b.mlaCache)
		b.mlaReady = false
	}

	rc := C.gguf_cuda_mla_init(&b.drv, b.ctx, &b.mlaCache, C.int(layers), C.int(maxSeq), C.int(p.KDim()), C.int(p.KVLora), C.int(p.QDim()), C.int(p.OutDim()))
	if rc != 0 {
		return fmt.Errorf("cuda: mla_init: code %d", int(rc))
	}

	b.mlaReady = true
	return nil
}

func (b *Backend) MLACacheAppend(layer, pos int, k, v []float32) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	if !b.mlaReady {
		return fmt.Errorf("cuda: mla cache not initialized")
	}

	if len(k) < int(b.mlaCache.k_dim) || len(v) < int(b.mlaCache.v_dim) {
		return fmt.Errorf("cuda: MLACacheAppend: k/v too short")
	}

	rc := C.gguf_cuda_mla_append(&b.drv, b.ctx, &b.mlaCache, C.int(layer), C.int(pos), (*C.float)(unsafe.Pointer(&k[0])), (*C.float)(unsafe.Pointer(&v[0])))
	if rc != 0 {
		return fmt.Errorf("cuda: mla_append layer=%d pos=%d: code %d", layer, pos, int(rc))
	}

	return nil
}

// ensureFP32Lazy returns the device copy of an FP32 matrix, loading host data only on a miss.
func (b *Backend) ensureFP32Lazy(name string, rows, cols int, load mlaspec.WeightLoader) (gpuMatrix, error) {
	if gm, ok := b.matrices[name]; ok && gm.rows == rows && gm.cols == cols {
		b.touchWeight(kindF32, name)
		return gm, nil
	}

	if load == nil {
		return gpuMatrix{}, fmt.Errorf("cuda: %q not resident and no loader", name)
	}

	host, err := load(name)
	if err != nil {
		return gpuMatrix{}, err
	}

	if len(host) < rows*cols {
		return gpuMatrix{}, fmt.Errorf("cuda: %q: %d floats, need %d", name, len(host), rows*cols)
	}

	return b.ensureFP32Matrix(name, host, rows, cols)
}

func (b *Backend) MLAAttention(layer int, dst, q []float32, seqLen int, p mlaspec.Params, kbName, vbName string, load mlaspec.WeightLoader) error {
	if !b.hasMLA {
		return fmt.Errorf("cuda: mla_attn kernel unavailable")
	}

	if len(dst) < p.OutDim() || len(q) < p.QDim() {
		return fmt.Errorf("cuda: MLAAttention: q/dst too short")
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	if !b.mlaReady {
		return fmt.Errorf("cuda: mla cache not initialized")
	}

	kb, err := b.ensureFP32Lazy(kbName, p.NHeads*p.KVLora, p.Nope, load)
	if err != nil {
		return err
	}

	vb, err := b.ensureFP32Lazy(vbName, p.NHeads*p.VHead, p.KVLora, load)
	if err != nil {
		return err
	}

	rc := C.gguf_cuda_mla_attention(
		&b.drv,
		b.ctx,
		b.fnMLA,
		&b.mlaCache,
		C.int(layer),
		(*C.float)(unsafe.Pointer(&dst[0])),
		(*C.float)(unsafe.Pointer(&q[0])),
		kb.ptr, vb.ptr,
		C.int(seqLen),
		C.int(p.NHeads),
		C.int(p.Nope),
		C.int(p.KVLora),
		C.int(p.Rope),
		C.int(p.VHead),
		C.float(p.Scale),
	)
	if rc != 0 {
		return fmt.Errorf("cuda: mla_attention layer=%d: code %d", layer, int(rc))
	}

	return nil
}
