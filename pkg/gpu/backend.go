package gpu

import (
	"errors"

	"github.com/magomedcoder/gogguf/pkg/format"
	"github.com/magomedcoder/gogguf/pkg/ops"
)

// RoPEMode - RoPE mode in fused QKV path (NeoX for Qwen/Mistral/Gemma, NORM for Llama)
type RoPEMode = ops.RoPEMode

const (
	RoPENeoX = ops.RoPENeoX
	RoPENorm = ops.RoPENorm
)

// ErrKVCacheUnavailable means the backend does not support GPU KV-cache
var ErrKVCacheUnavailable = errors.New("gpu: kv cache unavailable")

// ErrHiddenUnavailable means the backend cannot keep hidden on device
var ErrHiddenUnavailable = errors.New("gpu: device-resident hidden unavailable")

// FusedQuantSupported reports whether fused paths (FFN/QKV/residual/lm_head) support this weight type natively without dequantizing the full matrix to FP32
func FusedQuantSupported(t format.GGML) bool {
	switch t {
	case format.GgmlQ8_0, format.GgmlQ4_0, format.GgmlQ4_K, format.GgmlQ5_K, format.GgmlQ6_K:
		return true
	default:
		return false
	}
}

// Backend runs compute on GPU (CUDA)
type Backend interface {
	// Name returns device name, e.g. "CUDA:0 NVIDIA ..."
	Name() string

	// VRAMInfo returns used and total device VRAM in bytes.
	// Backend without VRAM (CPU) returns 0, 0, nil
	VRAMInfo() (used, total uint64, err error)

	// MatMulVec multiplies matrix[rows*cols] by vec[cols]
	MatMulVec(matrix []float32, rows, cols int, vec []float32) ([]float32, error)

	// MatMulVecCached like MatMulVec, but matrix is uploaded to GPU once by name
	MatMulVecCached(name string, matrix []float32, rows, cols int, vec []float32) ([]float32, error)

	// MatMulVecQ8_0Cached matmul Q8_0 matrix without dequant to FP32
	MatMulVecQ8_0Cached(name string, raw []byte, rows, cols int, vec []float32) ([]float32, error)

	// MatMulVecQ4_0Cached matmul Q4_0 matrix without full dequant to FP32
	MatMulVecQ4_0Cached(name string, raw []byte, rows, cols int, vec []float32) ([]float32, error)

	// MatMulVecQ4_KCached matmul Q4_K matrix (K-quant)
	MatMulVecQ4_KCached(name string, raw []byte, rows, cols int, vec []float32) ([]float32, error)

	// MatMulVecQ5_KCached matmul Q5_K matrix (K-quant)
	MatMulVecQ5_KCached(name string, raw []byte, rows, cols int, vec []float32) ([]float32, error)

	// MatMulVecQ6_KCached matmul Q6_K matrix (K-quant)
	MatMulVecQ6_KCached(name string, raw []byte, rows, cols int, vec []float32) ([]float32, error)

	// RMSNormInto writes RMS normalization to dst (GPU or CPU)
	RMSNormInto(dst, x, weight []float32, eps float32) error

	// ApplyRoPEHeads applies NeoX/Qwen RoPE to nHeads heads in v (in-place)
	ApplyRoPEHeads(v []float32, nHeads, headDim, pos int, freqBase float32) error

	// ApplyRoPEHeadsNorm applies Llama RoPE (adjacent dim pairs) to nHeads heads in v
	ApplyRoPEHeadsNorm(v []float32, nHeads, headDim, pos int, freqBase float32) error

	// SwiGLUInPlace computes silu(gate)*up, result in gate
	SwiGLUInPlace(gate, up []float32) error

	// FFNSwiGLUCached gate/up/down matmul + SwiGLU; on CUDA activations stay on GPU
	FFNSwiGLUCached(gateName, upName, downName string, gateW, upW, downW, x, out []float32, embd, ffn int) error

	// FFNSwiGLUQ8_0Cached same for Q8_0 weights
	FFNSwiGLUQ8_0Cached(gateName, upName, downName string, gateRaw, upRaw, downRaw []byte, x, out []float32, embd, ffn int) error

	// AttnFFNResidualCached WO(attn)+residual+RMSNorm+FFN+residual on GPU
	AttnFFNResidualCached(woName, ffnNormName, gateName, upName, downName string, woW, ffnNorm, gateW, upW, downW, x, attn []float32, embd, attnDim, ffn int, eps float32) error

	// AttnFFNResidualQ8_0Cached same for Q8_0 matmul weights (norm is FP32)
	AttnFFNResidualQ8_0Cached(woName, ffnNormName, gateName, upName, downName string, woRaw, gateRaw, upRaw, downRaw []byte, ffnNorm, x, attn []float32, embd, attnDim, ffn int, eps float32) error

	// QKVRoPEAttentionCached: h->QKV->head RMSNorm->RoPE->KV append->attention on GPU
	QKVRoPEAttentionCached(qName, kName, vName, qNormName, kNormName string, qW, kW, vW, qNorm, kNorm, h, cos, sin, attn, kOut, vOut []float32, embd, nHeads, nKVHeads, headDim, layer, kvPos, seqLen int, eps float32) error

	// QKVRoPEAttentionQ8_0Cached same for Q8_0 matmul weights (norm is FP32)
	QKVRoPEAttentionQ8_0Cached(qName, kName, vName, qNormName, kNormName string, qRaw, kRaw, vRaw []byte, qNorm, kNorm, h, cos, sin, attn, kOut, vOut []float32, embd, nHeads, nKVHeads, headDim, layer, kvPos, seqLen int, eps float32) error

	// FFNSwiGLUQuantCached FFN SwiGLU for Q8_0/Q4_0/Q4_K/Q5_K/Q6_K weights
	FFNSwiGLUQuantCached(t format.GGML, gateName, upName, downName string, gateRaw, upRaw, downRaw []byte, x, out []float32, embd, ffn int) error

	// FFNGeGLUQuantCached FFN GeGLU (Gemma) for the same weight types
	FFNGeGLUQuantCached(t format.GGML, gateName, upName, downName string, gateRaw, upRaw, downRaw []byte, x, out []float32, embd, ffn int) error

	// AttnFFNResidualQuantCached WO+residual+RMSNorm+FFN+residual for quantized weights.
	// If hidden resident (HiddenUpload), x is not uploaded to GPU and stays on device: host buffer x is not updated in that case.
	AttnFFNResidualQuantCached(t format.GGML, woName, ffnNormName, gateName, upName, downName string, woRaw, gateRaw, upRaw, downRaw []byte, ffnNorm, x, attn []float32, embd, attnDim, ffn int, eps float32) error

	// QKVRoPEAttentionQuantCached QKV+RoPE+attn for quantized weights.
	// If hidden resident, h computed on GPU as RMSNorm(resid, attnNorm) and host h is not read.
	// mode - RoPE layout; empty qNormName/kNormName - layer without QK-norm (Llama / Mistral).
	QKVRoPEAttentionQuantCached(t format.GGML, mode RoPEMode, qName, kName, vName, qNormName, kNormName, attnNormName string, qRaw, kRaw, vRaw []byte, qNorm, kNorm, attnNorm, h, cos, sin, attn, kOut, vOut []float32, embd, nHeads, nKVHeads, headDim, layer, kvPos, seqLen int, eps float32) error

	// HiddenResident reports whether the backend supports device-resident hidden state
	HiddenResident() bool

	// HiddenActive reports whether the current hidden state is on device.
	// After fused layer error: true - hidden intact (can HiddenDownload and fall back to CPU), false - residual partially modified, no valid hidden anywhere.
	HiddenActive() bool

	// HiddenUpload puts hidden state on device (one HtoD per token)
	HiddenUpload(x []float32) error

	// HiddenDownload fetches device-resident hidden state into dst and disables residency
	HiddenDownload(dst []float32) error

	// LogitsFromDevice computes RMSNorm(resident hidden) + lm_head on GPU; only logits are copied to host. Use headF32 for FP32 head, else headRaw.
	LogitsFromDevice(t format.GGML, normName string, norm []float32, headName string, headRaw []byte, headF32, logits []float32, vocab, embd int, eps float32) error

	// AttentionScoresInto writes scaled dot-product attention to dst
	AttentionScoresInto(dst, q, k, v, scores []float32, seqLen, nHeads, nKVHeads, headDim int) error

	// KVCacheInit allocates GPU K/V buffers for offloaded layers
	KVCacheInit(layers, maxSeq, kvDim, nHeads, headDim int) error

	// KVCacheReset resets GPU KV-cache
	KVCacheReset()

	// KVCacheAppend adds K/V for one token at position pos
	KVCacheAppend(layer, pos int, k, v []float32) error

	// KVCacheAppendN adds K/V for n tokens starting at position pos (batch prefill)
	KVCacheAppendN(layer, pos int, k, v []float32, n int) error

	// AttentionScoresKV attention with K/V from GPU KV-cache
	AttentionScoresKV(layer int, dst, q []float32, seqLen, nHeads, nKVHeads, headDim int) error

	Close() error
}

// LayerOnGPU returns true if transformer layer should run on GPU
// layer: 0..totalLayers-1
// ngl: number of layers to offload (like -ngl in llama.cpp)
func LayerOnGPU(layer, ngl, totalLayers int) bool {
	if ngl <= 0 || layer < 0 {
		return false
	}

	if layer >= totalLayers {
		return false
	}

	return layer < ngl
}
