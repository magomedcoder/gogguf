package gpu

import (
	"errors"

	"github.com/magomedcoder/gogguf/pkg/format"
	"github.com/magomedcoder/gogguf/pkg/ops"
)

// RoPEMode - режим RoPE во fused QKV-пути (NeoX для Qwen/Mistral/Gemma, NORM для Llama)
type RoPEMode = ops.RoPEMode

const (
	RoPENeoX = ops.RoPENeoX
	RoPENorm = ops.RoPENorm
)

// ErrKVCacheUnavailable означает, что backend не поддерживает GPU KV-cache
var ErrKVCacheUnavailable = errors.New("gpu: kv cache unavailable")

// ErrHiddenUnavailable означает, что backend не умеет держать hidden на устройстве
var ErrHiddenUnavailable = errors.New("gpu: device-resident hidden unavailable")

// FusedQuantSupported сообщает, умеют ли fused-пути (FFN/QKV/residual/lm_head) работать с этим типом весов нативно, без деквантизации всей матрицы в FP32
func FusedQuantSupported(t format.GGML) bool {
	switch t {
	case format.GgmlQ8_0, format.GgmlQ4_0, format.GgmlQ4_K, format.GgmlQ5_K, format.GgmlQ6_K:
		return true
	default:
		return false
	}
}

// Backend выполняет вычисления на GPU (CUDA)
type Backend interface {
	// Name возвращает имя устройства, например "CUDA:0 NVIDIA ..."
	Name() string

	// VRAMInfo возвращает занятую и общую видеопамять устройства в байтах.
	// Backend без видеопамяти (CPU) возвращает 0, 0, nil
	VRAMInfo() (used, total uint64, err error)

	// MatMulVec умножает matrix[rows*cols] на vec[cols]
	MatMulVec(matrix []float32, rows, cols int, vec []float32) ([]float32, error)

	// MatMulVecCached как MatMulVec, но matrix загружается на GPU один раз по name
	MatMulVecCached(name string, matrix []float32, rows, cols int, vec []float32) ([]float32, error)

	// MatMulVecQ8_0Cached matmul Q8_0-матрицы без деквантизации в FP32
	MatMulVecQ8_0Cached(name string, raw []byte, rows, cols int, vec []float32) ([]float32, error)

	// MatMulVecQ4_0Cached matmul Q4_0-матрицы без полной деквантизации в FP32
	MatMulVecQ4_0Cached(name string, raw []byte, rows, cols int, vec []float32) ([]float32, error)

	// MatMulVecQ4_KCached matmul Q4_K-матрицы (K-quant)
	MatMulVecQ4_KCached(name string, raw []byte, rows, cols int, vec []float32) ([]float32, error)

	// MatMulVecQ5_KCached matmul Q5_K-матрицы (K-quant)
	MatMulVecQ5_KCached(name string, raw []byte, rows, cols int, vec []float32) ([]float32, error)

	// MatMulVecQ6_KCached matmul Q6_K-матрицы (K-quant)
	MatMulVecQ6_KCached(name string, raw []byte, rows, cols int, vec []float32) ([]float32, error)

	// RMSNormInto записывает RMS-нормализацию в dst (GPU или CPU)
	RMSNormInto(dst, x, weight []float32, eps float32) error

	// ApplyRoPEHeads применяет NeoX/Qwen RoPE к nHeads головам в v (in-place)
	ApplyRoPEHeads(v []float32, nHeads, headDim, pos int, freqBase float32) error

	// ApplyRoPEHeadsNorm применяет Llama RoPE (пары соседних dim) к nHeads головам в v
	ApplyRoPEHeadsNorm(v []float32, nHeads, headDim, pos int, freqBase float32) error

	// SwiGLUInPlace вычисляет silu(gate)*up, результат в gate
	SwiGLUInPlace(gate, up []float32) error

	// FFNSwiGLUCached gate/up/down matmul + SwiGLU; на CUDA активации остаются на GPU
	FFNSwiGLUCached(gateName, upName, downName string, gateW, upW, downW, x, out []float32, embd, ffn int) error

	// FFNSwiGLUQ8_0Cached то же для Q8_0 весов
	FFNSwiGLUQ8_0Cached(gateName, upName, downName string, gateRaw, upRaw, downRaw []byte, x, out []float32, embd, ffn int) error

	// AttnFFNResidualCached WO(attn)+residual+RMSNorm+FFN+residual на GPU
	AttnFFNResidualCached(woName, ffnNormName, gateName, upName, downName string, woW, ffnNorm, gateW, upW, downW, x, attn []float32, embd, attnDim, ffn int, eps float32) error

	// AttnFFNResidualQ8_0Cached то же для Q8_0 matmul-весов (norm - FP32)
	AttnFFNResidualQ8_0Cached(woName, ffnNormName, gateName, upName, downName string, woRaw, gateRaw, upRaw, downRaw []byte, ffnNorm, x, attn []float32, embd, attnDim, ffn int, eps float32) error

	// QKVRoPEAttentionCached: h->QKV->head RMSNorm->RoPE->KV append->attention на GPU
	QKVRoPEAttentionCached(qName, kName, vName, qNormName, kNormName string, qW, kW, vW, qNorm, kNorm, h, cos, sin, attn, kOut, vOut []float32, embd, nHeads, nKVHeads, headDim, layer, kvPos, seqLen int, eps float32) error

	// QKVRoPEAttentionQ8_0Cached то же для Q8_0 matmul-весов (norm - FP32)
	QKVRoPEAttentionQ8_0Cached(qName, kName, vName, qNormName, kNormName string, qRaw, kRaw, vRaw []byte, qNorm, kNorm, h, cos, sin, attn, kOut, vOut []float32, embd, nHeads, nKVHeads, headDim, layer, kvPos, seqLen int, eps float32) error

	// FFNSwiGLUQuantCached FFN SwiGLU для Q8_0/Q4_0/Q4_K/Q5_K/Q6_K весов
	FFNSwiGLUQuantCached(t format.GGML, gateName, upName, downName string, gateRaw, upRaw, downRaw []byte, x, out []float32, embd, ffn int) error

	// FFNGeGLUQuantCached FFN GeGLU (Gemma) для тех же типов весов
	FFNGeGLUQuantCached(t format.GGML, gateName, upName, downName string, gateRaw, upRaw, downRaw []byte, x, out []float32, embd, ffn int) error

	// AttnFFNResidualQuantCached WO+residual+RMSNorm+FFN+residual для квантованных весов.
	// Если hidden резидентен (HiddenUpload), x не грузится на GPU и остаётся на устройстве: host-буфер x в этом случае не обновляется.
	AttnFFNResidualQuantCached(t format.GGML, woName, ffnNormName, gateName, upName, downName string, woRaw, gateRaw, upRaw, downRaw []byte, ffnNorm, x, attn []float32, embd, attnDim, ffn int, eps float32) error

	// QKVRoPEAttentionQuantCached QKV+RoPE+attn для квантованных весов.
	// Если hidden резидентен, h считается на GPU как RMSNorm(resid, attnNorm) и host h не читается.
	// mode - раскладка RoPE; пустые qNormName/kNormName - слой без QK-norm (Llama / Mistral).
	QKVRoPEAttentionQuantCached(t format.GGML, mode RoPEMode, qName, kName, vName, qNormName, kNormName, attnNormName string, qRaw, kRaw, vRaw []byte, qNorm, kNorm, attnNorm, h, cos, sin, attn, kOut, vOut []float32, embd, nHeads, nKVHeads, headDim, layer, kvPos, seqLen int, eps float32) error

	// HiddenResident сообщает, поддерживает ли backend device-resident hidden state
	HiddenResident() bool

	// HiddenActive сообщает, лежит ли актуальный hidden state на устройстве.
	// После ошибки fused-слоя: true - hidden цел (можно HiddenDownload и уйти на CPU), false - residual был частично изменён, актуального hidden нет нигде.
	HiddenActive() bool

	// HiddenUpload кладёт hidden state на устройство (один HtoD на токен)
	HiddenUpload(x []float32) error

	// HiddenDownload забирает device-resident hidden state в dst и выключает residency
	HiddenDownload(dst []float32) error

	// LogitsFromDevice считает RMSNorm(resident hidden) + lm_head на GPU; на host копируются только logits. Для FP32-головы используется headF32, иначе headRaw.
	LogitsFromDevice(t format.GGML, normName string, norm []float32, headName string, headRaw []byte, headF32, logits []float32, vocab, embd int, eps float32) error

	// AttentionScoresInto записывает scaled dot-product attention в dst
	AttentionScoresInto(dst, q, k, v, scores []float32, seqLen, nHeads, nKVHeads, headDim int) error

	// KVCacheInit выделяет GPU-буферы K/V для offloaded слоёв
	KVCacheInit(layers, maxSeq, kvDim, nHeads, headDim int) error

	// KVCacheReset сбрасывает GPU KV-cache
	KVCacheReset()

	// KVCacheAppend добавляет K/V одного токена в позицию pos
	KVCacheAppend(layer, pos int, k, v []float32) error

	// KVCacheAppendN добавляет K/V n токенов начиная с позиции pos (batch prefill)
	KVCacheAppendN(layer, pos int, k, v []float32, n int) error

	// AttentionScoresKV attention с K/V из GPU KV-cache
	AttentionScoresKV(layer int, dst, q []float32, seqLen, nHeads, nKVHeads, headDim int) error

	Close() error
}

// LayerOnGPU возвращает true, если transformer-слой layer должен выполняться на GPU
// layer: 0..totalLayers-1
// ngl: число слоёв для offload (как -ngl в llama.cpp)
func LayerOnGPU(layer, ngl, totalLayers int) bool {
	if ngl <= 0 || layer < 0 {
		return false
	}

	if layer >= totalLayers {
		return false
	}

	return layer < ngl
}
