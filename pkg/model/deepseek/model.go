package deepseek

import (
	"fmt"

	"github.com/magomedcoder/gogguf/pkg/format"
	"github.com/magomedcoder/gogguf/pkg/gpu"
	"github.com/magomedcoder/gogguf/pkg/model/moe"
	"github.com/magomedcoder/gogguf/pkg/ops"
	"github.com/magomedcoder/gogguf/pkg/weights"
)

// Model - DeepSeek / DeepSeek-MoE (RMSNorm, Llama RoPE, SwiGLU; dense lead + MoE)
type Model struct {
	cfg          Config
	weights      *weights.Store
	cache        *KVCache
	gpu          gpu.Backend
	ngl          int
	gpuMaxSeq    int
	scratch      scratch
	layerNorms   []layerNorms
	layerTensors []layerTensors
	outNorm      []float32
	lmHeadName   string
}

// Load создаёт DeepSeek из весов
func Load(w *weights.Store, g gpu.Backend, ngl, gpuMaxSeq int) (*Model, error) {
	cfg, err := ParseConfig(w.Reader())
	if err != nil {
		return nil, err
	}

	if ngl > cfg.NumLayers {
		ngl = cfg.NumLayers
	}

	layerNorms, outNorm, err := loadNormWeights(w, cfg.NumLayers)
	if err != nil {
		return nil, err
	}

	layerTensors, err := loadLayerTensors(w, cfg)
	if err != nil {
		return nil, err
	}

	lmHeadName, err := resolveLMHeadName(w)
	if err != nil {
		return nil, err
	}

	m := &Model{
		cfg:          cfg,
		weights:      w,
		cache:        NewKVCache(cfg),
		gpu:          g,
		ngl:          ngl,
		gpuMaxSeq:    gpuMaxSeq,
		scratch:      newScratch(cfg),
		layerNorms:   layerNorms,
		layerTensors: layerTensors,
		outNorm:      outNorm,
		lmHeadName:   lmHeadName,
	}

	if err := m.initGPUKVCache(); err != nil {
		return nil, err
	}

	return m, nil
}

func (m *Model) initGPUKVCache() error {
	if m.gpu == nil || m.ngl <= 0 {
		return nil
	}

	kvDim := m.cfg.NumKVHeads * m.cfg.HeadDim
	maxSeq := gpu.CapMaxSeq(m.cfg.ContextLength, m.gpuMaxSeq)

	return m.gpu.KVCacheInit(m.ngl, maxSeq, kvDim, m.cfg.NumHeads, m.cfg.HeadDim)
}

// Config возвращает конфигурацию модели
func (m *Model) Config() Config {
	return m.cfg
}

// ResetCache сбрасывает KV-cache
func (m *Model) ResetCache() {
	m.cache.Reset()
	if m.gpu != nil {
		m.gpu.KVCacheReset()
	}
}

// Close освобождает GPU-ресурсы модели
func (m *Model) Close() error {
	if m.gpu == nil {
		return nil
	}
	err := m.gpu.Close()
	m.gpu = nil
	return err
}

// Forward выполняет forward pass для последовательности tokenIDs начиная с startPos
func (m *Model) Forward(tokenIDs []int, startPos int) ([]float32, error) {
	if len(tokenIDs) == 0 {
		return nil, fmt.Errorf("deepseek: пустой ввод")
	}

	for i, tok := range tokenIDs {
		if err := m.forwardToken(tok, startPos+i); err != nil {
			return nil, err
		}
	}

	if err := m.logitsFromHidden(m.scratch.x); err != nil {
		return nil, err
	}

	copy(m.scratch.out, m.scratch.logits)
	return m.scratch.out, nil
}

// EmbeddingDim возвращает размер скрытого состояния
func (m *Model) EmbeddingDim() int {
	return m.cfg.EmbeddingDim
}

// Embed - last-token RMSNorm(hidden) до lm_head (сбрасывает KV)
func (m *Model) Embed(tokenIDs []int) ([]float32, error) {
	if len(tokenIDs) == 0 {
		return nil, fmt.Errorf("deepseek: пустой ввод")
	}

	m.ResetCache()
	defer m.ResetCache()

	for i, tok := range tokenIDs {
		if err := m.forwardToken(tok, i); err != nil {
			return nil, err
		}
	}

	if err := ops.RMSNormInto(m.scratch.h, m.scratch.x, m.outNorm, m.cfg.RMSNormEps); err != nil {
		return nil, err
	}

	out := make([]float32, m.cfg.EmbeddingDim)
	copy(out, m.scratch.h)
	return out, nil
}

func (m *Model) forwardToken(tokenID, pos int) error {
	if err := m.embedToken(tokenID); err != nil {
		return err
	}

	for layer := 0; layer < m.cfg.NumLayers; layer++ {
		if err := m.forwardBlock(layer, pos); err != nil {
			return err
		}
	}
	m.cache.Advance()
	return nil
}

func (m *Model) embedToken(tokenID int) error {
	raw, err := m.weights.Raw("token_embd.weight")
	if err != nil {
		return err
	}

	info, err := m.weights.Info("token_embd.weight")
	if err != nil {
		return err
	}

	switch info.Type {
	case format.GgmlQ8_0:
		return ops.EmbeddingQ8_0Into(m.scratch.x, raw, m.cfg.EmbeddingDim, tokenID)
	case format.GgmlQ4_0:
		return ops.EmbeddingQ4_0Into(m.scratch.x, raw, m.cfg.EmbeddingDim, tokenID)
	case format.GgmlQ4_1:
		return ops.EmbeddingQ4_1Into(m.scratch.x, raw, m.cfg.EmbeddingDim, tokenID)
	case format.GgmlQ5_0:
		return ops.EmbeddingQ5_0Into(m.scratch.x, raw, m.cfg.EmbeddingDim, tokenID)
	case format.GgmlQ5_1:
		return ops.EmbeddingQ5_1Into(m.scratch.x, raw, m.cfg.EmbeddingDim, tokenID)
	case format.GgmlQ4_K:
		return ops.EmbeddingQ4_KInto(m.scratch.x, raw, m.cfg.EmbeddingDim, tokenID)
	case format.GgmlQ5_K:
		return ops.EmbeddingQ5_KInto(m.scratch.x, raw, m.cfg.EmbeddingDim, tokenID)
	case format.GgmlQ6_K:
		return ops.EmbeddingQ6_KInto(m.scratch.x, raw, m.cfg.EmbeddingDim, tokenID)
	case format.GgmlQ2_K:
		return ops.EmbeddingQ2_KInto(m.scratch.x, raw, m.cfg.EmbeddingDim, tokenID)
	case format.GgmlQ3_K:
		return ops.EmbeddingQ3_KInto(m.scratch.x, raw, m.cfg.EmbeddingDim, tokenID)
	case format.GgmlQ8_K:
		return ops.EmbeddingQ8_KInto(m.scratch.x, raw, m.cfg.EmbeddingDim, tokenID)
	default:
		f32, err := m.weights.Floats("token_embd.weight")
		if err != nil {
			return err
		}
		off := tokenID * m.cfg.EmbeddingDim
		copy(m.scratch.x, f32[off:off+m.cfg.EmbeddingDim])
		return nil
	}
}

func (m *Model) forwardBlock(layer int, pos int) error {
	ln := m.layerNorms[layer]
	lt := m.layerTensors[layer]

	if err := ops.RMSNormInto(m.scratch.h, m.scratch.x, ln.attnNorm, m.cfg.RMSNormEps); err != nil {
		return err
	}

	if err := m.matmulInto(lt.attnQ, m.cfg.NumHeads*m.cfg.HeadDim, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.q, layer); err != nil {
		return err
	}

	if err := m.matmulInto(lt.attnK, m.cfg.NumKVHeads*m.cfg.HeadDim, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.k, layer); err != nil {
		return err
	}

	if err := m.matmulInto(lt.attnV, m.cfg.NumKVHeads*m.cfg.HeadDim, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.v, layer); err != nil {
		return err
	}

	ops.ApplyRoPEHeadsNorm(m.scratch.q, m.cfg.NumHeads, m.cfg.HeadDim, pos, m.cfg.RopeFreqBase)
	ops.ApplyRoPEHeadsNorm(m.scratch.k, m.cfg.NumKVHeads, m.cfg.HeadDim, pos, m.cfg.RopeFreqBase)

	kvPos := m.cache.Len()
	m.cache.Append(layer, m.scratch.k, m.scratch.v)
	if m.gpu != nil && gpu.LayerOnGPU(layer, m.ngl, m.cfg.NumLayers) {
		_ = m.gpu.KVCacheAppend(layer, kvPos, m.scratch.k, m.scratch.v)
	}

	seqLen := m.cache.Len() + 1
	if err := m.attentionScoresInto(m.scratch.attn, m.scratch.q, m.cache.KLayer(layer), m.cache.VLayer(layer), m.scratch.scores, seqLen, layer); err != nil {
		return err
	}

	if err := m.matmulInto(lt.attnOut, m.cfg.EmbeddingDim, m.cfg.NumHeads*m.cfg.HeadDim, m.scratch.attn, m.scratch.h, layer); err != nil {
		return err
	}
	ops.AddInPlace(m.scratch.x, m.scratch.h)

	if err := ops.RMSNormInto(m.scratch.h, m.scratch.x, ln.ffnNorm, m.cfg.RMSNormEps); err != nil {
		return err
	}

	if lt.moe {
		if err := m.ffnMoE(lt, layer); err != nil {
			return err
		}
		ops.AddInPlace(m.scratch.x, m.scratch.moeAcc)
		return nil
	}

	if m.gpu != nil && gpu.LayerOnGPU(layer, m.ngl, m.cfg.NumLayers) {
		if err := m.ffnGPU(lt, m.scratch.h, m.scratch.h); err == nil {
			ops.AddInPlace(m.scratch.x, m.scratch.h)
			return nil
		}
	}

	if err := m.matmulInto(lt.ffnGate, m.cfg.FFNHidden, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.gate, layer); err != nil {
		return err
	}

	if err := m.matmulInto(lt.ffnUp, m.cfg.FFNHidden, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.up, layer); err != nil {
		return err
	}

	ops.SwiGLUInPlace(m.scratch.gate[:m.cfg.FFNHidden], m.scratch.up[:m.cfg.FFNHidden])
	if err := m.matmulInto(lt.ffnDown, m.cfg.EmbeddingDim, m.cfg.FFNHidden, m.scratch.gate[:m.cfg.FFNHidden], m.scratch.h, layer); err != nil {
		return err
	}
	ops.AddInPlace(m.scratch.x, m.scratch.h)

	return nil
}

func (m *Model) ffnMoE(lt layerTensors, layer int) error {
	embd := m.cfg.EmbeddingDim
	nExp := m.cfg.ExpertCount
	ffn := m.cfg.ExpertFFN

	if err := m.matmulInto(lt.gateInp, nExp, embd, m.scratch.h, m.scratch.router[:nExp], layer); err != nil {
		return err
	}

	idxs, weights := moe.TopKSoftmax(m.scratch.router[:nExp], m.cfg.ExpertUsedCount, false, m.cfg.ExpertWeightScale)

	clear(m.scratch.moeAcc)

	gateW, err := m.weights.Floats(lt.gateExps)
	if err != nil {
		return err
	}

	upW, err := m.weights.Floats(lt.upExps)
	if err != nil {
		return err
	}

	downW, err := m.weights.Floats(lt.downExps)
	if err != nil {
		return err
	}

	expertElems := ffn * embd
	gateBuf := m.scratch.gate[:ffn]
	upBuf := m.scratch.up[:ffn]

	for i, ei := range idxs {
		gOff := ei * expertElems
		uOff := ei * expertElems
		dOff := ei * expertElems
		if err := ops.MatMulVecInto(gateW[gOff:gOff+expertElems], ffn, embd, m.scratch.h, gateBuf); err != nil {
			return err
		}

		if err := ops.MatMulVecInto(upW[uOff:uOff+expertElems], ffn, embd, m.scratch.h, upBuf); err != nil {
			return err
		}

		ops.SwiGLUInPlace(gateBuf, upBuf)
		if err := ops.MatMulVecInto(downW[dOff:dOff+expertElems], embd, ffn, gateBuf, m.scratch.tmp); err != nil {
			return err
		}

		w := weights[i]
		for j := range embd {
			m.scratch.moeAcc[j] += m.scratch.tmp[j] * w
		}
	}

	if m.cfg.ExpertShared > 0 && lt.gateShexp != "" {
		shared := m.cfg.sharedFFN()
		sg := m.scratch.gate[:shared]
		su := m.scratch.up[:shared]
		if err := m.matmulInto(lt.gateShexp, shared, embd, m.scratch.h, sg, layer); err != nil {
			return err
		}

		if err := m.matmulInto(lt.upShexp, shared, embd, m.scratch.h, su, layer); err != nil {
			return err
		}

		ops.SwiGLUInPlace(sg, su)
		if err := m.matmulInto(lt.downShexp, embd, shared, sg, m.scratch.tmp, layer); err != nil {
			return err
		}

		ops.AddInPlace(m.scratch.moeAcc, m.scratch.tmp)
	}

	return nil
}

func (m *Model) ffnGPU(lt layerTensors, x, out []float32) error {
	info, err := m.weights.Info(lt.ffnGate)
	if err != nil {
		return err
	}

	embd, ffn := m.cfg.EmbeddingDim, m.cfg.FFNHidden
	if info.Type == format.GgmlQ8_0 {
		gateRaw, err := m.weights.Raw(lt.ffnGate)
		if err != nil {
			return err
		}

		upRaw, err := m.weights.Raw(lt.ffnUp)
		if err != nil {
			return err
		}

		downRaw, err := m.weights.Raw(lt.ffnDown)
		if err != nil {
			return err
		}

		return m.gpu.FFNSwiGLUQ8_0Cached(lt.ffnGate, lt.ffnUp, lt.ffnDown, gateRaw, upRaw, downRaw, x, out, embd, ffn)
	}

	gateW, err := m.weights.Floats(lt.ffnGate)
	if err != nil {
		return err
	}

	upW, err := m.weights.Floats(lt.ffnUp)
	if err != nil {
		return err
	}

	downW, err := m.weights.Floats(lt.ffnDown)
	if err != nil {
		return err
	}

	return m.gpu.FFNSwiGLUCached(lt.ffnGate, lt.ffnUp, lt.ffnDown, gateW, upW, downW, x, out, embd, ffn)
}

func (m *Model) matmulInto(name string, rows, cols int, vec, out []float32, layer int) error {
	if m.gpu != nil && gpu.LayerOnGPU(layer, m.ngl, m.cfg.NumLayers) {
		got, err := m.matmulGPU(name, rows, cols, vec)
		if err != nil {
			return err
		}
		copy(out, got)
		return nil
	}

	raw, err := m.weights.Raw(name)
	if err != nil {
		return err
	}

	info, err := m.weights.Info(name)
	if err != nil {
		return err
	}

	switch info.Type {
	case format.GgmlQ8_0:
		return ops.MatMulVecQ8_0Into(raw, rows, cols, vec, out)
	case format.GgmlQ4_0:
		return ops.MatMulVecQ4_0Into(raw, rows, cols, vec, out)
	case format.GgmlQ4_1:
		return ops.MatMulVecQ4_1Into(raw, rows, cols, vec, out)
	case format.GgmlQ5_0:
		return ops.MatMulVecQ5_0Into(raw, rows, cols, vec, out)
	case format.GgmlQ5_1:
		return ops.MatMulVecQ5_1Into(raw, rows, cols, vec, out)
	case format.GgmlQ4_K:
		return ops.MatMulVecQ4_KInto(raw, rows, cols, vec, out)
	case format.GgmlQ5_K:
		return ops.MatMulVecQ5_KInto(raw, rows, cols, vec, out)
	case format.GgmlQ6_K:
		return ops.MatMulVecQ6_KInto(raw, rows, cols, vec, out)
	case format.GgmlQ2_K:
		return ops.MatMulVecQ2_KInto(raw, rows, cols, vec, out)
	case format.GgmlQ3_K:
		return ops.MatMulVecQ3_KInto(raw, rows, cols, vec, out)
	case format.GgmlQ8_K:
		return ops.MatMulVecQ8_KInto(raw, rows, cols, vec, out)
	default:
		f32, err := m.weights.Floats(name)
		if err != nil {
			return err
		}

		return ops.MatMulVecInto(f32, rows, cols, vec, out)
	}
}

func (m *Model) matmulGPU(name string, rows, cols int, vec []float32) ([]float32, error) {
	info, err := m.weights.Info(name)
	if err != nil {
		return nil, err
	}

	if info.Type == format.GgmlQ8_0 {
		raw, err := m.weights.Raw(name)
		if err != nil {
			return nil, err
		}
		return m.gpu.MatMulVecQ8_0Cached(name, raw, rows, cols, vec)
	}

	if info.Type == format.GgmlQ4_0 {
		raw, err := m.weights.Raw(name)
		if err != nil {
			return nil, err
		}
		return m.gpu.MatMulVecQ4_0Cached(name, raw, rows, cols, vec)
	}

	if info.Type == format.GgmlQ4_K {
		raw, err := m.weights.Raw(name)
		if err != nil {
			return nil, err
		}
		return m.gpu.MatMulVecQ4_KCached(name, raw, rows, cols, vec)
	}

	if info.Type == format.GgmlQ5_K {
		raw, err := m.weights.Raw(name)
		if err != nil {
			return nil, err
		}
		return m.gpu.MatMulVecQ5_KCached(name, raw, rows, cols, vec)
	}

	if info.Type == format.GgmlQ6_K {
		raw, err := m.weights.Raw(name)
		if err != nil {
			return nil, err
		}
		return m.gpu.MatMulVecQ6_KCached(name, raw, rows, cols, vec)
	}

	f32, err := m.weights.Floats(name)
	if err != nil {
		return nil, err
	}

	return m.gpu.MatMulVecCached(name, f32, rows, cols, vec)
}

func (m *Model) logitsFromHidden(x []float32) error {
	if err := ops.RMSNormInto(m.scratch.h, x, m.outNorm, m.cfg.RMSNormEps); err != nil {
		return err
	}

	name := m.lmHeadName
	raw, err := m.weights.Raw(name)
	if err != nil {
		return err
	}

	info, err := m.weights.Info(name)
	if err != nil {
		return err
	}

	switch info.Type {
	case format.GgmlQ8_0:
		return ops.MatMulVecQ8_0Into(raw, m.cfg.VocabSize, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.logits)
	case format.GgmlQ4_0:
		return ops.MatMulVecQ4_0Into(raw, m.cfg.VocabSize, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.logits)
	case format.GgmlQ4_1:
		return ops.MatMulVecQ4_1Into(raw, m.cfg.VocabSize, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.logits)
	case format.GgmlQ5_0:
		return ops.MatMulVecQ5_0Into(raw, m.cfg.VocabSize, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.logits)
	case format.GgmlQ5_1:
		return ops.MatMulVecQ5_1Into(raw, m.cfg.VocabSize, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.logits)
	case format.GgmlQ4_K:
		return ops.MatMulVecQ4_KInto(raw, m.cfg.VocabSize, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.logits)
	case format.GgmlQ5_K:
		return ops.MatMulVecQ5_KInto(raw, m.cfg.VocabSize, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.logits)
	case format.GgmlQ6_K:
		return ops.MatMulVecQ6_KInto(raw, m.cfg.VocabSize, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.logits)
	case format.GgmlQ2_K:
		return ops.MatMulVecQ2_KInto(raw, m.cfg.VocabSize, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.logits)
	case format.GgmlQ3_K:
		return ops.MatMulVecQ3_KInto(raw, m.cfg.VocabSize, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.logits)
	case format.GgmlQ8_K:
		return ops.MatMulVecQ8_KInto(raw, m.cfg.VocabSize, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.logits)
	default:
		f32, err := m.weights.Floats(name)
		if err != nil {
			return err
		}

		return ops.MatMulVecInto(f32, m.cfg.VocabSize, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.logits)
	}
}

func (m *Model) attentionScoresInto(dst, q, k, v, scores []float32, seqLen, layer int) error {
	if m.gpu != nil && gpu.LayerOnGPU(layer, m.ngl, m.cfg.NumLayers) {
		if err := m.gpu.AttentionScoresKV(layer, dst, q, seqLen, m.cfg.NumHeads, m.cfg.NumKVHeads, m.cfg.HeadDim); err == nil {
			return nil
		}

		if err := m.gpu.AttentionScoresInto(dst, q, k, v, scores, seqLen, m.cfg.NumHeads, m.cfg.NumKVHeads, m.cfg.HeadDim); err == nil {
			return nil
		}
	}

	return ops.AttentionScoresInto(dst, q, k, v, scores, seqLen, m.cfg.NumHeads, m.cfg.NumKVHeads, m.cfg.HeadDim)
}
