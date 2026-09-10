package qwen3

import (
	"fmt"
	"os"

	"github.com/magomedcoder/gogguf/pkg/format"
	"github.com/magomedcoder/gogguf/pkg/gpu"
	"github.com/magomedcoder/gogguf/pkg/model/moe"
	"github.com/magomedcoder/gogguf/pkg/ops"
	"github.com/magomedcoder/gogguf/pkg/weights"
)

// Model - Qwen3 transformer
type Model struct {
	cfg          Config
	weights      *weights.Store
	cache        *KVCache
	gpu          gpu.Backend
	ngl          int
	gpuMaxSeq    int
	nBatch       int // размер chunk prefill (1 = по токену); decode всегда 1
	debug        *DebugHooks
	scratch      scratch
	layerNorms   []layerNorms
	layerTensors []layerTensors
	outNorm      []float32
	lmHeadName   string
}

// Load создаёт Qwen3 из весов
func Load(w *weights.Store, g gpu.Backend, ngl, gpuMaxSeq, nBatch int) (*Model, error) {
	return loadWithConfig(w, g, ngl, gpuMaxSeq, nBatch, ParseConfig)
}

// LoadMoE создаёт Qwen3-MoE (qwen3moe.*, QK-norm + MoE)
func LoadMoE(w *weights.Store, g gpu.Backend, ngl, gpuMaxSeq, nBatch int) (*Model, error) {
	return loadWithConfig(w, g, ngl, gpuMaxSeq, nBatch, ParseConfigMoE)
}

func loadWithConfig(w *weights.Store, g gpu.Backend, ngl, gpuMaxSeq, nBatch int, parse func(*format.Reader) (Config, error)) (*Model, error) {
	cfg, err := parse(w.Reader())
	if err != nil {
		return nil, err
	}

	if ngl > cfg.NumLayers {
		ngl = cfg.NumLayers
	}

	if nBatch < 1 {
		nBatch = 1
	}

	if nBatch > 512 {
		nBatch = 512
	}

	// MoE: пока только serial (n_batch=1)
	if cfg.ExpertCount > 0 {
		nBatch = 1
	}

	layerNorms, outNorm, err := loadNormWeights(w, cfg.NumLayers)
	if err != nil {
		return nil, err
	}

	lmHeadName, err := resolveLMHeadName(w)
	if err != nil {
		return nil, err
	}

	layerTensors, err := loadLayerTensors(w, cfg)
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
		nBatch:       nBatch,
		scratch:      newScratch(cfg, nBatch),
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

// SetDebugHooks включает колбэки для пошаговой отладки forward pass
func (m *Model) SetDebugHooks(h *DebugHooks) {
	m.debug = h
}

// Forward выполняет forward pass для последовательности tokenIDs начиная с startPos
// Возвращает logits для последнего токена [vocabSize]
func (m *Model) Forward(tokenIDs []int, startPos int) ([]float32, error) {
	if len(tokenIDs) == 0 {
		return nil, fmt.Errorf("qwen3: пустой ввод")
	}

	// n_batch>1: CPU multi-token prefill; с GPU/debug/MoE - serial (как раньше)
	useBatch := m.nBatch > 1 && m.cfg.ExpertCount == 0 && m.debug == nil && (m.gpu == nil || m.ngl == 0)
	if useBatch {
		for i := 0; i < len(tokenIDs); {
			end := min(i+m.nBatch, len(tokenIDs))

			needLogits := end == len(tokenIDs)
			if err := m.forwardBatch(tokenIDs[i:end], startPos+i, needLogits); err != nil {
				return nil, err
			}
			i = end
		}
	} else {
		var err error
		for i, tok := range tokenIDs {
			pos := startPos + i
			last := i == len(tokenIDs)-1
			if err = m.forwardToken(tok, pos, last); err != nil {
				return nil, err
			}
		}

		if err = m.logitsFinish(); err != nil {
			return nil, err
		}
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
		return nil, fmt.Errorf("qwen3: пустой ввод")
	}
	m.ResetCache()
	defer m.ResetCache()

	embd := m.cfg.EmbeddingDim
	for i, tok := range tokenIDs {
		if err := m.forwardToken(tok, i, false); err != nil {
			return nil, err
		}
	}

	x := m.scratch.x[:embd]
	h := m.scratch.h[:embd]
	if err := ops.RMSNormInto(h, x, m.outNorm, m.cfg.RMSNormEps); err != nil {
		return nil, err
	}

	out := make([]float32, embd)
	copy(out, h)
	return out, nil
}

func (m *Model) forwardToken(tokenID, pos int, debug bool) error {
	if err := m.embedToken(tokenID); err != nil {
		return err
	}

	if debug && m.debug != nil && m.debug.OnEmbed != nil {
		m.debug.OnEmbed(m.scratch.x[:m.cfg.EmbeddingDim])
	}

	for layer := 0; layer < m.cfg.NumLayers; layer++ {
		if err := m.forwardBlock(layer, pos); err != nil {
			return err
		}

		if debug && m.debug != nil {
			if m.debug.OnLayer != nil {
				m.debug.OnLayer(layer, m.scratch.x[:m.cfg.EmbeddingDim])
			}

			if m.debug.OnLayerLogits != nil {
				if err := m.logitsFromHidden(m.scratch.x[:m.cfg.EmbeddingDim]); err != nil {
					return err
				}

				m.debug.OnLayerLogits(layer, m.scratch.logits)
			}
		}
	}
	m.cache.Advance()

	return nil
}

func (m *Model) embedToken(tokenID int) error {
	return m.embedTokenInto(tokenID, m.scratch.x[:m.cfg.EmbeddingDim])
}

func (m *Model) forwardBlock(layer int, pos int) error {
	ln := m.layerNorms[layer]
	lt := m.layerTensors[layer]
	usedQKVResidency := false

	embd := m.cfg.EmbeddingDim
	qDim := m.cfg.NumHeads * m.cfg.HeadDim
	kvDim := m.cfg.NumKVHeads * m.cfg.HeadDim
	ffn := m.cfg.FFNHidden
	x := m.scratch.x[:embd]
	h := m.scratch.h[:embd]
	q := m.scratch.q[:qDim]
	k := m.scratch.k[:kvDim]
	v := m.scratch.v[:kvDim]
	attn := m.scratch.attn[:qDim]
	gate := m.scratch.gate[:ffn]
	up := m.scratch.up[:ffn]

	if err := m.rmsNormInto(h, x, ln.attnNorm, layer); err != nil {
		return err
	}

	if m.gpu != nil && gpu.LayerOnGPU(layer, m.ngl, m.cfg.NumLayers) && qkvResidencyEnabled() {
		if err := m.qkvAttnGPU(layer, pos, lt, ln, h, q, k, v, attn); err == nil {
			m.cache.Append(layer, k, v)
			usedQKVResidency = true
		}
	}

	if !usedQKVResidency {
		if err := m.matmulInto(lt.attnQ, qDim, embd, h, q, layer); err != nil {
			return err
		}

		if err := m.matmulInto(lt.attnK, kvDim, embd, h, k, layer); err != nil {
			return err
		}

		if err := m.matmulInto(lt.attnV, kvDim, embd, h, v, layer); err != nil {
			return err
		}

		if err := m.normHeadsInto(q, ln.qNorm, m.cfg.NumHeads, layer); err != nil {
			return err
		}

		if err := m.normHeadsInto(k, ln.kNorm, m.cfg.NumKVHeads, layer); err != nil {
			return err
		}

		m.applyRoPEHeads(q, m.cfg.NumHeads, pos, layer)
		m.applyRoPEHeads(k, m.cfg.NumKVHeads, pos, layer)

		kvPos := m.cache.Len()
		m.cache.Append(layer, k, v)
		if m.gpu != nil && gpu.LayerOnGPU(layer, m.ngl, m.cfg.NumLayers) {
			_ = m.gpu.KVCacheAppend(layer, kvPos, k, v)
		}

		seqLen := m.cache.Len() + 1

		if err := m.attentionScoresInto(attn, q, m.cache.KLayer(layer), m.cache.VLayer(layer), m.scratch.scores, seqLen, layer); err != nil {
			return err
		}
	}
	// WO+RMSNorm+FFN residency (отключить: GGUF_ATTN_FFN_RESIDENCY=0)
	if !lt.moe && m.gpu != nil && gpu.LayerOnGPU(layer, m.ngl, m.cfg.NumLayers) && attnFFNResidencyEnabled() {
		if err := m.attnFFNGPU(layer, lt, ln, x, attn); err == nil {
			return nil
		}
	}

	if err := m.matmulInto(lt.attnOut, embd, qDim, attn, h, layer); err != nil {
		return err
	}
	ops.AddInPlace(x, h)

	if err := m.rmsNormInto(h, x, ln.ffnNorm, layer); err != nil {
		return err
	}

	if lt.moe {
		if err := m.ffnMoE(lt, layer); err != nil {
			return err
		}

		ops.AddInPlace(x, m.scratch.moeAcc)
		return nil
	}

	if m.gpu != nil && gpu.LayerOnGPU(layer, m.ngl, m.cfg.NumLayers) {
		if err := m.ffnGPU(lt, h, h); err == nil {
			ops.AddInPlace(x, h)
			return nil
		}
	}

	if err := m.matmulInto(lt.ffnGate, ffn, embd, h, gate, layer); err != nil {
		return err
	}

	if err := m.matmulInto(lt.ffnUp, ffn, embd, h, up, layer); err != nil {
		return err
	}
	m.swigluInPlace(gate, up, layer)

	if err := m.matmulInto(lt.ffnDown, embd, ffn, gate, h, layer); err != nil {
		return err
	}

	ops.AddInPlace(x, h)
	return nil
}

func (m *Model) ffnMoE(lt layerTensors, layer int) error {
	embd := m.cfg.EmbeddingDim
	nExp := m.cfg.ExpertCount
	ffn := m.cfg.expertFFN()
	h := m.scratch.h[:embd]

	if err := m.matmulInto(lt.gateInp, nExp, embd, h, m.scratch.router[:nExp], layer); err != nil {
		return err
	}

	idxs, weights := moe.TopKSoftmax(m.scratch.router[:nExp], m.cfg.ExpertUsedCount, m.cfg.MoENormWeights, m.cfg.ExpertWeightScale)

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
		if err := ops.MatMulVecInto(gateW[gOff:gOff+expertElems], ffn, embd, h, gateBuf); err != nil {
			return err
		}

		if err := ops.MatMulVecInto(upW[gOff:gOff+expertElems], ffn, embd, h, upBuf); err != nil {
			return err
		}

		ops.SwiGLUInPlace(gateBuf, upBuf)

		if err := ops.MatMulVecInto(downW[gOff:gOff+expertElems], embd, ffn, gateBuf, m.scratch.tmp); err != nil {
			return err
		}

		w := weights[i]
		for j := range embd {
			m.scratch.moeAcc[j] += m.scratch.tmp[j] * w
		}
	}

	return nil
}

func attnFFNResidencyEnabled() bool {
	v := os.Getenv("GGUF_ATTN_FFN_RESIDENCY")
	if v == "0" || v == "false" || v == "off" {
		return false
	}
	// default: on (parallel RMSNorm)
	return true
}

func qkvResidencyEnabled() bool {
	v := os.Getenv("GGUF_QKV_RESIDENCY")
	if v == "0" || v == "false" || v == "off" {
		return false
	}

	return true
}

func (m *Model) attnFFNGPU(layer int, lt layerTensors, ln layerNorms, x, attn []float32) error {
	info, err := m.weights.Info(lt.attnOut)
	if err != nil {
		return err
	}

	embd := m.cfg.EmbeddingDim
	attnDim := m.cfg.NumHeads * m.cfg.HeadDim
	ffn := m.cfg.FFNHidden
	eps := m.cfg.RMSNormEps
	normName := fmt.Sprintf("blk.%d.ffn_norm.weight", layer)

	if info.Type == format.GgmlQ8_0 {
		woRaw, err := m.weights.Raw(lt.attnOut)
		if err != nil {
			return err
		}

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
		return m.gpu.AttnFFNResidualQ8_0Cached(lt.attnOut, normName, lt.ffnGate, lt.ffnUp, lt.ffnDown, woRaw, gateRaw, upRaw, downRaw, ln.ffnNorm, x, attn, embd, attnDim, ffn, eps)
	}

	woW, err := m.weights.Floats(lt.attnOut)
	if err != nil {
		return err
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

	return m.gpu.AttnFFNResidualCached(lt.attnOut, normName, lt.ffnGate, lt.ffnUp, lt.ffnDown, woW, ln.ffnNorm, gateW, upW, downW, x, attn, embd, attnDim, ffn, eps)
}

func (m *Model) qkvAttnGPU(layer, pos int, lt layerTensors, ln layerNorms, h, qOut, kOut, vOut, attn []float32) error {
	info, err := m.weights.Info(lt.attnQ)
	if err != nil {
		return err
	}

	half := m.cfg.HeadDim / 2
	cos := make([]float32, half)
	sin := make([]float32, half)
	ops.RoPECosSin(cos, sin, m.cfg.HeadDim, pos, m.cfg.RopeFreqBase)

	kvPos := m.cache.Len()
	seqLen := kvPos + 1
	qNormName := fmt.Sprintf("blk.%d.attn_q_norm.weight", layer)
	kNormName := fmt.Sprintf("blk.%d.attn_k_norm.weight", layer)

	if info.Type == format.GgmlQ8_0 {
		qRaw, err := m.weights.Raw(lt.attnQ)
		if err != nil {
			return err
		}

		kRaw, err := m.weights.Raw(lt.attnK)
		if err != nil {
			return err
		}

		vRaw, err := m.weights.Raw(lt.attnV)
		if err != nil {
			return err
		}

		return m.gpu.QKVRoPEAttentionQ8_0Cached(lt.attnQ, lt.attnK, lt.attnV, qNormName, kNormName, qRaw, kRaw, vRaw, ln.qNorm, ln.kNorm, h, cos, sin, attn, kOut, vOut, m.cfg.EmbeddingDim, m.cfg.NumHeads, m.cfg.NumKVHeads, m.cfg.HeadDim, layer, kvPos, seqLen, m.cfg.RMSNormEps)
	}

	qW, err := m.weights.Floats(lt.attnQ)
	if err != nil {
		return err
	}

	kW, err := m.weights.Floats(lt.attnK)
	if err != nil {
		return err
	}

	vW, err := m.weights.Floats(lt.attnV)
	if err != nil {
		return err
	}

	return m.gpu.QKVRoPEAttentionCached(lt.attnQ, lt.attnK, lt.attnV, qNormName, kNormName, qW, kW, vW, ln.qNorm, ln.kNorm, h, cos, sin, attn, kOut, vOut, m.cfg.EmbeddingDim, m.cfg.NumHeads, m.cfg.NumKVHeads, m.cfg.HeadDim, layer, kvPos, seqLen, m.cfg.RMSNormEps)
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

func (m *Model) matmul(name string, rows, cols int, vec []float32, layer int) ([]float32, error) {
	if m.gpu != nil && gpu.LayerOnGPU(layer, m.ngl, m.cfg.NumLayers) {
		return m.matmulGPU(name, rows, cols, vec)
	}

	raw, err := m.weights.Raw(name)
	if err != nil {
		return nil, err
	}

	info, err := m.weights.Info(name)
	if err != nil {
		return nil, err
	}

	switch info.Type {
	case format.GgmlQ8_0:
		return ops.MatMulVecQ8_0(raw, rows, cols, vec)
	case format.GgmlQ4_0:
		return ops.MatMulVecQ4_0(raw, rows, cols, vec)
	case format.GgmlQ4_1:
		return ops.MatMulVecQ4_1(raw, rows, cols, vec)
	case format.GgmlQ5_0:
		return ops.MatMulVecQ5_0(raw, rows, cols, vec)
	case format.GgmlQ5_1:
		return ops.MatMulVecQ5_1(raw, rows, cols, vec)
	case format.GgmlQ4_K:
		return ops.MatMulVecQ4_K(raw, rows, cols, vec)
	case format.GgmlQ5_K:
		return ops.MatMulVecQ5_K(raw, rows, cols, vec)
	case format.GgmlQ6_K:
		return ops.MatMulVecQ6_K(raw, rows, cols, vec)
	case format.GgmlQ2_K:
		return ops.MatMulVecQ2_K(raw, rows, cols, vec)
	case format.GgmlQ3_K:
		return ops.MatMulVecQ3_K(raw, rows, cols, vec)
	case format.GgmlQ8_K:
		return ops.MatMulVecQ8_K(raw, rows, cols, vec)
	default:
		f32, err := m.weights.Floats(name)
		if err != nil {
			return nil, err
		}

		return ops.MatMulVec(f32, rows, cols, vec)
	}
}

// matmulGPU выполняет matmul на GPU (Q8_0/Q4_0 без полной деквантизации, иначе FP32)
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

func (m *Model) logits() error {
	return m.logitsFromHidden(m.scratch.x[:m.cfg.EmbeddingDim])
}

func (m *Model) logitsFromHidden(x []float32) error {
	h := m.scratch.h[:len(x)]
	if err := ops.RMSNormInto(h, x, m.outNorm, m.cfg.RMSNormEps); err != nil {
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
		err = ops.MatMulVecQ8_0Into(raw, m.cfg.VocabSize, m.cfg.EmbeddingDim, h, m.scratch.logits)
	case format.GgmlQ4_0:
		err = ops.MatMulVecQ4_0Into(raw, m.cfg.VocabSize, m.cfg.EmbeddingDim, h, m.scratch.logits)
	case format.GgmlQ4_1:
		err = ops.MatMulVecQ4_1Into(raw, m.cfg.VocabSize, m.cfg.EmbeddingDim, h, m.scratch.logits)
	case format.GgmlQ5_0:
		err = ops.MatMulVecQ5_0Into(raw, m.cfg.VocabSize, m.cfg.EmbeddingDim, h, m.scratch.logits)
	case format.GgmlQ5_1:
		err = ops.MatMulVecQ5_1Into(raw, m.cfg.VocabSize, m.cfg.EmbeddingDim, h, m.scratch.logits)
	case format.GgmlQ4_K:
		err = ops.MatMulVecQ4_KInto(raw, m.cfg.VocabSize, m.cfg.EmbeddingDim, h, m.scratch.logits)
	case format.GgmlQ5_K:
		err = ops.MatMulVecQ5_KInto(raw, m.cfg.VocabSize, m.cfg.EmbeddingDim, h, m.scratch.logits)
	case format.GgmlQ6_K:
		err = ops.MatMulVecQ6_KInto(raw, m.cfg.VocabSize, m.cfg.EmbeddingDim, h, m.scratch.logits)
	case format.GgmlQ2_K:
		err = ops.MatMulVecQ2_KInto(raw, m.cfg.VocabSize, m.cfg.EmbeddingDim, h, m.scratch.logits)
	case format.GgmlQ3_K:
		err = ops.MatMulVecQ3_KInto(raw, m.cfg.VocabSize, m.cfg.EmbeddingDim, h, m.scratch.logits)
	case format.GgmlQ8_K:
		err = ops.MatMulVecQ8_KInto(raw, m.cfg.VocabSize, m.cfg.EmbeddingDim, h, m.scratch.logits)
	default:
		f32, err := m.weights.Floats(name)
		if err != nil {
			return err
		}
		err = ops.MatMulVecInto(f32, m.cfg.VocabSize, m.cfg.EmbeddingDim, h, m.scratch.logits)
	}

	return err
}

func (m *Model) emitLogitsDebug() {
	if m.debug != nil && m.debug.OnLogits != nil {
		m.debug.OnLogits(m.scratch.logits)
	}
}

func (m *Model) logitsFinish() error {
	if err := m.logits(); err != nil {
		return err
	}
	m.emitLogitsDebug()

	return nil
}

func (m *Model) rmsNormInto(dst, x, weight []float32, layer int) error {
	// RMSNorm на GPU невыгоден на PCIe (маленький буфер): только CPU.
	_ = layer
	return ops.RMSNormInto(dst, x, weight, m.cfg.RMSNormEps)
}

func (m *Model) applyRoPEHeads(v []float32, nHeads, pos, layer int) {
	// RoPE на GPU: HtoD/DtoH > выгоды на decode; CPU быстрее для residency-пути.
	_ = layer
	ops.ApplyRoPEHeads(v, nHeads, m.cfg.HeadDim, pos, m.cfg.RopeFreqBase)
}

func (m *Model) swigluInPlace(gate, up []float32, layer int) {
	if m.gpu != nil && gpu.LayerOnGPU(layer, m.ngl, m.cfg.NumLayers) {
		if err := m.gpu.SwiGLUInPlace(gate, up); err == nil {
			return
		}
	}

	ops.SwiGLUInPlace(gate, up)
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

func (m *Model) normHeadsInto(v []float32, weight []float32, nHeads, layer int) error {
	if len(weight) != m.cfg.HeadDim {
		return fmt.Errorf("qwen3: len(weight)=%d, head_dim=%d", len(weight), m.cfg.HeadDim)
	}

	for h := range nHeads {
		off := h * m.cfg.HeadDim
		if err := m.rmsNormInto(v[off:off+m.cfg.HeadDim], v[off:off+m.cfg.HeadDim], weight, layer); err != nil {
			return err
		}
	}

	return nil
}
