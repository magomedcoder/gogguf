package qwen3

import (
	"fmt"

	"github.com/magomedcoder/gogguf/pkg/format"
	"github.com/magomedcoder/gogguf/pkg/gpu"
	"github.com/magomedcoder/gogguf/pkg/model/gpuresid"
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

	layerQuant  []format.GGML // тип matmul-весов слоя для fused-путей
	residency   bool          // слои умеют держать hidden state на устройстве
	residDevice bool          // hidden state сейчас на устройстве, host-буфер x устарел
	logitsOnGPU bool          // out_norm + lm_head считаются на GPU
	gpuKVStale  bool          // GPU KV-cache неполный: attention только на CPU

	fused     *gpuresid.Runner   // fused GPU-пути слоя (§5)
	fusedDims gpuresid.Dims      // размерности для fused-путей
	gpuLayers []gpuresid.Tensors // имена весов слоя для fused-путей
	moeGPU    bool               // эксперты MoE считаются на GPU (§6)
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

	m.layerQuant = resolveLayerQuants(w, layerTensors)
	m.initFused()
	m.initResidency()

	return m, nil
}

// initFused готовит общую обвязку fused GPU-путей (§5): Qwen3 - NeoX RoPE + QK-norm
func (m *Model) initFused() {
	if m.gpu == nil {
		return
	}

	m.fused = gpuresid.New(m.weights, m.gpu, gpu.RoPENeoX)
	m.fusedDims = gpuresid.Dims{
		Embd:     m.cfg.EmbeddingDim,
		NHeads:   m.cfg.NumHeads,
		NKVHeads: m.cfg.NumKVHeads,
		HeadDim:  m.cfg.HeadDim,
		FFN:      m.cfg.FFNHidden,
		Eps:      m.cfg.RMSNormEps,
		RopeBase: m.cfg.RopeFreqBase,
	}

	m.gpuLayers = make([]gpuresid.Tensors, m.cfg.NumLayers)
	for i, lt := range m.layerTensors {
		p := fmt.Sprintf("blk.%d.", i)
		m.gpuLayers[i] = gpuresid.Tensors{
			AttnQ:    lt.attnQ,
			AttnK:    lt.attnK,
			AttnV:    lt.attnV,
			AttnOut:  lt.attnOut,
			FFNGate:  lt.ffnGate,
			FFNUp:    lt.ffnUp,
			FFNDown:  lt.ffnDown,
			QNorm:    p + "attn_q_norm.weight",
			KNorm:    p + "attn_k_norm.weight",
			AttnNorm: p + "attn_norm.weight",
			FFNNorm:  p + "ffn_norm.weight",
		}
	}

	// §6: эксперты считаются на GPU только если все веса экспертов нативно квантованы
	m.moeGPU = m.cfg.ExpertCount > 0
	for _, lt := range m.layerTensors {
		if lt.moe && !m.fused.MoESupported(lt.gateExps, lt.upExps, lt.downExps) {
			m.moeGPU = false
			break
		}
	}
}

// resolveLayerQuants возвращает тип matmul-весов каждого слоя; format.GgmlFloat32 как «нет единого квантованного типа» - такой слой пойдёт по FP32/discrete пути
func resolveLayerQuants(w *weights.Store, lts []layerTensors) []format.GGML {
	kinds := make([]format.GGML, len(lts))
	for i, lt := range lts {
		names := []string{lt.attnQ, lt.attnK, lt.attnV, lt.attnOut}
		if !lt.moe {
			names = append(names, lt.ffnGate, lt.ffnUp, lt.ffnDown)
		}

		kinds[i] = gpuresid.LayerQuant(w, names...)
	}

	return kinds
}

// initResidency включает device-resident hidden state (§1): требуются все слои на GPU,
// fused QKV/residual пути и единый поддерживаемый квант весов во всех слоях
func (m *Model) initResidency() {
	if m.gpu == nil || m.ngl < m.cfg.NumLayers || m.cfg.ExpertCount > 0 {
		return
	}

	if !hiddenResidencyEnabled() || !qkvResidencyEnabled() || !attnFFNResidencyEnabled() {
		return
	}

	if !m.gpu.HiddenResident() {
		return
	}

	for _, t := range m.layerQuant {
		if !gpu.FusedQuantSupported(t) {
			return
		}
	}

	m.residency = true
	m.logitsOnGPU = true
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
	m.residDevice = false
	m.gpuKVStale = false
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

	// n_batch>1: multi-token prefill; при -ngl matmul/attn чанка на GPU (loop MatMulVec*Cached + AttentionScoresKV), иначе CPU-GEMM. MoE/debug - serial.
	useBatch := m.nBatch > 1 && m.cfg.ExpertCount == 0 && m.debug == nil && len(tokenIDs) > 1
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

	// Embed читает hidden на host: при residency сначала забираем его с устройства
	if err := m.syncHiddenFromDevice(); err != nil {
		return nil, err
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

	// §1: единственный HtoD hidden state за токен, дальше слои работают на устройстве.
	// Отладочные хуки читают host-буфер x, поэтому с ними residency не включаем.
	// gpuKVStale: fused QKV считает attention по GPU KV-cache, а он неполный.
	if m.residency && m.debug == nil && !m.gpuKVStale {
		if err := m.gpu.HiddenUpload(m.scratch.x[:m.cfg.EmbeddingDim]); err != nil {
			m.residency = false
		} else {
			m.residDevice = true
		}
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

	// §1: слой целиком на устройстве - attn_norm, QKV, attention, WO, FFN и оба residual.
	// host x при этом не читается и не обновляется.
	if m.residDevice {
		err := m.forwardBlockDevice(layer, pos, ln, k, v, attn)
		if err == nil {
			return nil
		}

		if rerr := m.recoverHiddenFromDevice(layer, err); rerr != nil {
			return rerr
		}
	}

	if err := m.rmsNormInto(h, x, ln.attnNorm, layer); err != nil {
		return err
	}

	// fused QKV берёт attention из GPU KV-cache: при gpuKVStale он неполный
	if m.gpu != nil && gpu.LayerOnGPU(layer, m.ngl, m.cfg.NumLayers) && qkvResidencyEnabled() && !m.gpuKVStale {
		if err := m.qkvAttnGPU(layer, pos, ln, h, k, v, attn); err == nil {
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
		// токен не попал в GPU KV-cache - GPU attention отключаем до ResetCache, иначе fused QKV/AttentionScoresKV посчитают по неполному кешу
		if m.gpu != nil && gpu.LayerOnGPU(layer, m.ngl, m.cfg.NumLayers) && !m.gpuKVStale {
			if err := m.gpu.KVCacheAppend(layer, kvPos, k, v); err != nil {
				m.gpuKVStale = true
			}
		}

		seqLen := m.cache.Len() + 1

		if err := m.attentionScoresInto(attn, q, m.cache.KLayer(layer), m.cache.VLayer(layer), m.scratch.scores, seqLen, layer); err != nil {
			return err
		}
	}
	// WO+RMSNorm+FFN residency (отключить: GGUF_ATTN_FFN_RESIDENCY=0).
	// MoE-слой сюда не идёт: fused-путь считает и dense FFN, а его заменяют эксперты
	if !lt.moe && m.gpu != nil && gpu.LayerOnGPU(layer, m.ngl, m.cfg.NumLayers) && attnFFNResidencyEnabled() {
		if err := m.attnFFNGPU(layer, ln, x, attn); err == nil {
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
		if err := m.ffnGPU(layer, h, h); err == nil {
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

// forwardBlockDevice считает слой на GPU поверх резидентного hidden state.
// k/v копируются на host только для зеркала CPU KV-cache (нужно для fallback-путей).
func (m *Model) forwardBlockDevice(layer, pos int, ln layerNorms, k, v, attn []float32) error {
	if err := m.qkvAttnGPU(layer, pos, ln, nil, k, v, attn); err != nil {
		return err
	}
	m.cache.Append(layer, k, v)

	return m.attnFFNGPU(layer, ln, nil, attn)
}

// recoverHiddenFromDevice возвращает hidden state на host после сбоя device-слоя.
// Если residual уже был частично изменён, актуального состояния нет нигде - это ошибка.
func (m *Model) recoverHiddenFromDevice(layer int, cause error) error {
	m.residDevice = false
	m.residency = false

	if !m.gpu.HiddenActive() {
		return fmt.Errorf("qwen3: слой %d на GPU: %w (hidden state потерян)", layer, cause)
	}

	if err := m.gpu.HiddenDownload(m.scratch.x[:m.cfg.EmbeddingDim]); err != nil {
		return fmt.Errorf("qwen3: слой %d на GPU: %w (откат на host: %v)", layer, cause, err)
	}

	return nil
}

// syncHiddenFromDevice забирает hidden state с устройства, если он там (для host-читателей)
func (m *Model) syncHiddenFromDevice() error {
	if !m.residDevice {
		return nil
	}

	m.residDevice = false

	return m.gpu.HiddenDownload(m.scratch.x[:m.cfg.EmbeddingDim])
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

	// §6: выбранные эксперты считаются на GPU; веса эксперта кешируются на устройстве срезом raw-тензора, host-деквантизации всей матрицы экспертов нет
	if m.moeGPU && gpu.LayerOnGPU(layer, m.ngl, m.cfg.NumLayers) {
		if err := m.moeExpertsGPU(lt, idxs, weights, h, ffn); err == nil {
			return nil
		}

		// эксперт не пошёл на GPU (VRAM / тип весов): дальше только CPU
		m.moeGPU = false
		clear(m.scratch.moeAcc)
	}

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

// moeExpertsGPU считает выбранных экспертов fused-FFN на GPU и копит взвешенную сумму
func (m *Model) moeExpertsGPU(lt layerTensors, idxs []int, weights []float32, h []float32, ffn int) error {
	embd := m.cfg.EmbeddingDim
	for i, ei := range idxs {
		if err := m.fused.ExpertFFN(lt.gateExps, lt.upExps, lt.downExps, ei, h, m.scratch.tmp[:embd], embd, ffn); err != nil {
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
	return gpuresid.AttnFFNResidencyEnabled()
}

// hiddenResidencyEnabled - device-resident hidden state (выкл.: GGUF_HIDDEN_RESIDENCY=0)
func hiddenResidencyEnabled() bool {
	return gpuresid.HiddenResidencyEnabled()
}

func qkvResidencyEnabled() bool {
	return gpuresid.QKVResidencyEnabled()
}

// attnFFNGPU: WO + residual + ffn_norm + FFN + residual на GPU.
// x == nil - hidden резидентен на устройстве (§1).
func (m *Model) attnFFNGPU(layer int, ln layerNorms, x, attn []float32) error {
	return m.fused.AttnFFN(m.gpuLayers[layer], m.fusedDims, ln.ffnNorm, x, attn)
}

// qkvAttnGPU: QKV + QK-norm + RoPE + KV append + attention на GPU.
// h == nil - hidden резидентен: attn_norm считается на устройстве из residual (§1).
func (m *Model) qkvAttnGPU(layer, pos int, ln layerNorms, h, kOut, vOut, attn []float32) error {
	kvPos := m.cache.Len()

	return m.fused.QKVAttn(m.gpuLayers[layer], m.fusedDims, ln.qNorm, ln.kNorm, ln.attnNorm, h, attn, kOut, vOut, layer, pos, kvPos, kvPos+1)
}

func (m *Model) ffnGPU(layer int, x, out []float32) error {
	return m.fused.FFN(m.gpuLayers[layer], m.fusedDims, x, out)
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
	// §2: hidden на устройстве - out_norm + lm_head тоже на GPU, на host уходят только logits
	if m.residDevice && m.logitsOnGPU {
		if err := m.logitsDevice(); err == nil {
			return nil
		}

		// lm_head на GPU не получился (тип весов / VRAM): дальше считаем на host
		m.logitsOnGPU = false
	}

	if err := m.syncHiddenFromDevice(); err != nil {
		return err
	}

	return m.logitsFromHidden(m.scratch.x[:m.cfg.EmbeddingDim])
}

// logitsDevice считает RMSNorm(resident hidden) + lm_head на GPU без DtoH скрытого состояния
func (m *Model) logitsDevice() error {
	return m.fused.LogitsDevice("output_norm.weight", m.outNorm, m.lmHeadName, m.scratch.logits, m.cfg.VocabSize, m.cfg.EmbeddingDim, m.cfg.RMSNormEps)
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
		f32, ferr := m.weights.Floats(name)
		if ferr != nil {
			return ferr
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
		// gpuKVStale: часть токенов не попала в GPU KV-cache - attention только по CPU-кешу
		if !m.gpuKVStale {
			if err := m.gpu.AttentionScoresKV(layer, dst, q, seqLen, m.cfg.NumHeads, m.cfg.NumKVHeads, m.cfg.HeadDim); err == nil {
				return nil
			}
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
