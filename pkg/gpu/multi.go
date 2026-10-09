package gpu

import (
	"errors"
	"fmt"
	"math"
	"slices"
	"strings"
	"sync"

	"github.com/magomedcoder/gogguf/pkg/format"
)

// MultiBackend distributes transformer layers across multiple devices:
// layers 0..n1-1 on first GPU, n1..n2-1 on second, etc.
// At device boundaries hidden state moves via host (HiddenDownload + HiddenUpload); within one device's range residual stays resident.
//
// Limitation: each device's weights and KV are cached independently, so total VRAM is sum per device, not shared.
// Layers run sequentially (pipeline parallelism without parallel execution).
type MultiBackend struct {
	devs  []Backend
	split []float64

	mu        sync.Mutex
	bounds    []int // bounds[i] - first layer of device i, len = len(devs)+1
	layers    int
	cur       int       // index of device with current hidden state
	hidden    []float32 // buffer for hidden transfer at device boundaries
	hiddenLen int
}

var _ Backend = (*MultiBackend)(nil)

// NewMultiBackend combines devices into one Backend with layer split.
// split sets layer proportions per device (like -tensor-split); nil = even
func NewMultiBackend(devs []Backend, split []float64) (*MultiBackend, error) {
	if len(devs) == 0 {
		return nil, errors.New("gpu: multi: пустой список устройств")
	}

	for i, d := range devs {
		if d == nil {
			return nil, fmt.Errorf("gpu: multi: устройство %d = nil", i)
		}
	}

	if split != nil && len(split) != len(devs) {
		return nil, fmt.Errorf("gpu: multi: tensor-split из %d значений при %d устройствах", len(split), len(devs))
	}

	return &MultiBackend{devs: devs, split: split}, nil
}

// LayerSplitBounds splits layers across n devices.
// Returns n+1 bounds: device i computes layers bounds[i]..bounds[i+1]-1.
// split - proportion weights (nil/invalid = even)
func LayerSplitBounds(layers, n int, split []float64) []int {
	if n <= 0 {
		return nil
	}

	if layers < 0 {
		layers = 0
	}

	bounds := make([]int, n+1)

	var total float64
	if len(split) == n {
		for _, w := range split {
			if w > 0 {
				total += w
			}
		}
	}

	if total <= 0 {
		// Even split: first layers%n devices get one extra layer
		base, rem := layers/n, layers%n
		pos := 0
		for i := range n {
			pos += base
			if i < rem {
				pos++
			}

			bounds[i+1] = pos
		}

		return bounds
	}

	var acc float64
	for i := range n {
		if w := split[i]; w > 0 {
			acc += w
		}

		b := int(math.Round(acc / total * float64(layers)))
		bounds[i+1] = min(max(b, bounds[i]), layers)
	}

	bounds[n] = layers

	return bounds
}

// Devices returns devices in layer order
func (m *MultiBackend) Devices() []Backend {
	return m.devs
}

// Plan describes layer distribution across devices (for logs)
func (m *MultiBackend) Plan() string {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.bounds == nil {
		var names []string
		for _, d := range m.devs {
			names = append(names, d.Name())
		}

		return strings.Join(names, " + ") + " (слои не распределены)"
	}

	parts := make([]string, 0, len(m.devs))
	for i, d := range m.devs {
		from, to := m.bounds[i], m.bounds[i+1]
		if to <= from {
			parts = append(parts, fmt.Sprintf("%s: без слоёв", d.Name()))
			continue
		}

		parts = append(parts, fmt.Sprintf("%s: слои %d-%d", d.Name(), from, to-1))
	}

	return strings.Join(parts, ", ")
}

func (m *MultiBackend) Name() string {
	names := make([]string, 0, len(m.devs))
	for _, d := range m.devs {
		names = append(names, d.Name())
	}

	return fmt.Sprintf("MULTI[%s]", strings.Join(names, " | "))
}

// VRAMInfo sums memory of all devices
func (m *MultiBackend) VRAMInfo() (used, total uint64, err error) {
	for _, d := range m.devs {
		u, t, err := d.VRAMInfo()
		if err != nil {
			return 0, 0, err
		}

		used += u
		total += t
	}

	return used, total, nil
}

// deviceForLayer returns device index computing layer (under m.mu)
func (m *MultiBackend) deviceForLayer(layer int) int {
	if m.bounds == nil || layer < 0 {
		return 0
	}

	for i := range m.devs {
		if layer >= m.bounds[i] && layer < m.bounds[i+1] {
			return i
		}
	}

	// Layer outside plan (e.g. KV wider than ngl): assign to last non-empty device
	for i := range slices.Backward(m.devs) {
		if m.bounds[i+1] > m.bounds[i] {
			return i
		}
	}

	return 0
}

// localLayer maps global layer index to index within device KV-cache
func (m *MultiBackend) localLayer(idx, layer int) int {
	if m.bounds == nil {
		return layer
	}

	return layer - m.bounds[idx]
}

// route selects device for layer, transferring hidden state on device change
func (m *MultiBackend) route(layer int) (Backend, int, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	idx := m.deviceForLayer(layer)
	if idx != m.cur {
		if err := m.moveHiddenLocked(idx); err != nil {
			return nil, 0, err
		}
	}

	return m.devs[idx], m.localLayer(idx, layer), nil
}

// moveHiddenLocked transfers resident hidden state from current device to dst
func (m *MultiBackend) moveHiddenLocked(dst int) error {
	src := m.devs[m.cur]
	if m.hiddenLen > 0 && src.HiddenActive() {
		if cap(m.hidden) < m.hiddenLen {
			m.hidden = make([]float32, m.hiddenLen)
		}

		buf := m.hidden[:m.hiddenLen]
		if err := src.HiddenDownload(buf); err != nil {
			return fmt.Errorf("gpu: multi: hidden %s -> host: %w", src.Name(), err)
		}

		if err := m.devs[dst].HiddenUpload(buf); err != nil {
			return fmt.Errorf("gpu: multi: hidden host -> %s: %w", m.devs[dst].Name(), err)
		}
	}

	m.cur = dst

	return nil
}

// current returns the device holding the current hidden state
func (m *MultiBackend) current() Backend {
	m.mu.Lock()
	defer m.mu.Unlock()

	return m.devs[m.cur]
}

func (m *MultiBackend) MatMulVec(matrix []float32, rows, cols int, vec []float32) ([]float32, error) {
	return m.current().MatMulVec(matrix, rows, cols, vec)
}

func (m *MultiBackend) MatMulVecCached(name string, matrix []float32, rows, cols int, vec []float32) ([]float32, error) {
	return m.current().MatMulVecCached(name, matrix, rows, cols, vec)
}

func (m *MultiBackend) MatMulVecQ8_0Cached(name string, raw []byte, rows, cols int, vec []float32) ([]float32, error) {
	return m.current().MatMulVecQ8_0Cached(name, raw, rows, cols, vec)
}

func (m *MultiBackend) MatMulVecQ4_0Cached(name string, raw []byte, rows, cols int, vec []float32) ([]float32, error) {
	return m.current().MatMulVecQ4_0Cached(name, raw, rows, cols, vec)
}

func (m *MultiBackend) MatMulVecQ4_KCached(name string, raw []byte, rows, cols int, vec []float32) ([]float32, error) {
	return m.current().MatMulVecQ4_KCached(name, raw, rows, cols, vec)
}

func (m *MultiBackend) MatMulVecQ5_KCached(name string, raw []byte, rows, cols int, vec []float32) ([]float32, error) {
	return m.current().MatMulVecQ5_KCached(name, raw, rows, cols, vec)
}

func (m *MultiBackend) MatMulVecQ6_KCached(name string, raw []byte, rows, cols int, vec []float32) ([]float32, error) {
	return m.current().MatMulVecQ6_KCached(name, raw, rows, cols, vec)
}

func (m *MultiBackend) RMSNormInto(dst, x, weight []float32, eps float32) error {
	return m.current().RMSNormInto(dst, x, weight, eps)
}

func (m *MultiBackend) ApplyRoPEHeads(v []float32, nHeads, headDim, pos int, freqBase float32) error {
	return m.current().ApplyRoPEHeads(v, nHeads, headDim, pos, freqBase)
}

func (m *MultiBackend) ApplyRoPEHeadsNorm(v []float32, nHeads, headDim, pos int, freqBase float32) error {
	return m.current().ApplyRoPEHeadsNorm(v, nHeads, headDim, pos, freqBase)
}

func (m *MultiBackend) SwiGLUInPlace(gate, up []float32) error {
	return m.current().SwiGLUInPlace(gate, up)
}

func (m *MultiBackend) FFNSwiGLUCached(gateName, upName, downName string, gateW, upW, downW, x, out []float32, embd, ffn int) error {
	return m.current().FFNSwiGLUCached(gateName, upName, downName, gateW, upW, downW, x, out, embd, ffn)
}

func (m *MultiBackend) FFNSwiGLUQ8_0Cached(gateName, upName, downName string, gateRaw, upRaw, downRaw []byte, x, out []float32, embd, ffn int) error {
	return m.current().FFNSwiGLUQ8_0Cached(gateName, upName, downName, gateRaw, upRaw, downRaw, x, out, embd, ffn)
}

func (m *MultiBackend) AttnFFNResidualCached(woName, ffnNormName, gateName, upName, downName string, woW, ffnNorm, gateW, upW, downW, x, attn []float32, embd, attnDim, ffn int, eps float32) error {
	return m.current().AttnFFNResidualCached(woName, ffnNormName, gateName, upName, downName, woW, ffnNorm, gateW, upW, downW, x, attn, embd, attnDim, ffn, eps)
}

func (m *MultiBackend) AttnFFNResidualQ8_0Cached(woName, ffnNormName, gateName, upName, downName string, woRaw, gateRaw, upRaw, downRaw []byte, ffnNorm, x, attn []float32, embd, attnDim, ffn int, eps float32) error {
	return m.current().AttnFFNResidualQ8_0Cached(woName, ffnNormName, gateName, upName, downName, woRaw, gateRaw, upRaw, downRaw, ffnNorm, x, attn, embd, attnDim, ffn, eps)
}

func (m *MultiBackend) FFNSwiGLUQuantCached(t format.GGML, gateName, upName, downName string, gateRaw, upRaw, downRaw []byte, x, out []float32, embd, ffn int) error {
	return m.current().FFNSwiGLUQuantCached(t, gateName, upName, downName, gateRaw, upRaw, downRaw, x, out, embd, ffn)
}

func (m *MultiBackend) FFNGeGLUQuantCached(t format.GGML, gateName, upName, downName string, gateRaw, upRaw, downRaw []byte, x, out []float32, embd, ffn int) error {
	return m.current().FFNGeGLUQuantCached(t, gateName, upName, downName, gateRaw, upRaw, downRaw, x, out, embd, ffn)
}

func (m *MultiBackend) AttnFFNResidualQuantCached(t format.GGML, woName, ffnNormName, gateName, upName, downName string, woRaw, gateRaw, upRaw, downRaw []byte, ffnNorm, x, attn []float32, embd, attnDim, ffn int, eps float32) error {
	return m.current().AttnFFNResidualQuantCached(t, woName, ffnNormName, gateName, upName, downName, woRaw, gateRaw, upRaw, downRaw, ffnNorm, x, attn, embd, attnDim, ffn, eps)
}

// QKVRoPEAttentionCached routes layer to its device (boundaries - hidden transfer)
func (m *MultiBackend) QKVRoPEAttentionCached(qName, kName, vName, qNormName, kNormName string, qW, kW, vW, qNorm, kNorm, h, cos, sin, attn, kOut, vOut []float32, embd, nHeads, nKVHeads, headDim, layer, kvPos, seqLen int, eps float32) error {
	dev, local, err := m.route(layer)
	if err != nil {
		return err
	}

	return dev.QKVRoPEAttentionCached(qName, kName, vName, qNormName, kNormName, qW, kW, vW, qNorm, kNorm, h, cos, sin, attn, kOut, vOut, embd, nHeads, nKVHeads, headDim, local, kvPos, seqLen, eps)
}

func (m *MultiBackend) QKVRoPEAttentionQ8_0Cached(qName, kName, vName, qNormName, kNormName string, qRaw, kRaw, vRaw []byte, qNorm, kNorm, h, cos, sin, attn, kOut, vOut []float32, embd, nHeads, nKVHeads, headDim, layer, kvPos, seqLen int, eps float32) error {
	dev, local, err := m.route(layer)
	if err != nil {
		return err
	}

	return dev.QKVRoPEAttentionQ8_0Cached(qName, kName, vName, qNormName, kNormName, qRaw, kRaw, vRaw, qNorm, kNorm, h, cos, sin, attn, kOut, vOut, embd, nHeads, nKVHeads, headDim, local, kvPos, seqLen, eps)
}

func (m *MultiBackend) QKVRoPEAttentionQuantCached(t format.GGML, mode RoPEMode, qName, kName, vName, qNormName, kNormName, attnNormName string, qRaw, kRaw, vRaw []byte, qNorm, kNorm, attnNorm, h, cos, sin, attn, kOut, vOut []float32, embd, nHeads, nKVHeads, headDim, layer, kvPos, seqLen int, eps float32) error {
	dev, local, err := m.route(layer)
	if err != nil {
		return err
	}

	return dev.QKVRoPEAttentionQuantCached(t, mode, qName, kName, vName, qNormName, kNormName, attnNormName, qRaw, kRaw, vRaw, qNorm, kNorm, attnNorm, h, cos, sin, attn, kOut, vOut, embd, nHeads, nKVHeads, headDim, local, kvPos, seqLen, eps)
}

// HiddenResident: residency is possible only if all devices support it (otherwise hidden cannot cross boundaries)
func (m *MultiBackend) HiddenResident() bool {
	for _, d := range m.devs {
		if !d.HiddenResident() {
			return false
		}
	}

	return true
}

func (m *MultiBackend) HiddenActive() bool {
	return m.current().HiddenActive()
}

// HiddenUpload puts hidden on the first layer's device; others clear residency
func (m *MultiBackend) HiddenUpload(x []float32) error {
	if len(x) == 0 {
		return fmt.Errorf("gpu: multi: HiddenUpload: пустой x")
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	m.hiddenLen = len(x)
	first := m.deviceForLayer(0)

	// Stale residual on other devices cleared via download to scratch
	for i, d := range m.devs {
		if i == first || !d.HiddenActive() {
			continue
		}

		if cap(m.hidden) < len(x) {
			m.hidden = make([]float32, len(x))
		}

		_ = d.HiddenDownload(m.hidden[:len(x)])
	}

	m.cur = first

	return m.devs[first].HiddenUpload(x)
}

func (m *MultiBackend) HiddenDownload(dst []float32) error {
	return m.current().HiddenDownload(dst)
}

func (m *MultiBackend) LogitsFromDevice(t format.GGML, normName string, norm []float32, headName string, headRaw []byte, headF32, logits []float32, vocab, embd int, eps float32) error {
	return m.current().LogitsFromDevice(t, normName, norm, headName, headRaw, headF32, logits, vocab, embd, eps)
}

func (m *MultiBackend) AttentionScoresInto(dst, q, k, v, scores []float32, seqLen, nHeads, nKVHeads, headDim int) error {
	return m.current().AttentionScoresInto(dst, q, k, v, scores, seqLen, nHeads, nKVHeads, headDim)
}

// KVCacheInit builds the layer plan and allocates KV on each device for its range
func (m *MultiBackend) KVCacheInit(layers, maxSeq, kvDim, nHeads, headDim int) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	bounds := LayerSplitBounds(layers, len(m.devs), m.split)
	for i, d := range m.devs {
		n := bounds[i+1] - bounds[i]
		if n <= 0 {
			continue
		}

		if err := d.KVCacheInit(n, maxSeq, kvDim, nHeads, headDim); err != nil {
			return fmt.Errorf("gpu: multi: kv init %s: %w", d.Name(), err)
		}
	}

	m.bounds = bounds
	m.layers = layers
	m.cur = m.deviceForLayer(0)

	return nil
}

func (m *MultiBackend) KVCacheReset() {
	for _, d := range m.devs {
		d.KVCacheReset()
	}
}

func (m *MultiBackend) KVCacheAppend(layer, pos int, k, v []float32) error {
	dev, local, err := m.route(layer)
	if err != nil {
		return err
	}

	return dev.KVCacheAppend(local, pos, k, v)
}

func (m *MultiBackend) KVCacheAppendN(layer, pos int, k, v []float32, n int) error {
	dev, local, err := m.route(layer)
	if err != nil {
		return err
	}

	return dev.KVCacheAppendN(local, pos, k, v, n)
}

func (m *MultiBackend) AttentionScoresKV(layer int, dst, q []float32, seqLen, nHeads, nKVHeads, headDim int) error {
	dev, local, err := m.route(layer)
	if err != nil {
		return err
	}

	return dev.AttentionScoresKV(local, dst, q, seqLen, nHeads, nKVHeads, headDim)
}

func (m *MultiBackend) Close() error {
	var errs []error
	for _, d := range m.devs {
		if err := d.Close(); err != nil {
			errs = append(errs, err)
		}
	}

	return errors.Join(errs...)
}

// Describe returns the backend name; for multi-GPU, the layer plan
func Describe(b Backend) string {
	if b == nil {
		return "CPU"
	}

	if p, ok := b.(interface{ Plan() string }); ok {
		return p.Plan()
	}

	return b.Name()
}
