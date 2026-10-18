package device

import (
	"errors"
	"fmt"
	"math"
	"slices"
	"strings"
	"sync"

	"github.com/magomedcoder/gogguf/pkg/dtype"
)

// MultiBackend distributes transformer layers across multiple devices: layers 0..n1-1 on first GPU, n1..n2-1 on second, etc.
// At device boundaries the resident hidden state moves device-to-device when the pair supports P2P (CUDA cuMemcpyPeer, see PeerHiddenFunc); otherwise via host (HiddenDownload + HiddenUpload). Within one device's range residual stays resident.
//
// Limitation: each device's weights and KV are cached independently, so total VRAM is sum per device, not shared.
// Layers run sequentially: a boundary peer copy replaces the host bounce but does not overlap with compute on the other device (a token's layers form a strict dependency chain; real overlap would need micro-batch pipelining across tokens/sequences).
type MultiBackend struct {
	devs  []Backend
	split []float64

	peer       PeerHiddenFunc  // nil = host bounce only
	peerBroken map[[2]int]bool // pairs whose peer copy failed once: host bounce from then on
	peerCopies int             // boundary moves done over P2P
	hostCopies int             // boundary moves done via host

	mu        sync.Mutex
	bounds    []int // bounds[i] - first layer of device i, len = len(devs)+1
	layers    int
	cur       int       // index of device with current hidden state
	hidden    []float32 // buffer for hidden transfer at device boundaries
	hiddenLen int
}

var _ Backend = (*MultiBackend)(nil)

// PeerHiddenFunc copies the resident hidden state (n floats) from src to dst without touching host memory.
// Returns nil when src's residency moved to dst (src residency cleared, dst active). Any error means "not moved": src still holds the hidden state and MultiBackend falls back to HiddenDownload + HiddenUpload.
type PeerHiddenFunc func(src, dst Backend, n int) error

// defaultPeerHidden is set by build-tagged files (cuda_open.go) when a peer-capable backend exists
var defaultPeerHidden PeerHiddenFunc

// SetPeerHidden overrides the device-to-device hidden mover (nil disables P2P: host bounce only)
func (m *MultiBackend) SetPeerHidden(f PeerHiddenFunc) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.peer = f
	m.peerBroken = nil
}

// TransferStats reports how many hidden-state moves between devices used P2P vs the host bounce
func (m *MultiBackend) TransferStats() (peer, host int) {
	m.mu.Lock()
	defer m.mu.Unlock()

	return m.peerCopies, m.hostCopies
}

// NewMultiBackend combines devices into one Backend with layer split.
// split sets layer proportions per device (like -tensor-split); nil = even
func NewMultiBackend(devs []Backend, split []float64) (*MultiBackend, error) {
	if len(devs) == 0 {
		return nil, errors.New("device: multi: empty device list")
	}

	for i, d := range devs {
		if d == nil {
			return nil, fmt.Errorf("device: multi: device %d is nil", i)
		}
	}

	if split != nil && len(split) != len(devs) {
		return nil, fmt.Errorf("device: multi: tensor-split has %d values for %d devices", len(split), len(devs))
	}

	return &MultiBackend{devs: devs, split: split, peer: defaultPeerHidden}, nil
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

		return strings.Join(names, " + ") + " (layers not assigned)"
	}

	parts := make([]string, 0, len(m.devs))
	for i, d := range m.devs {
		from, to := m.bounds[i], m.bounds[i+1]
		if to <= from {
			parts = append(parts, fmt.Sprintf("%s: no layers", d.Name()))
			continue
		}

		parts = append(parts, fmt.Sprintf("%s: layers %d-%d", d.Name(), from, to-1))
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

// moveHiddenLocked transfers resident hidden state from current device to dst.
// Prefers a device-to-device copy (P2P); falls back to the host bounce when peers are unsupported or the copy fails.
func (m *MultiBackend) moveHiddenLocked(dst int) error {
	src := m.devs[m.cur]
	if m.hiddenLen > 0 && src.HiddenActive() {
		if m.tryPeerLocked(m.cur, dst) {
			m.peerCopies++
			m.cur = dst

			return nil
		}

		if cap(m.hidden) < m.hiddenLen {
			m.hidden = make([]float32, m.hiddenLen)
		}

		buf := m.hidden[:m.hiddenLen]
		if err := src.HiddenDownload(buf); err != nil {
			return fmt.Errorf("device: multi: hidden %s -> host: %w", src.Name(), err)
		}

		if err := m.devs[dst].HiddenUpload(buf); err != nil {
			return fmt.Errorf("device: multi: hidden host -> %s: %w", m.devs[dst].Name(), err)
		}

		m.hostCopies++
	}

	m.cur = dst

	return nil
}

// tryPeerLocked attempts the P2P hidden move src -> dst; false = caller must use the host path (src residency untouched)
func (m *MultiBackend) tryPeerLocked(src, dst int) bool {
	if m.peer == nil || m.peerBroken[[2]int{src, dst}] {
		return false
	}

	if err := m.peer(m.devs[src], m.devs[dst], m.hiddenLen); err != nil {
		if m.peerBroken == nil {
			m.peerBroken = make(map[[2]int]bool)
		}

		// Remember both directions: topology/driver limits are symmetric, don't retry (and re-probe) every token
		m.peerBroken[[2]int{src, dst}] = true
		m.peerBroken[[2]int{dst, src}] = true

		return false
	}

	return true
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

func (m *MultiBackend) MatMulVecQ4_1Cached(name string, raw []byte, rows, cols int, vec []float32) ([]float32, error) {
	return m.current().MatMulVecQ4_1Cached(name, raw, rows, cols, vec)
}

func (m *MultiBackend) MatMulVecQ5_0Cached(name string, raw []byte, rows, cols int, vec []float32) ([]float32, error) {
	return m.current().MatMulVecQ5_0Cached(name, raw, rows, cols, vec)
}

func (m *MultiBackend) MatMulVecQ5_1Cached(name string, raw []byte, rows, cols int, vec []float32) ([]float32, error) {
	return m.current().MatMulVecQ5_1Cached(name, raw, rows, cols, vec)
}

func (m *MultiBackend) MatMulVecQ2_KCached(name string, raw []byte, rows, cols int, vec []float32) ([]float32, error) {
	return m.current().MatMulVecQ2_KCached(name, raw, rows, cols, vec)
}

func (m *MultiBackend) MatMulVecQ3_KCached(name string, raw []byte, rows, cols int, vec []float32) ([]float32, error) {
	return m.current().MatMulVecQ3_KCached(name, raw, rows, cols, vec)
}

func (m *MultiBackend) MatMulVecQ8_KCached(name string, raw []byte, rows, cols int, vec []float32) ([]float32, error) {
	return m.current().MatMulVecQ8_KCached(name, raw, rows, cols, vec)
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

func (m *MultiBackend) FFNSwiGLUQuantCached(t dtype.DType, gateName, upName, downName string, gateRaw, upRaw, downRaw []byte, x, out []float32, embd, ffn int) error {
	return m.current().FFNSwiGLUQuantCached(t, gateName, upName, downName, gateRaw, upRaw, downRaw, x, out, embd, ffn)
}

func (m *MultiBackend) FFNGeGLUQuantCached(t dtype.DType, gateName, upName, downName string, gateRaw, upRaw, downRaw []byte, x, out []float32, embd, ffn int) error {
	return m.current().FFNGeGLUQuantCached(t, gateName, upName, downName, gateRaw, upRaw, downRaw, x, out, embd, ffn)
}

func (m *MultiBackend) AttnFFNResidualQuantCached(t dtype.DType, woName, ffnNormName, gateName, upName, downName string, woRaw, gateRaw, upRaw, downRaw []byte, ffnNorm, x, attn []float32, embd, attnDim, ffn int, eps float32) error {
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

func (m *MultiBackend) QKVRoPEAttentionQuantCached(t dtype.DType, mode RoPEMode, qName, kName, vName, qNormName, kNormName, attnNormName string, qRaw, kRaw, vRaw []byte, qNorm, kNorm, attnNorm, h, cos, sin, attn, kOut, vOut []float32, embd, nHeads, nKVHeads, headDim, layer, kvPos, seqLen int, eps float32) error {
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
		return fmt.Errorf("device: multi: HiddenUpload: empty x")
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

func (m *MultiBackend) LogitsFromDevice(t dtype.DType, normName string, norm []float32, headName string, headRaw []byte, headF32, logits []float32, vocab, embd int, eps float32) error {
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
			return fmt.Errorf("device: multi: kv init %s: %w", d.Name(), err)
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

// AttentionScoresKVBatch routes causal batch attention to the device owning layer (falls back to a per-query loop when that device lacks KVBatchAttentionBackend).
func (m *MultiBackend) AttentionScoresKVBatch(layer int, dst, q []float32, n, pastLen, nHeads, nKVHeads, headDim int) error {
	dev, local, err := m.route(layer)
	if err != nil {
		return err
	}

	return AttentionScoresKVBatch(dev, local, dst, q, n, pastLen, nHeads, nKVHeads, headDim)
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
