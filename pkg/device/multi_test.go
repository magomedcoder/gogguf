package device

import (
	"fmt"
	"slices"
	"testing"

	"github.com/magomedcoder/gogguf/pkg/dtype"
)

// fakeDevice - device stub: counts calls and mimics residency (hidden on device, download clears it)
type fakeDevice struct {
	CPUBackend

	name      string
	resident  bool
	noResid   bool // HiddenResident() = false
	vramUsed  uint64
	vramTotal uint64

	hidden    []float32
	kvLayers  int
	uploads   int
	downloads int
	qkvLayers []int
	kvAppends []int
	logits    int
}

func newFakeDevice(name string) *fakeDevice {
	return &fakeDevice{name: name, vramUsed: 100, vramTotal: 4096}
}

func (f *fakeDevice) Name() string {
	return f.name
}

func (f *fakeDevice) VRAMInfo() (used, total uint64, err error) {
	return f.vramUsed, f.vramTotal, nil
}

func (f *fakeDevice) HiddenResident() bool {
	return !f.noResid
}

func (f *fakeDevice) HiddenActive() bool {
	return f.resident
}

func (f *fakeDevice) HiddenUpload(x []float32) error {
	f.hidden = slices.Clone(x)
	f.resident = true
	f.uploads++

	return nil
}

func (f *fakeDevice) HiddenDownload(dst []float32) error {
	if !f.resident {
		return fmt.Errorf("%s: residency not active", f.name)
	}

	copy(dst, f.hidden)
	f.resident = false
	f.downloads++

	return nil
}

func (f *fakeDevice) KVCacheInit(layers, _, _, _, _ int) error {
	f.kvLayers = layers
	return nil
}

func (f *fakeDevice) KVCacheAppend(layer, _ int, _, _ []float32) error {
	f.kvAppends = append(f.kvAppends, layer)
	return nil
}

func (f *fakeDevice) AttentionScoresKV(layer int, _, _ []float32, _, _, _, _ int) error {
	f.kvAppends = append(f.kvAppends, layer)
	return nil
}

func (f *fakeDevice) QKVRoPEAttentionQuantCached(_ dtype.DType, _ RoPEMode, _, _, _, _, _, _ string, _, _, _ []byte, _, _, _, _, _, _, _, _, _ []float32, _, _, _, _ int, layer, _, _ int, _ float32) error {
	f.qkvLayers = append(f.qkvLayers, layer)
	return nil
}

func (f *fakeDevice) LogitsFromDevice(dtype.DType, string, []float32, string, []byte, []float32, []float32, int, int, float32) error {
	f.logits++
	return nil
}

// qkv invokes fused QKV of layer via multi-backend
func qkv(m *MultiBackend, layer int) error {
	return m.QKVRoPEAttentionQuantCached(dtype.Q8_0, RoPENeoX, "q", "k", "v", "qn", "kn", "an", nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, 4, 1, 1, 4, layer, 0, 1, 1e-6)
}

func TestNewMultiBackendErrors(t *testing.T) {
	if _, err := NewMultiBackend(nil, nil); err == nil {
		t.Fatal("NewMultiBackend(nil): expected error")
	}

	if _, err := NewMultiBackend([]Backend{nil}, nil); err == nil {
		t.Fatal("NewMultiBackend([nil]): expected error")
	}

	dev := newFakeDevice("CUDA:0")
	if _, err := NewMultiBackend([]Backend{dev}, []float64{0.5, 0.5}); err == nil {
		t.Fatal("NewMultiBackend: expected tensor-split length error")
	}
}

// Single device list: transparent routing, no hidden transfers
func TestMultiBackendSingleDevice(t *testing.T) {
	dev := newFakeDevice("CUDA:0")
	m, err := NewMultiBackend([]Backend{dev}, nil)
	if err != nil {
		t.Fatal(err)
	}

	if err := m.KVCacheInit(4, 128, 8, 2, 4); err != nil {
		t.Fatal(err)
	}

	if dev.kvLayers != 4 {
		t.Fatalf("KVCacheInit: layers on device %d, expected 4", dev.kvLayers)
	}

	if err := m.HiddenUpload([]float32{1, 2, 3, 4}); err != nil {
		t.Fatal(err)
	}

	for layer := range 4 {
		if err := qkv(m, layer); err != nil {
			t.Fatal(err)
		}
	}

	if !slices.Equal(dev.qkvLayers, []int{0, 1, 2, 3}) {
		t.Fatalf("local layers = %v, expected 0..3", dev.qkvLayers)
	}

	if dev.uploads != 1 || dev.downloads != 0 {
		t.Fatalf("uploads=%d downloads=%d, expected 1/0 (no transfers)", dev.uploads, dev.downloads)
	}

	if err := m.LogitsFromDevice(dtype.Q8_0, "n", nil, "h", nil, nil, nil, 8, 4, 1e-6); err != nil {
		t.Fatal(err)
	}

	if dev.logits != 1 {
		t.Fatalf("LogitsFromDevice called %d times", dev.logits)
	}
}

// Two devices: layers split evenly, hidden moves exactly once at boundary
func TestMultiBackendLayerSplitMovesHidden(t *testing.T) {
	d0, d1 := newFakeDevice("CUDA:0"), newFakeDevice("CUDA:1")
	m, err := NewMultiBackend([]Backend{d0, d1}, nil)
	if err != nil {
		t.Fatal(err)
	}

	if err := m.KVCacheInit(4, 128, 8, 2, 4); err != nil {
		t.Fatal(err)
	}

	if d0.kvLayers != 2 || d1.kvLayers != 2 {
		t.Fatalf("KV per device = %d/%d, expected 2/2", d0.kvLayers, d1.kvLayers)
	}

	x := []float32{1, 2, 3, 4}
	if err := m.HiddenUpload(x); err != nil {
		t.Fatal(err)
	}

	for layer := range 4 {
		if err := qkv(m, layer); err != nil {
			t.Fatal(err)
		}

		if err := m.KVCacheAppend(layer, 0, []float32{1}, []float32{2}); err != nil {
			t.Fatal(err)
		}
	}

	// Global layers 0,1 -> local 0,1 on GPU0; layers 2,3 -> local 0,1 on GPU1
	if !slices.Equal(d0.qkvLayers, []int{0, 1}) || !slices.Equal(d1.qkvLayers, []int{0, 1}) {
		t.Fatalf("local layers: GPU0=%v GPU1=%v", d0.qkvLayers, d1.qkvLayers)
	}

	if !slices.Equal(d0.kvAppends, []int{0, 1}) || !slices.Equal(d1.kvAppends, []int{0, 1}) {
		t.Fatalf("KV append: GPU0=%v GPU1=%v", d0.kvAppends, d1.kvAppends)
	}

	if d0.uploads != 1 || d0.downloads != 1 {
		t.Fatalf("GPU0: uploads=%d downloads=%d, expected 1/1", d0.uploads, d0.downloads)
	}

	if d1.uploads != 1 || d1.downloads != 0 {
		t.Fatalf("GPU1: uploads=%d downloads=%d, expected 1/0", d1.uploads, d1.downloads)
	}

	if !slices.Equal(d1.hidden, x) {
		t.Fatalf("hidden on GPU1 = %v, expected %v", d1.hidden, x)
	}

	// Last layer's device computes token result
	if err := m.LogitsFromDevice(dtype.Q8_0, "n", nil, "h", nil, nil, nil, 8, 4, 1e-6); err != nil {
		t.Fatal(err)
	}

	if d0.logits != 0 || d1.logits != 1 {
		t.Fatalf("logits: GPU0=%d GPU1=%d, expected 0/1", d0.logits, d1.logits)
	}

	// New token returns hidden to first layer's device and clears GPU1 residency
	if err := m.HiddenUpload(x); err != nil {
		t.Fatal(err)
	}

	if d0.uploads != 2 || d1.resident {
		t.Fatalf("after new token: GPU0 uploads=%d, GPU1 resident=%v", d0.uploads, d1.resident)
	}
}

// -tensor-split sets unequal layer ranges
func TestMultiBackendTensorSplit(t *testing.T) {
	d0, d1 := newFakeDevice("CUDA:0"), newFakeDevice("CUDA:1")
	m, err := NewMultiBackend([]Backend{d0, d1}, []float64{0.75, 0.25})
	if err != nil {
		t.Fatal(err)
	}

	if err := m.KVCacheInit(28, 128, 8, 2, 4); err != nil {
		t.Fatal(err)
	}

	if d0.kvLayers != 21 || d1.kvLayers != 7 {
		t.Fatalf("KV per device = %d/%d, expected 21/7", d0.kvLayers, d1.kvLayers)
	}

	if err := m.HiddenUpload([]float32{1, 2, 3, 4}); err != nil {
		t.Fatal(err)
	}

	if err := qkv(m, 20); err != nil {
		t.Fatal(err)
	}

	if err := qkv(m, 21); err != nil {
		t.Fatal(err)
	}

	if !slices.Equal(d0.qkvLayers, []int{20}) || !slices.Equal(d1.qkvLayers, []int{0}) {
		t.Fatalf("split boundaries: GPU0=%v GPU1=%v", d0.qkvLayers, d1.qkvLayers)
	}
}

func TestMultiBackendVRAMInfoSums(t *testing.T) {
	d0, d1 := newFakeDevice("CUDA:0"), newFakeDevice("CUDA:1")
	d1.vramUsed, d1.vramTotal = 200, 8192
	m, err := NewMultiBackend([]Backend{d0, d1}, nil)
	if err != nil {
		t.Fatal(err)
	}

	used, total, err := m.VRAMInfo()
	if err != nil {
		t.Fatal(err)
	}

	if used != 300 || total != 12288 {
		t.Fatalf("VRAMInfo = %d/%d, expected 300/12288", used, total)
	}
}

// Residency available only if all devices support it (otherwise hidden cannot be transferred)
func TestMultiBackendHiddenResidentRequiresAll(t *testing.T) {
	d0, d1 := newFakeDevice("CUDA:0"), newFakeDevice("CUDA:1")
	m, err := NewMultiBackend([]Backend{d0, d1}, nil)
	if err != nil {
		t.Fatal(err)
	}

	if !m.HiddenResident() {
		t.Fatal("HiddenResident() = false with two resident devices")
	}

	d1.noResid = true
	if m.HiddenResident() {
		t.Fatal("HiddenResident() = true, although GPU1 does not hold hidden")
	}
}

func TestMultiBackendPlanAndName(t *testing.T) {
	d0, d1 := newFakeDevice("CUDA:0"), newFakeDevice("CUDA:1")
	m, err := NewMultiBackend([]Backend{d0, d1}, nil)
	if err != nil {
		t.Fatal(err)
	}

	if err := m.KVCacheInit(4, 128, 8, 2, 4); err != nil {
		t.Fatal(err)
	}

	plan := Describe(m)
	if plan != "CUDA:0: layers 0-1, CUDA:1: layers 2-3" {
		t.Fatalf("Describe = %q", plan)
	}

	if m.Name() != "MULTI[CUDA:0 | CUDA:1]" {
		t.Fatalf("Name = %q", m.Name())
	}

	if got := Describe(CPUBackend{}); got != "CPU" {
		t.Fatalf("Describe(CPU) = %q", got)
	}

	if got := Describe(nil); got != "CPU" {
		t.Fatalf("Describe(nil) = %q", got)
	}
}

// fakePeer mimics a P2P move between fakeDevice instances: no HiddenDownload/HiddenUpload counters change
func fakePeer(calls *int, fail error) PeerHiddenFunc {
	return func(src, dst Backend, n int) error {
		*calls++
		if fail != nil {
			return fail
		}

		s, ok := src.(*fakeDevice)
		d, ok2 := dst.(*fakeDevice)
		if !ok || !ok2 {
			return fmt.Errorf("fakePeer: unexpected backend types")
		}

		if !s.resident || len(s.hidden) != n {
			return fmt.Errorf("fakePeer: source not resident")
		}

		d.hidden = slices.Clone(s.hidden)
		d.resident = true
		s.resident = false

		return nil
	}
}

func runTokenThroughBoundary(t *testing.T, m *MultiBackend, x []float32) {
	t.Helper()

	if err := m.HiddenUpload(x); err != nil {
		t.Fatal(err)
	}

	for layer := range 4 {
		if err := qkv(m, layer); err != nil {
			t.Fatal(err)
		}
	}
}

// Boundary move uses the peer path: no host download/upload, hidden arrives intact
func TestMultiBackendPeerHiddenSkipsHost(t *testing.T) {
	d0, d1 := newFakeDevice("CUDA:0"), newFakeDevice("CUDA:1")
	m, err := NewMultiBackend([]Backend{d0, d1}, nil)
	if err != nil {
		t.Fatal(err)
	}

	var calls int
	m.SetPeerHidden(fakePeer(&calls, nil))

	if err := m.KVCacheInit(4, 128, 8, 2, 4); err != nil {
		t.Fatal(err)
	}

	x := []float32{1, 2, 3, 4}
	runTokenThroughBoundary(t, m, x)

	if calls != 1 {
		t.Fatalf("peer calls = %d, expected 1", calls)
	}

	if d0.downloads != 0 || d1.uploads != 0 {
		t.Fatalf("host bounce used: GPU0 downloads=%d GPU1 uploads=%d, expected 0/0", d0.downloads, d1.uploads)
	}

	if !slices.Equal(d1.hidden, x) || !d1.resident || d0.resident {
		t.Fatalf("hidden after peer move: GPU1=%v resident(GPU1=%v GPU0=%v)", d1.hidden, d1.resident, d0.resident)
	}

	if p, h := m.TransferStats(); p != 1 || h != 0 {
		t.Fatalf("TransferStats = %d/%d, expected 1/0", p, h)
	}

	// Next token: stale residency on GPU1 is cleared, boundary uses peer again
	runTokenThroughBoundary(t, m, x)

	if calls != 2 {
		t.Fatalf("peer calls after 2nd token = %d, expected 2", calls)
	}

	if p, h := m.TransferStats(); p != 2 || h != 0 {
		t.Fatalf("TransferStats = %d/%d, expected 2/0", p, h)
	}
}

// Failed peer copy: host bounce takes over with correct data and the pair is not retried every token
func TestMultiBackendPeerFailureFallsBackToHost(t *testing.T) {
	d0, d1 := newFakeDevice("CUDA:0"), newFakeDevice("CUDA:1")
	m, err := NewMultiBackend([]Backend{d0, d1}, nil)
	if err != nil {
		t.Fatal(err)
	}

	var calls int
	m.SetPeerHidden(fakePeer(&calls, fmt.Errorf("no P2P between these GPUs")))

	if err := m.KVCacheInit(4, 128, 8, 2, 4); err != nil {
		t.Fatal(err)
	}

	x := []float32{5, 6, 7, 8}
	runTokenThroughBoundary(t, m, x)
	runTokenThroughBoundary(t, m, x)

	if calls != 1 {
		t.Fatalf("peer attempted %d times, expected exactly 1 (pair marked broken)", calls)
	}

	if d0.downloads != 2 || d1.uploads != 2 {
		t.Fatalf("host fallback: GPU0 downloads=%d GPU1 uploads=%d, expected 2/2", d0.downloads, d1.uploads)
	}

	if !slices.Equal(d1.hidden, x) {
		t.Fatalf("hidden on GPU1 = %v, expected %v", d1.hidden, x)
	}

	if p, h := m.TransferStats(); p != 0 || h != 2 {
		t.Fatalf("TransferStats = %d/%d, expected 0/2", p, h)
	}
}

// SetPeerHidden(nil) forces host bounce; no peer mover (non-CUDA devices) behaves like before
func TestMultiBackendPeerDisabled(t *testing.T) {
	d0, d1 := newFakeDevice("CUDA:0"), newFakeDevice("CUDA:1")
	m, err := NewMultiBackend([]Backend{d0, d1}, nil)
	if err != nil {
		t.Fatal(err)
	}

	var calls int
	m.SetPeerHidden(fakePeer(&calls, nil))
	m.SetPeerHidden(nil)

	if err := m.KVCacheInit(4, 128, 8, 2, 4); err != nil {
		t.Fatal(err)
	}

	runTokenThroughBoundary(t, m, []float32{1, 2, 3, 4})

	if calls != 0 || d0.downloads != 1 || d1.uploads != 1 {
		t.Fatalf("calls=%d downloads=%d uploads=%d, expected host bounce 0/1/1", calls, d0.downloads, d1.uploads)
	}
}

// No hidden resident on the source (host path models): the peer mover must not be invoked
func TestMultiBackendPeerSkippedWithoutResidency(t *testing.T) {
	d0, d1 := newFakeDevice("CUDA:0"), newFakeDevice("CUDA:1")
	m, err := NewMultiBackend([]Backend{d0, d1}, nil)
	if err != nil {
		t.Fatal(err)
	}

	var calls int
	m.SetPeerHidden(fakePeer(&calls, nil))

	if err := m.KVCacheInit(4, 128, 8, 2, 4); err != nil {
		t.Fatal(err)
	}

	for layer := range 4 {
		if err := qkv(m, layer); err != nil {
			t.Fatal(err)
		}
	}

	if calls != 0 {
		t.Fatalf("peer mover called %d times without resident hidden", calls)
	}
}
