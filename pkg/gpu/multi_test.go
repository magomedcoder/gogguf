package gpu

import (
	"fmt"
	"slices"
	"testing"

	"github.com/magomedcoder/gogguf/pkg/format"
)

// fakeDevice - устройство-заглушка: считает вызовы и повторяет семантику residency (hidden живёт на устройстве, download его снимает)
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
		return fmt.Errorf("%s: residency не активна", f.name)
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

func (f *fakeDevice) QKVRoPEAttentionQuantCached(_ format.GGML, _ RoPEMode, _, _, _, _, _, _ string, _, _, _ []byte, _, _, _, _, _, _, _, _, _ []float32, _, _, _, _ int, layer, _, _ int, _ float32) error {
	f.qkvLayers = append(f.qkvLayers, layer)
	return nil
}

func (f *fakeDevice) LogitsFromDevice(format.GGML, string, []float32, string, []byte, []float32, []float32, int, int, float32) error {
	f.logits++
	return nil
}

// qkv дёргает fused QKV слоя layer через multi-backend
func qkv(m *MultiBackend, layer int) error {
	return m.QKVRoPEAttentionQuantCached(format.GgmlQ8_0, RoPENeoX, "q", "k", "v", "qn", "kn", "an",
		nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil,
		4, 1, 1, 4, layer, 0, 1, 1e-6)
}

func TestNewMultiBackendErrors(t *testing.T) {
	if _, err := NewMultiBackend(nil, nil); err == nil {
		t.Fatal("NewMultiBackend(nil): ожидали ошибку")
	}

	if _, err := NewMultiBackend([]Backend{nil}, nil); err == nil {
		t.Fatal("NewMultiBackend([nil]): ожидали ошибку")
	}

	dev := newFakeDevice("CUDA:0")
	if _, err := NewMultiBackend([]Backend{dev}, []float64{0.5, 0.5}); err == nil {
		t.Fatal("NewMultiBackend: ожидали ошибку длины tensor-split")
	}
}

// Список из одного устройства: маршрутизация прозрачная, переносов hidden нет
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
		t.Fatalf("KVCacheInit: слоёв на устройстве %d, ожидали 4", dev.kvLayers)
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
		t.Fatalf("локальные слои = %v, ожидали 0..3", dev.qkvLayers)
	}

	if dev.uploads != 1 || dev.downloads != 0 {
		t.Fatalf("uploads=%d downloads=%d, ожидали 1/0 (переносов нет)", dev.uploads, dev.downloads)
	}

	if err := m.LogitsFromDevice(format.GgmlQ8_0, "n", nil, "h", nil, nil, nil, 8, 4, 1e-6); err != nil {
		t.Fatal(err)
	}

	if dev.logits != 1 {
		t.Fatalf("LogitsFromDevice вызван %d раз", dev.logits)
	}
}

// Два устройства: слои делятся пополам, hidden переезжает ровно один раз на границе
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
		t.Fatalf("KV по устройствам = %d/%d, ожидали 2/2", d0.kvLayers, d1.kvLayers)
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

	// Глобальные слои 0,1 -> локальные 0,1 на GPU0; слои 2,3 -> локальные 0,1 на GPU1
	if !slices.Equal(d0.qkvLayers, []int{0, 1}) || !slices.Equal(d1.qkvLayers, []int{0, 1}) {
		t.Fatalf("локальные слои: GPU0=%v GPU1=%v", d0.qkvLayers, d1.qkvLayers)
	}

	if !slices.Equal(d0.kvAppends, []int{0, 1}) || !slices.Equal(d1.kvAppends, []int{0, 1}) {
		t.Fatalf("KV append: GPU0=%v GPU1=%v", d0.kvAppends, d1.kvAppends)
	}

	if d0.uploads != 1 || d0.downloads != 1 {
		t.Fatalf("GPU0: uploads=%d downloads=%d, ожидали 1/1", d0.uploads, d0.downloads)
	}

	if d1.uploads != 1 || d1.downloads != 0 {
		t.Fatalf("GPU1: uploads=%d downloads=%d, ожидали 1/0", d1.uploads, d1.downloads)
	}

	if !slices.Equal(d1.hidden, x) {
		t.Fatalf("hidden на GPU1 = %v, ожидали %v", d1.hidden, x)
	}

	// Итог токена считает устройство последнего слоя
	if err := m.LogitsFromDevice(format.GgmlQ8_0, "n", nil, "h", nil, nil, nil, 8, 4, 1e-6); err != nil {
		t.Fatal(err)
	}

	if d0.logits != 0 || d1.logits != 1 {
		t.Fatalf("logits: GPU0=%d GPU1=%d, ожидали 0/1", d0.logits, d1.logits)
	}

	// Новый токен возвращает hidden на устройство первого слоя и гасит residency GPU1
	if err := m.HiddenUpload(x); err != nil {
		t.Fatal(err)
	}

	if d0.uploads != 2 || d1.resident {
		t.Fatalf("после нового токена: GPU0 uploads=%d, GPU1 resident=%v", d0.uploads, d1.resident)
	}
}

// -tensor-split задаёт неравные диапазоны слоёв
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
		t.Fatalf("KV по устройствам = %d/%d, ожидали 21/7", d0.kvLayers, d1.kvLayers)
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
		t.Fatalf("границы split: GPU0=%v GPU1=%v", d0.qkvLayers, d1.qkvLayers)
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
		t.Fatalf("VRAMInfo = %d/%d, ожидали 300/12288", used, total)
	}
}

// Residency доступна, только если её держат все устройства (иначе hidden не перенести)
func TestMultiBackendHiddenResidentRequiresAll(t *testing.T) {
	d0, d1 := newFakeDevice("CUDA:0"), newFakeDevice("CUDA:1")
	m, err := NewMultiBackend([]Backend{d0, d1}, nil)
	if err != nil {
		t.Fatal(err)
	}

	if !m.HiddenResident() {
		t.Fatal("HiddenResident() = false при двух resident-устройствах")
	}

	d1.noResid = true
	if m.HiddenResident() {
		t.Fatal("HiddenResident() = true, хотя GPU1 не держит hidden")
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
	if plan != "CUDA:0: слои 0-1, CUDA:1: слои 2-3" {
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
