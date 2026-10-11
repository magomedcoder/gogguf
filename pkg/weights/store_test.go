package weights

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"

	"github.com/magomedcoder/gogguf/pkg/dtype"
	"github.com/magomedcoder/gogguf/pkg/format"
)

func TestStoreRawAndFloats(t *testing.T) {
	vals := []float32{1.5, -2, 3.25, 0}
	raw := buildTinyGGUF("norm.weight", []uint64{4}, vals)
	r, err := format.Open(bytes.NewReader(raw))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}

	s := New(r)
	if s.Reader() != r {
		t.Fatal("Reader mismatch")
	}

	info, err := s.Info("norm.weight")
	if err != nil {
		t.Fatalf("Info: %v", err)
	}

	if info.Type != dtype.F32 {
		t.Fatalf("Type=%v", info.Type)
	}

	b1, err := s.Raw("norm.weight")
	if err != nil {
		t.Fatalf("Raw: %v", err)
	}

	b2, err := s.Raw("norm.weight")
	if err != nil {
		t.Fatalf("Raw cache: %v", err)
	}

	if &b1[0] != &b2[0] {
		t.Fatal("Raw must return cached slice")
	}

	f1, err := s.Floats("norm.weight")
	if err != nil {
		t.Fatalf("Floats: %v", err)
	}

	f2, err := s.Floats("norm.weight")
	if err != nil {
		t.Fatalf("Floats cache: %v", err)
	}

	if &f1[0] != &f2[0] {
		t.Fatal("Floats must return cached slice")
	}

	for i, want := range vals {
		if f1[i] != want {
			t.Fatalf("floats[%d]=%v want %v", i, f1[i], want)
		}
	}

	if _, err := s.Raw("missing"); err == nil {
		t.Fatal("expected missing tensor error")
	}
}

func TestLoadRawViewMmap(t *testing.T) {
	vals := []float32{4, 3, 2, 1}
	raw := buildTinyGGUF("w", []uint64{4}, vals)
	path := filepath.Join(t.TempDir(), "tiny.gguf")
	if err := os.WriteFile(path, raw, 0o644); err != nil {
		t.Fatal(err)
	}

	mr, err := format.OpenFileMapped(path)
	if err != nil {
		t.Fatalf("OpenFileMapped: %v", err)
	}
	defer mr.Close()

	s := New(mr.Reader)
	b, err := s.Raw("w")
	if err != nil {
		t.Fatalf("Raw: %v", err)
	}

	info, _ := s.Info("w")
	view, ok := info.RawView()
	if !ok {
		t.Fatal("expected RawView")
	}

	if &b[0] != &view[0] {
		t.Fatal("Store.Raw should keep mmap view")
	}

	copied, err := LoadRaw(info)
	if err != nil {
		t.Fatalf("LoadRaw: %v", err)
	}

	if &copied[0] == &view[0] {
		t.Fatal("LoadRaw must copy")
	}

	if len(copied) != len(view) {
		t.Fatalf("len copy=%d view=%d", len(copied), len(view))
	}
}

// buildTinyGGUF mirrors format test helper (minimal GGUF v3, one F32 tensor)
func buildTinyGGUF(name string, dims []uint64, values []float32) []byte {
	const (
		magic     = "GGUF"
		alignment = 32
	)
	var hdr bytes.Buffer
	bo := binary.LittleEndian

	hdr.WriteString(magic)
	_ = binary.Write(&hdr, bo, uint32(3))
	_ = binary.Write(&hdr, bo, uint64(1))
	_ = binary.Write(&hdr, bo, uint64(1))

	writeStr := func(s string) {
		_ = binary.Write(&hdr, bo, uint64(len(s)))
		hdr.WriteString(s)
	}
	writeStr("general.architecture")
	_ = binary.Write(&hdr, bo, format.String)
	writeStr("test")

	writeStr(name)
	_ = binary.Write(&hdr, bo, uint32(len(dims)))
	for _, d := range dims {
		_ = binary.Write(&hdr, bo, d)
	}

	_ = binary.Write(&hdr, bo, uint32(dtype.F32))
	_ = binary.Write(&hdr, bo, uint64(0))

	pad := (alignment - hdr.Len()%alignment) % alignment
	hdr.Write(make([]byte, pad))

	var data bytes.Buffer
	for _, v := range values {
		_ = binary.Write(&data, bo, v)
	}

	return append(hdr.Bytes(), data.Bytes()...)
}
