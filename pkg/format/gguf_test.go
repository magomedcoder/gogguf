package format

import (
	"bytes"
	"encoding/binary"
	"github.com/magomedcoder/gogguf/pkg/dtype"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// buildTinyGGUF builds a minimal little-endian GGUF v3 with one F32 tensor.
func buildTinyGGUF(name string, dims []uint64, values []float32) []byte {
	var hdr bytes.Buffer
	bo := binary.LittleEndian

	hdr.WriteString(magic)
	_ = binary.Write(&hdr, bo, uint32(3))
	_ = binary.Write(&hdr, bo, uint64(1)) // tensor_count
	_ = binary.Write(&hdr, bo, uint64(3)) // metadata_count

	writeGGUFString(&hdr, bo, "general.architecture")
	_ = binary.Write(&hdr, bo, String)
	writeGGUFString(&hdr, bo, "test")

	writeGGUFString(&hdr, bo, "general.name")
	_ = binary.Write(&hdr, bo, String)
	writeGGUFString(&hdr, bo, "tiny")

	writeGGUFString(&hdr, bo, "general.file_type")
	_ = binary.Write(&hdr, bo, Uint32)
	_ = binary.Write(&hdr, bo, uint32(AllF32))

	writeGGUFString(&hdr, bo, name)
	_ = binary.Write(&hdr, bo, uint32(len(dims)))
	for _, d := range dims {
		_ = binary.Write(&hdr, bo, d)
	}

	_ = binary.Write(&hdr, bo, uint32(dtype.F32))
	_ = binary.Write(&hdr, bo, uint64(0)) // offset within tensor block

	pad := (defaultAlignment - int64(hdr.Len())%defaultAlignment) % defaultAlignment
	hdr.Write(make([]byte, pad))

	var data bytes.Buffer
	for _, v := range values {
		_ = binary.Write(&data, bo, v)
	}

	return append(hdr.Bytes(), data.Bytes()...)
}

func writeGGUFString(buf *bytes.Buffer, bo binary.ByteOrder, s string) {
	_ = binary.Write(buf, bo, uint64(len(s)))
	buf.WriteString(s)
}

func TestOpenTinyGGUF(t *testing.T) {
	vals := []float32{1, 2, 3, 4}
	raw := buildTinyGGUF("weight", []uint64{2, 2}, vals)
	r, err := Open(bytes.NewReader(raw))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}

	if r.Version != 3 {
		t.Fatalf("Version=%d", r.Version)
	}

	if r.ByteOrder != binary.LittleEndian {
		t.Fatal("expected little endian")
	}

	arch, err := r.Metadata.String("general.architecture")
	if err != nil || arch != "test" {
		t.Fatalf("architecture=%q err=%v", arch, err)
	}

	if ft, err := MetaValue[Filetype](r.Metadata, "general.file_type"); err != nil || ft != AllF32 {
		t.Fatalf("file_type=%v err=%v", ft, err)
	}

	info, err := r.TensorInfo("weight")
	if err != nil {
		t.Fatalf("TensorInfo: %v", err)
	}

	if info.Type != dtype.F32 {
		t.Fatalf("Type=%v", info.Type)
	}

	if info.ValuesCount() != 4 {
		t.Fatalf("ValuesCount=%d", info.ValuesCount())
	}

	if info.Size() != 16 {
		t.Fatalf("Size=%d", info.Size())
	}

	if r.TensorSize() != 16 {
		t.Fatalf("TensorSize=%d", r.TensorSize())
	}

	sec, err := info.Reader()
	if err != nil {
		t.Fatalf("Reader: %v", err)
	}

	got := make([]byte, info.Size())
	if _, err := sec.Read(got); err != nil {
		t.Fatalf("read tensor: %v", err)
	}

	for i, want := range vals {
		have := math.Float32frombits(binary.LittleEndian.Uint32(got[i*4:]))
		if have != want {
			t.Fatalf("value[%d]=%v want %v", i, have, want)
		}
	}
}

func TestOpenBadMagic(t *testing.T) {
	_, err := Open(bytes.NewReader([]byte("XXXX\x03\x00\x00\x00")))
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestOpenBadVersion(t *testing.T) {
	var buf bytes.Buffer
	buf.WriteString(magic)
	_ = binary.Write(&buf, binary.LittleEndian, uint32(1))
	_ = binary.Write(&buf, binary.LittleEndian, uint64(0))
	_ = binary.Write(&buf, binary.LittleEndian, uint64(0))
	_, err := Open(bytes.NewReader(buf.Bytes()))
	if err == nil {
		t.Fatal("expected error for version 1")
	}
}

func TestTensorInfoMissing(t *testing.T) {
	raw := buildTinyGGUF("weight", []uint64{4}, []float32{1, 2, 3, 4})
	r, err := Open(bytes.NewReader(raw))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}

	if _, err := r.TensorInfo("nope"); err == nil {
		t.Fatal("expected missing tensor error")
	}
}

func TestMetadataHelpers(t *testing.T) {
	raw := buildTinyGGUF("w", []uint64{1}, []float32{0})
	r, err := Open(bytes.NewReader(raw))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}

	if got := r.Metadata.IntOptional("missing", 7); got != 7 {
		t.Fatalf("IntOptional=%d", got)
	}

	name, err := r.Metadata.String("general.name")
	if err != nil || name != "tiny" {
		t.Fatalf("name=%q err=%v", name, err)
	}

	if _, err := r.Metadata.Int("general.name"); err == nil {
		t.Fatal("expected type error for Int(string)")
	}
}

func TestTypeAndDTypeString(t *testing.T) {
	if String.String() != "string" {
		t.Fatalf("Type.String=%q", String.String())
	}

	if Type(99).String() != "unknown-type-99" {
		t.Fatalf("unknown Type.String=%q", Type(99).String())
	}

	if dtype.Q8_0.String() != "q8_0" {
		t.Fatalf("DType.String=%q", dtype.Q8_0.String())
	}

	if AllF32.String() != "all F32" {
		t.Fatalf("Filetype.String=%q", AllF32.String())
	}
}

func TestOpenFileMappedRawView(t *testing.T) {
	vals := []float32{9, 8, 7, 6}
	raw := buildTinyGGUF("m", []uint64{4}, vals)
	path := filepath.Join(t.TempDir(), "tiny.gguf")
	if err := os.WriteFile(path, raw, 0o644); err != nil {
		t.Fatal(err)
	}

	mr, err := OpenFileMapped(path)
	if err != nil {
		t.Fatalf("OpenFileMapped: %v", err)
	}
	defer mr.Close()

	info, err := mr.TensorInfo("m")
	if err != nil {
		t.Fatalf("TensorInfo: %v", err)
	}

	view, ok := info.RawView()
	if !ok {
		t.Fatal("RawView expected for mmap")
	}

	if len(view) != 16 {
		t.Fatalf("view len=%d", len(view))
	}

	if math.Float32frombits(binary.LittleEndian.Uint32(view)) != 9 {
		t.Fatalf("first=%v", math.Float32frombits(binary.LittleEndian.Uint32(view)))
	}
}
