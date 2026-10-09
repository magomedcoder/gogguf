package format

import "io"

// TensorInfo represents tensor in GGUF file
type TensorInfo struct {
	reader *Reader

	Name       string
	Dimensions []uint64
	Type       GGML
	Offset     uint64
}

// RawView returns zero-copy tensor data slice when source is mmap or []byte
func (t *TensorInfo) RawView() ([]byte, bool) {
	start := t.reader.tensorOffset + int64(t.Offset)
	size := t.Size()

	if dr, ok := t.reader.r.(dataReader); ok {
		return dr.Slice(start, size), true
	}

	return nil, false
}

type dataReader interface {
	Slice(off, size int64) []byte
}

// Reader returns io.Reader for reading tensor data
// Reader is limited to tensor data size and does not change source file position
func (t *TensorInfo) Reader() (io.Reader, error) {
	start := t.reader.tensorOffset + int64(t.Offset)
	size := t.Size()

	ra, ok := t.reader.r.(io.ReaderAt)
	if !ok {
		return nil, errReaderAtRequired
	}

	return io.NewSectionReader(ra, start, size), nil
}

// Size returns tensor data size in bytes
func (t *TensorInfo) Size() int64 {
	return t.Type.dataSize(t.Dimensions)
}

// ValuesCount returns number of tensor elements
func (t *TensorInfo) ValuesCount() int64 {
	n := uint64(1)
	for _, d := range t.Dimensions {
		n *= d
	}

	return int64(n)
}
