package format

import (
	"fmt"
	"io"
	"os"
)

// MappedReader - GGUF reader с memory-mapped файлом
type MappedReader struct {
	*Reader
	file *os.File
	data []byte
}

// Data возвращает mmap-срез всего файла
func (m *MappedReader) Data() []byte {
	return m.data
}

type mmapSource struct {
	data []byte
	pos  int64
}

func (m *mmapSource) Read(p []byte) (int, error) {
	if m.pos >= int64(len(m.data)) {
		return 0, io.EOF
	}

	n := copy(p, m.data[m.pos:])
	m.pos += int64(n)

	return n, nil
}

func (m *mmapSource) Seek(offset int64, whence int) (int64, error) {
	var abs int64
	switch whence {
	case 0:
		abs = offset
	case 1:
		abs = m.pos + offset
	case 2:
		abs = int64(len(m.data)) + offset
	default:
		return 0, fmt.Errorf("invalid whence")
	}

	if abs < 0 || abs > int64(len(m.data)) {
		return 0, fmt.Errorf("seek out of range")
	}

	m.pos = abs

	return abs, nil
}

func (m *mmapSource) ReadAt(p []byte, off int64) (int, error) {
	if off < 0 || off > int64(len(m.data)) {
		return 0, fmt.Errorf("readat out of range")
	}

	n := copy(p, m.data[off:])
	if n < len(p) {
		return n, io.EOF
	}

	return n, nil
}

func (m *mmapSource) Slice(off, size int64) []byte {
	if off < 0 || off+size > int64(len(m.data)) {
		return nil
	}

	return m.data[off : off+size]
}
