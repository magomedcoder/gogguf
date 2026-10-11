//go:build !unix && !windows

package format

import (
	"os"
)

// OpenFileMapped loads the entire GGUF into RAM (no OS mmap on this platform).
// RSS ≈ file size; not zero-copy. Windows and Unix use real mmap.
func OpenFileMapped(path string) (*MappedReader, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	src := &mmapSource{data: data}
	r, err := Open(src)
	if err != nil {
		return nil, err
	}

	return &MappedReader{
		Reader: r,
		data:   data,
	}, nil
}

// Close releases reference to loaded data
func (m *MappedReader) Close() error {
	m.data = nil

	if m.file != nil {
		err := m.file.Close()
		m.file = nil
		return err
	}

	return nil
}
