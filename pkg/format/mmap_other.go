//go:build !unix

package format

import (
	"os"
)

// OpenFileMapped opens GGUF and loads content into memory (fallback without mmap)
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
