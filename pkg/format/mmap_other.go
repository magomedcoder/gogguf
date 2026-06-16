//go:build !unix

package format

import (
	"os"
)

// OpenFileMapped открывает GGUF и загружает содержимое в память (fallback без mmap)
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

// Close освобождает ссылку на загруженные данные
func (m *MappedReader) Close() error {
	m.data = nil

	if m.file != nil {
		err := m.file.Close()
		m.file = nil
		return err
	}

	return nil
}
