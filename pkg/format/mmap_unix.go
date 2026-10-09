//go:build unix

package format

import (
	"fmt"
	"os"
	"syscall"
)

// OpenFileMapped opens GGUF via mmap for zero-copy weight access
func OpenFileMapped(path string) (*MappedReader, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}

	info, err := f.Stat()
	if err != nil {
		f.Close()
		return nil, err
	}

	data, err := syscall.Mmap(int(f.Fd()), 0, int(info.Size()), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		f.Close()
		return nil, fmt.Errorf("gguf: mmap: %w", err)
	}

	src := &mmapSource{data: data}
	r, err := Open(src)
	if err != nil {
		syscall.Munmap(data)
		f.Close()
		return nil, err
	}

	return &MappedReader{
		Reader: r,
		file:   f,
		data:   data,
	}, nil
}

// Close unmaps and closes file
func (m *MappedReader) Close() error {
	if m.data != nil {
		_ = syscall.Munmap(m.data)
		m.data = nil
	}

	if m.file != nil {
		err := m.file.Close()
		m.file = nil
		return err
	}

	return nil
}
