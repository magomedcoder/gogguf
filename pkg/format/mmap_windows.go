//go:build windows

package format

import (
	"fmt"
	"os"
	"syscall"
	"unsafe"
)

// OpenFileMapped maps the GGUF via CreateFileMapping / MapViewOfFile (zero-copy).
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

	size := info.Size()
	if size < 0 {
		f.Close()
		return nil, fmt.Errorf("gguf: mmap: negative file size")
	}

	if size == 0 {
		src := &mmapSource{data: []byte{}}
		r, err := Open(src)
		if err != nil {
			f.Close()
			return nil, err
		}

		return &MappedReader{
			Reader: r,
			file:   f,
			data:   []byte{},
		}, nil
	}

	h, err := syscall.CreateFileMapping(syscall.Handle(f.Fd()), nil, syscall.PAGE_READONLY, 0, 0, nil)
	if err != nil {
		f.Close()
		return nil, fmt.Errorf("gguf: CreateFileMapping: %w", err)
	}

	addr, err := syscall.MapViewOfFile(h, syscall.FILE_MAP_READ, 0, 0, 0)
	_ = syscall.CloseHandle(h)
	if err != nil {
		f.Close()
		return nil, fmt.Errorf("gguf: MapViewOfFile: %w", err)
	}

	data := unsafe.Slice((*byte)(unsafe.Pointer(addr)), int(size))
	src := &mmapSource{
		data: data,
	}
	r, err := Open(src)
	if err != nil {
		_ = syscall.UnmapViewOfFile(addr)
		f.Close()
		return nil, err
	}

	return &MappedReader{
		Reader:  r,
		file:    f,
		data:    data,
		winAddr: addr,
	}, nil
}

// Close unmaps the view and closes the file.
func (m *MappedReader) Close() error {
	if m.winAddr != 0 {
		_ = syscall.UnmapViewOfFile(m.winAddr)
		m.winAddr = 0
	}
	m.data = nil

	if m.file != nil {
		err := m.file.Close()
		m.file = nil
		return err
	}

	return nil
}
