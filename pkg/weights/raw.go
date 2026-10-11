package weights

import (
	"fmt"
	"io"

	"github.com/magomedcoder/gogguf/pkg/format"
)

// LoadRawView returns a zero-copy slice of the tensor's raw bytes when the source supports mmap/Slice; otherwise it copies via LoadRaw
func LoadRawView(info *format.TensorInfo) ([]byte, error) {
	if view, ok := info.RawView(); ok {
		return view, nil
	}

	return LoadRaw(info)
}

// LoadRaw reads the tensor's raw bytes from the GGUF file
func LoadRaw(info *format.TensorInfo) ([]byte, error) {
	r, err := info.Reader()
	if err != nil {
		return nil, err
	}

	raw := make([]byte, info.Size())
	if _, err := io.ReadFull(r, raw); err != nil {
		return nil, fmt.Errorf("tensor %q: %w", info.Name, err)
	}

	return raw, nil
}
