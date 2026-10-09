package tensor

import (
	"fmt"
	"io"

	"github.com/magomedcoder/gogguf/pkg/format"
	"github.com/magomedcoder/gogguf/pkg/quant"
)

// Tensor is a dequantized float32 tensor.
type Tensor struct {
	Name  string
	Shape []int
	Type  format.GGML
	Data  []float32
	Raw   []byte
}

// LoadRawView returns a zero-copy slice of the tensor's raw bytes when the source supports mmap/ReaderAt.
func LoadRawView(info *format.TensorInfo) ([]byte, error) {
	if view, ok := info.RawView(); ok {
		return view, nil
	}

	return LoadRaw(info)
}

// LoadRaw reads the tensor's raw bytes from the GGUF file.
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

// LoadFloats loads and dequantizes a tensor to float32.
func LoadFloats(info *format.TensorInfo) ([]float32, error) {
	raw, err := LoadRaw(info)
	if err != nil {
		return nil, err
	}

	return quant.ToFloat32(info.Type, raw, int(info.ValuesCount()))
}

// FromGGUF loads a tensor from GGUF.
func FromGGUF(info *format.TensorInfo) (*Tensor, error) {
	raw, err := LoadRaw(info)
	if err != nil {
		return nil, err
	}

	n := int(info.ValuesCount())
	data, err := quant.ToFloat32(info.Type, raw, n)
	if err != nil {
		return nil, err
	}

	shape := make([]int, len(info.Dimensions))
	for i, d := range info.Dimensions {
		shape[i] = int(d)
	}

	return &Tensor{
		Name:  info.Name,
		Shape: shape,
		Type:  info.Type,
		Data:  data,
		Raw:   raw,
	}, nil
}
