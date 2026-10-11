package quant

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/magomedcoder/gogguf/pkg/dtype"
)

// ToFloat32 dequantizes raw GGML tensor bytes to float32
func ToFloat32(typ dtype.DType, data []byte, n int) ([]float32, error) {
	switch typ {
	case dtype.F32:
		return dequantF32(data, n)
	case dtype.F16:
		return dequantF16(data, n)
	case dtype.Q8_0:
		return DequantQ8_0(data, n)
	case dtype.Q4_0:
		return DequantQ4_0(data, n)
	case dtype.Q4_1:
		return DequantQ4_1(data, n)
	case dtype.Q5_0:
		return DequantQ5_0(data, n)
	case dtype.Q5_1:
		return DequantQ5_1(data, n)
	case dtype.Q2_K:
		return DequantQ2_K(data, n)
	case dtype.Q3_K:
		return DequantQ3_K(data, n)
	case dtype.Q4_K:
		return DequantQ4_K(data, n)
	case dtype.Q5_K:
		return DequantQ5_K(data, n)
	case dtype.Q6_K:
		return DequantQ6_K(data, n)
	case dtype.Q8_K:
		return DequantQ8_K(data, n)
	case dtype.Int32:
		return dequantI32(data, n)
	default:
		return nil, fmt.Errorf("quant: type %s not yet supported", typ)
	}
}

// dequantF32 reads n float32 from buffer
func dequantF32(data []byte, n int) ([]float32, error) {
	want := n * 4
	if len(data) < want {
		return nil, fmt.Errorf("quant: insufficient F32 data: need %d, have %d", want, len(data))
	}

	out := make([]float32, n)
	for i := range n {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
	}

	return out, nil
}

// dequantF16 converts n fp16 to float32
func dequantF16(data []byte, n int) ([]float32, error) {
	want := n * 2
	if len(data) < want {
		return nil, fmt.Errorf("quant: insufficient F16 data: need %d, have %d", want, len(data))
	}

	out := make([]float32, n)
	for i := range n {
		out[i] = FP16ToFP32(binary.LittleEndian.Uint16(data[i*2:]))
	}

	return out, nil
}

// dequantI32 reads n int32 and converts to float32
func dequantI32(data []byte, n int) ([]float32, error) {
	want := n * 4
	if len(data) < want {
		return nil, fmt.Errorf("quant: insufficient I32 data: need %d, have %d", want, len(data))
	}

	out := make([]float32, n)
	for i := range n {
		out[i] = float32(int32(binary.LittleEndian.Uint32(data[i*4:])))
	}

	return out, nil
}
