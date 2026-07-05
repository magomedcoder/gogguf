package quant

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/magomedcoder/gogguf/pkg/format"
)

// ToFloat32 деквантизирует сырые байты GGML-тензора в float32
func ToFloat32(typ format.GGML, data []byte, n int) ([]float32, error) {
	switch typ {
	case format.GgmlFloat32:
		return dequantF32(data, n)
	case format.GgmlFloat16:
		return dequantF16(data, n)
	case format.GgmlQ8_0:
		return DequantQ8_0(data, n)
	case format.GgmlQ4_0:
		return DequantQ4_0(data, n)
	case format.GgmlQ4_K:
		return DequantQ4_K(data, n)
	case format.GgmlInt32:
		return dequantI32(data, n)
	default:
		return nil, fmt.Errorf("quant: тип %s пока не поддерживается", typ)
	}
}

// dequantF32 читает n float32 из буфера
func dequantF32(data []byte, n int) ([]float32, error) {
	want := n * 4
	if len(data) < want {
		return nil, fmt.Errorf("quant: F32 данных недостаточно: нужно %d, есть %d", want, len(data))
	}

	out := make([]float32, n)
	for i := range n {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
	}

	return out, nil
}

// dequantF16 конвертирует n fp16 в float32
func dequantF16(data []byte, n int) ([]float32, error) {
	want := n * 2
	if len(data) < want {
		return nil, fmt.Errorf("quant: F16 данных недостаточно: нужно %d, есть %d", want, len(data))
	}

	out := make([]float32, n)
	for i := range n {
		out[i] = FP16ToFP32(binary.LittleEndian.Uint16(data[i*2:]))
	}

	return out, nil
}

// dequantI32 читает n int32 и приводит к float32
func dequantI32(data []byte, n int) ([]float32, error) {
	want := n * 4
	if len(data) < want {
		return nil, fmt.Errorf("quant: I32 данных недостаточно: нужно %d, есть %d", want, len(data))
	}

	out := make([]float32, n)
	for i := range n {
		out[i] = float32(int32(binary.LittleEndian.Uint32(data[i*4:])))
	}

	return out, nil
}
