package quant

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/magomedcoder/gogguf/pkg/dtype"
)

// BF16ToFP32 converts one bfloat16 (high 16 bits of IEEE754 f32) to float32.
func BF16ToFP32(h uint16) float32 {
	return math.Float32frombits(uint32(h) << 16)
}

// DequantBF16 converts n bf16 values to float32.
func DequantBF16(data []byte, n int) ([]float32, error) {
	want := n * 2
	if len(data) < want {
		return nil, fmt.Errorf("quant: insufficient BF16 data: need %d, have %d", want, len(data))
	}

	out := make([]float32, n)
	for i := range n {
		out[i] = BF16ToFP32(binary.LittleEndian.Uint16(data[i*2:]))
	}

	return out, nil
}

// DequantIQStub rejects IQ-* / MXFP4 full dequant (layouts registered for Load; fused CUDA not shipped).
// Callers that need Floats should requantize to a fused type or use a host path once real kernels exist.
func DequantIQStub(t dtype.DType, data []byte, n int) ([]float32, error) {
	_ = data
	_ = n
	return nil, fmt.Errorf("quant: %s CPU dequant not implemented (dtype registered for Load; use Q8_1/Q*_K fused or FP16/BF16)", t)
}
