package gpuresid

import (
	"fmt"
	"github.com/magomedcoder/gogguf/pkg/dtype"

	"github.com/magomedcoder/gogguf/pkg/quant"
)

// QuantBytes returns raw byte size of a rows*cols matrix for a quantized type.
// Used to slice MoE expert tensor without dequantization
func QuantBytes(t dtype.DType, rows, cols int) (int, error) {
	if rows <= 0 || cols <= 0 {
		return 0, fmt.Errorf("rows=%d cols=%d", rows, cols)
	}

	var blk, size int
	switch t {
	case dtype.Q8_0:
		blk, size = quant.QK8_0, quant.BlockQ8_0Size
	case dtype.Q4_0:
		blk, size = quant.QK4_0, quant.BlockQ4_0Size
	case dtype.Q4_K:
		blk, size = quant.QK_K, quant.BlockQ4_KSize
	case dtype.Q5_K:
		blk, size = quant.QK_K, quant.BlockQ5_KSize
	case dtype.Q6_K:
		blk, size = quant.QK_K, quant.BlockQ6_KSize
	case dtype.Q4_1:
		blk, size = quant.QK4_1, quant.BlockQ4_1Size
	case dtype.Q5_0:
		blk, size = quant.QK5_0, quant.BlockQ5_0Size
	case dtype.Q5_1:
		blk, size = quant.QK5_1, quant.BlockQ5_1Size
	case dtype.Q2_K:
		blk, size = quant.QK_K, quant.BlockQ2_KSize
	case dtype.Q3_K:
		blk, size = quant.QK_K, quant.BlockQ3_KSize
	case dtype.Q8_K:
		blk, size = quant.QK_K, quant.BlockQ8_KSize
	default:
		return 0, fmt.Errorf("type %s not supported", t)
	}

	if cols%blk != 0 {
		return 0, fmt.Errorf("cols=%d not divisible by %d", cols, blk)
	}

	return rows * (cols / blk) * size, nil
}
