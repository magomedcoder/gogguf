package gpuresid

import (
	"fmt"

	"github.com/magomedcoder/gogguf/pkg/format"
	"github.com/magomedcoder/gogguf/pkg/quant"
)

// QuantBytes returns raw byte size of a rows*cols matrix for a quantized type.
// Used to slice MoE expert tensor without dequantization (§6)
func QuantBytes(t format.GGML, rows, cols int) (int, error) {
	if rows <= 0 || cols <= 0 {
		return 0, fmt.Errorf("rows=%d cols=%d", rows, cols)
	}

	var blk, size int
	switch t {
	case format.GgmlQ8_0:
		blk, size = quant.QK8_0, quant.BlockQ8_0Size
	case format.GgmlQ4_0:
		blk, size = quant.QK4_0, quant.BlockQ4_0Size
	case format.GgmlQ4_K:
		blk, size = quant.QK_K, quant.BlockQ4_KSize
	case format.GgmlQ5_K:
		blk, size = quant.QK_K, quant.BlockQ5_KSize
	case format.GgmlQ6_K:
		blk, size = quant.QK_K, quant.BlockQ6_KSize
	default:
		return 0, fmt.Errorf("тип %s не поддерживается", t)
	}

	if cols%blk != 0 {
		return 0, fmt.Errorf("cols=%d не кратно %d", cols, blk)
	}

	return rows * (cols / blk) * size, nil
}
