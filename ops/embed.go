package ops

import (
	"fmt"

	"github.com/magomedcoder/gguf.go/quant"
)

// EmbeddingQ8_0 извлекает строку embedding из Q8_0-матрицы [vocab×dim]
func EmbeddingQ8_0(raw []byte, dim, tokenID int) ([]float32, error) {
	if dim%quant.QK8_0 != 0 {
		return nil, fmt.Errorf("ops: dim=%d не кратно %d", dim, quant.QK8_0)
	}

	blocksPerRow := dim / quant.QK8_0
	rowBytes := blocksPerRow * quant.BlockQ8_0Size
	off := tokenID * rowBytes

	if off+rowBytes > len(raw) {
		return nil, fmt.Errorf("ops: tokenID=%d вне диапазона", tokenID)
	}

	return quant.DequantQ8_0(raw[off:off+rowBytes], dim)
}
