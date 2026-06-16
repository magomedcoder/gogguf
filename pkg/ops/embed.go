package ops

import (
	"fmt"

	"github.com/magomedcoder/gguf.go/pkg/quant"
)

// EmbeddingQ8_0 извлекает строку embedding из Q8_0-матрицы [vocab*dim]
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

// EmbeddingQ4_0 извлекает строку embedding из Q4_0-матрицы [vocab*dim]
func EmbeddingQ4_0(raw []byte, dim, tokenID int) ([]float32, error) {
	if dim%quant.QK4_0 != 0 {
		return nil, fmt.Errorf("ops: dim=%d не кратно %d", dim, quant.QK4_0)
	}

	blocksPerRow := dim / quant.QK4_0
	rowBytes := blocksPerRow * quant.BlockQ4_0Size
	off := tokenID * rowBytes
	if off+rowBytes > len(raw) {
		return nil, fmt.Errorf("ops: tokenID=%d вне диапазона", tokenID)
	}

	return quant.DequantQ4_0(raw[off:off+rowBytes], dim)
}

// EmbeddingQ4_K извлекает строку embedding из Q4_K-матрицы [vocab*dim]
func EmbeddingQ4_K(raw []byte, dim, tokenID int) ([]float32, error) {
	if dim%quant.QK_K != 0 {
		return nil, fmt.Errorf("ops: dim=%d не кратно %d", dim, quant.QK_K)
	}

	blocksPerRow := dim / quant.QK_K
	rowBytes := blocksPerRow * quant.BlockQ4_KSize
	off := tokenID * rowBytes
	if off+rowBytes > len(raw) {
		return nil, fmt.Errorf("ops: tokenID=%d вне диапазона", tokenID)
	}

	return quant.DequantQ4_K(raw[off:off+rowBytes], dim)
}
