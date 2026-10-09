package ops

import (
	"fmt"

	"github.com/magomedcoder/gogguf/pkg/quant"
)

func embeddingRowOffset(dim, tokenID, blockSize, qk int) (int, int, error) {
	if dim%qk != 0 {
		return 0, 0, fmt.Errorf("ops: dim=%d не кратно %d", dim, qk)
	}

	blocksPerRow := dim / qk
	rowBytes := blocksPerRow * blockSize
	off := tokenID * rowBytes
	return off, rowBytes, nil
}

// EmbeddingQ8_0Into dequantizes an embedding row into dst
func EmbeddingQ8_0Into(dst []float32, raw []byte, dim, tokenID int) error {
	off, rowBytes, err := embeddingRowOffset(dim, tokenID, quant.BlockQ8_0Size, quant.QK8_0)
	if err != nil {
		return err
	}

	if off+rowBytes > len(raw) {
		return fmt.Errorf("ops: tokenID=%d вне диапазона", tokenID)
	}

	return quant.DequantQ8_0Into(dst, raw[off:off+rowBytes], dim)
}

// EmbeddingQ4_0Into dequantizes an embedding row into dst
func EmbeddingQ4_0Into(dst []float32, raw []byte, dim, tokenID int) error {
	off, rowBytes, err := embeddingRowOffset(dim, tokenID, quant.BlockQ4_0Size, quant.QK4_0)
	if err != nil {
		return err
	}

	if off+rowBytes > len(raw) {
		return fmt.Errorf("ops: tokenID=%d вне диапазона", tokenID)
	}

	return quant.DequantQ4_0Into(dst, raw[off:off+rowBytes], dim)
}

// EmbeddingQ4_KInto dequantizes an embedding row into dst
func EmbeddingQ4_KInto(dst []float32, raw []byte, dim, tokenID int) error {
	off, rowBytes, err := embeddingRowOffset(dim, tokenID, quant.BlockQ4_KSize, quant.QK_K)
	if err != nil {
		return err
	}

	if off+rowBytes > len(raw) {
		return fmt.Errorf("ops: tokenID=%d вне диапазона", tokenID)
	}

	return quant.DequantQ4_KInto(dst, raw[off:off+rowBytes], dim)
}

// EmbeddingQ5_KInto dequantizes an embedding row into dst
func EmbeddingQ5_KInto(dst []float32, raw []byte, dim, tokenID int) error {
	off, rowBytes, err := embeddingRowOffset(dim, tokenID, quant.BlockQ5_KSize, quant.QK_K)
	if err != nil {
		return err
	}

	if off+rowBytes > len(raw) {
		return fmt.Errorf("ops: tokenID=%d вне диапазона", tokenID)
	}

	return quant.DequantQ5_KInto(dst, raw[off:off+rowBytes], dim)
}

// EmbeddingQ6_KInto dequantizes an embedding row into dst
func EmbeddingQ6_KInto(dst []float32, raw []byte, dim, tokenID int) error {
	off, rowBytes, err := embeddingRowOffset(dim, tokenID, quant.BlockQ6_KSize, quant.QK_K)
	if err != nil {
		return err
	}

	if off+rowBytes > len(raw) {
		return fmt.Errorf("ops: tokenID=%d вне диапазона", tokenID)
	}

	return quant.DequantQ6_KInto(dst, raw[off:off+rowBytes], dim)
}

// EmbeddingQ8_0 extracts an embedding row from a Q8_0 matrix [vocab*dim]
func EmbeddingQ8_0(raw []byte, dim, tokenID int) ([]float32, error) {
	out := make([]float32, dim)
	if err := EmbeddingQ8_0Into(out, raw, dim, tokenID); err != nil {
		return nil, err
	}

	return out, nil
}

// EmbeddingQ4_0 extracts an embedding row from a Q4_0 matrix [vocab*dim]
func EmbeddingQ4_0(raw []byte, dim, tokenID int) ([]float32, error) {
	out := make([]float32, dim)
	if err := EmbeddingQ4_0Into(out, raw, dim, tokenID); err != nil {
		return nil, err
	}

	return out, nil
}

// EmbeddingQ4_K extracts an embedding row from a Q4_K matrix [vocab*dim]
func EmbeddingQ4_K(raw []byte, dim, tokenID int) ([]float32, error) {
	out := make([]float32, dim)
	if err := EmbeddingQ4_KInto(out, raw, dim, tokenID); err != nil {
		return nil, err
	}

	return out, nil
}
