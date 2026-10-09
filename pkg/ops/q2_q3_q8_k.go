package ops

import (
	"fmt"

	"github.com/magomedcoder/gogguf/pkg/quant"
)

func matMulVecKQuant(
	raw []byte, rows, cols int, vec, out []float32,
	blockSize, qk int,
	dot func([]byte, []float32) (float32, error),
	name string,
) error {
	if len(vec) != cols {
		return fmt.Errorf("ops: len(vec)=%d, cols=%d", len(vec), cols)
	}

	if cols%qk != 0 {
		return fmt.Errorf("ops: cols=%d не кратно %d", cols, qk)
	}

	blocksPerRow := cols / qk
	want := rows * blocksPerRow * blockSize
	if len(raw) < want {
		return fmt.Errorf("ops: %s matrix слишком короткая", name)
	}

	if len(out) < rows {
		return fmt.Errorf("ops: out слишком короткий")
	}

	parallelForRows(rows, func(rowStart, rowEnd int) {
		for r := rowStart; r < rowEnd; r++ {
			var sum float32
			rowOff := r * blocksPerRow * blockSize
			for b := range blocksPerRow {
				block := raw[rowOff+b*blockSize:]
				vecOff := b * qk
				d, _ := dot(block, vec[vecOff:vecOff+qk])
				sum += d
			}
			out[r] = sum
		}
	})

	return nil
}

// MatMulVecQ2_K multiplies Q2_K matrix by float32 vector
func MatMulVecQ2_K(raw []byte, rows, cols int, vec []float32) ([]float32, error) {
	out := make([]float32, rows)
	if err := MatMulVecQ2_KInto(raw, rows, cols, vec, out); err != nil {
		return nil, err
	}

	return out, nil
}

// MatMulVecQ2_KInto writes Q2_K matmul to out
func MatMulVecQ2_KInto(raw []byte, rows, cols int, vec, out []float32) error {
	return matMulVecKQuant(raw, rows, cols, vec, out, quant.BlockQ2_KSize, quant.QK_K, quant.DotBlockQ2_K, "Q2_K")
}

// MatMulVecQ3_K multiplies Q3_K matrix by float32 vector
func MatMulVecQ3_K(raw []byte, rows, cols int, vec []float32) ([]float32, error) {
	out := make([]float32, rows)
	if err := MatMulVecQ3_KInto(raw, rows, cols, vec, out); err != nil {
		return nil, err
	}

	return out, nil
}

// MatMulVecQ3_KInto writes Q3_K matmul to out
func MatMulVecQ3_KInto(raw []byte, rows, cols int, vec, out []float32) error {
	return matMulVecKQuant(raw, rows, cols, vec, out, quant.BlockQ3_KSize, quant.QK_K, quant.DotBlockQ3_K, "Q3_K")
}

// MatMulVecQ8_K multiplies Q8_K matrix by float32 vector
func MatMulVecQ8_K(raw []byte, rows, cols int, vec []float32) ([]float32, error) {
	out := make([]float32, rows)
	if err := MatMulVecQ8_KInto(raw, rows, cols, vec, out); err != nil {
		return nil, err
	}

	return out, nil
}

// MatMulVecQ8_KInto writes Q8_K matmul to out
func MatMulVecQ8_KInto(raw []byte, rows, cols int, vec, out []float32) error {
	return matMulVecKQuant(raw, rows, cols, vec, out, quant.BlockQ8_KSize, quant.QK_K, quant.DotBlockQ8_K, "Q8_K")
}

// EmbeddingQ2_KInto dequantizes an embedding row into dst
func EmbeddingQ2_KInto(dst []float32, raw []byte, dim, tokenID int) error {
	off, rowBytes, err := embeddingRowOffset(dim, tokenID, quant.BlockQ2_KSize, quant.QK_K)
	if err != nil {
		return err
	}

	if off+rowBytes > len(raw) {
		return fmt.Errorf("ops: tokenID=%d вне диапазона", tokenID)
	}

	return quant.DequantQ2_KInto(dst, raw[off:off+rowBytes], dim)
}

// EmbeddingQ3_KInto dequantizes an embedding row into dst
func EmbeddingQ3_KInto(dst []float32, raw []byte, dim, tokenID int) error {
	off, rowBytes, err := embeddingRowOffset(dim, tokenID, quant.BlockQ3_KSize, quant.QK_K)
	if err != nil {
		return err
	}

	if off+rowBytes > len(raw) {
		return fmt.Errorf("ops: tokenID=%d вне диапазона", tokenID)
	}

	return quant.DequantQ3_KInto(dst, raw[off:off+rowBytes], dim)
}

// EmbeddingQ8_KInto dequantizes an embedding row into dst
func EmbeddingQ8_KInto(dst []float32, raw []byte, dim, tokenID int) error {
	off, rowBytes, err := embeddingRowOffset(dim, tokenID, quant.BlockQ8_KSize, quant.QK_K)
	if err != nil {
		return err
	}

	if off+rowBytes > len(raw) {
		return fmt.Errorf("ops: tokenID=%d вне диапазона", tokenID)
	}

	return quant.DequantQ8_KInto(dst, raw[off:off+rowBytes], dim)
}
