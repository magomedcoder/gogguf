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

// MatMulVecQ2_K умножает Q2_K-матрицу на float32-вектор
func MatMulVecQ2_K(raw []byte, rows, cols int, vec []float32) ([]float32, error) {
	out := make([]float32, rows)
	if err := MatMulVecQ2_KInto(raw, rows, cols, vec, out); err != nil {
		return nil, err
	}

	return out, nil
}

// MatMulVecQ2_KInto записывает Q2_K matmul в out
func MatMulVecQ2_KInto(raw []byte, rows, cols int, vec, out []float32) error {
	return matMulVecKQuant(raw, rows, cols, vec, out, quant.BlockQ2_KSize, quant.QK_K, quant.DotBlockQ2_K, "Q2_K")
}

// MatMulVecQ3_K умножает Q3_K-матрицу на float32-вектор
func MatMulVecQ3_K(raw []byte, rows, cols int, vec []float32) ([]float32, error) {
	out := make([]float32, rows)
	if err := MatMulVecQ3_KInto(raw, rows, cols, vec, out); err != nil {
		return nil, err
	}

	return out, nil
}

// MatMulVecQ3_KInto записывает Q3_K matmul в out
func MatMulVecQ3_KInto(raw []byte, rows, cols int, vec, out []float32) error {
	return matMulVecKQuant(raw, rows, cols, vec, out, quant.BlockQ3_KSize, quant.QK_K, quant.DotBlockQ3_K, "Q3_K")
}

// MatMulVecQ8_K умножает Q8_K-матрицу на float32-вектор
func MatMulVecQ8_K(raw []byte, rows, cols int, vec []float32) ([]float32, error) {
	out := make([]float32, rows)
	if err := MatMulVecQ8_KInto(raw, rows, cols, vec, out); err != nil {
		return nil, err
	}

	return out, nil
}

// MatMulVecQ8_KInto записывает Q8_K matmul в out
func MatMulVecQ8_KInto(raw []byte, rows, cols int, vec, out []float32) error {
	return matMulVecKQuant(raw, rows, cols, vec, out, quant.BlockQ8_KSize, quant.QK_K, quant.DotBlockQ8_K, "Q8_K")
}

// EmbeddingQ2_KInto деквантизирует embedding-строку в dst
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

// EmbeddingQ3_KInto деквантизирует embedding-строку в dst
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

// EmbeddingQ8_KInto деквантизирует embedding-строку в dst
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
