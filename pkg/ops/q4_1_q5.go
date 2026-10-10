package ops

import (
	"fmt"

	"github.com/magomedcoder/gogguf/pkg/quant"
)

// MatMulVecQ4_1 multiplies Q4_1 matrix [rows*cols] by float32 vector [cols]
func MatMulVecQ4_1(raw []byte, rows, cols int, vec []float32) ([]float32, error) {
	out := make([]float32, rows)
	if err := MatMulVecQ4_1Into(raw, rows, cols, vec, out); err != nil {
		return nil, err
	}

	return out, nil
}

// MatMulVecQ4_1Into writes Q4_1 matmul to out [rows]
func MatMulVecQ4_1Into(raw []byte, rows, cols int, vec, out []float32) error {
	if len(vec) != cols {
		return fmt.Errorf("ops: len(vec)=%d, cols=%d", len(vec), cols)
	}

	if cols%quant.QK4_1 != 0 {
		return fmt.Errorf("ops: cols=%d not divisible by %d", cols, quant.QK4_1)
	}

	blocksPerRow := cols / quant.QK4_1
	want := rows * blocksPerRow * quant.BlockQ4_1Size
	if len(raw) < want {
		return fmt.Errorf("ops: Q4_1 matrix too short")
	}

	if len(out) < rows {
		return fmt.Errorf("ops: out too short")
	}

	parallelForRows(rows, func(rowStart, rowEnd int) {
		matMulVecQ4_1Rows(raw, vec, out, rowStart, rowEnd, blocksPerRow)
	})

	return nil
}

func matMulVecQ4_1Rows(raw []byte, vec, out []float32, rowStart, rowEnd, blocksPerRow int) {
	for r := rowStart; r < rowEnd; r++ {
		var sum float32
		rowOff := r * blocksPerRow * quant.BlockQ4_1Size
		for b := range blocksPerRow {
			block := raw[rowOff+b*quant.BlockQ4_1Size:]
			vecOff := b * quant.QK4_1
			dot, _ := quant.DotBlockQ4_1(block, vec[vecOff:vecOff+quant.QK4_1])
			sum += dot
		}
		out[r] = sum
	}
}

// MatMulVecQ5_0 multiplies Q5_0 matrix [rows*cols] by float32 vector [cols]
func MatMulVecQ5_0(raw []byte, rows, cols int, vec []float32) ([]float32, error) {
	out := make([]float32, rows)
	if err := MatMulVecQ5_0Into(raw, rows, cols, vec, out); err != nil {
		return nil, err
	}

	return out, nil
}

// MatMulVecQ5_0Into writes Q5_0 matmul to out [rows]
func MatMulVecQ5_0Into(raw []byte, rows, cols int, vec, out []float32) error {
	if len(vec) != cols {
		return fmt.Errorf("ops: len(vec)=%d, cols=%d", len(vec), cols)
	}

	if cols%quant.QK5_0 != 0 {
		return fmt.Errorf("ops: cols=%d not divisible by %d", cols, quant.QK5_0)
	}

	blocksPerRow := cols / quant.QK5_0
	want := rows * blocksPerRow * quant.BlockQ5_0Size
	if len(raw) < want {
		return fmt.Errorf("ops: Q5_0 matrix too short")
	}

	if len(out) < rows {
		return fmt.Errorf("ops: out too short")
	}

	parallelForRows(rows, func(rowStart, rowEnd int) {
		matMulVecQ5_0Rows(raw, vec, out, rowStart, rowEnd, blocksPerRow)
	})

	return nil
}

func matMulVecQ5_0Rows(raw []byte, vec, out []float32, rowStart, rowEnd, blocksPerRow int) {
	for r := rowStart; r < rowEnd; r++ {
		var sum float32
		rowOff := r * blocksPerRow * quant.BlockQ5_0Size
		for b := range blocksPerRow {
			block := raw[rowOff+b*quant.BlockQ5_0Size:]
			vecOff := b * quant.QK5_0
			dot, _ := quant.DotBlockQ5_0(block, vec[vecOff:vecOff+quant.QK5_0])
			sum += dot
		}
		out[r] = sum
	}
}

// MatMulVecQ5_1 multiplies Q5_1 matrix [rows*cols] by float32 vector [cols]
func MatMulVecQ5_1(raw []byte, rows, cols int, vec []float32) ([]float32, error) {
	out := make([]float32, rows)
	if err := MatMulVecQ5_1Into(raw, rows, cols, vec, out); err != nil {
		return nil, err
	}

	return out, nil
}

// MatMulVecQ5_1Into writes Q5_1 matmul to out [rows]
func MatMulVecQ5_1Into(raw []byte, rows, cols int, vec, out []float32) error {
	if len(vec) != cols {
		return fmt.Errorf("ops: len(vec)=%d, cols=%d", len(vec), cols)
	}

	if cols%quant.QK5_1 != 0 {
		return fmt.Errorf("ops: cols=%d not divisible by %d", cols, quant.QK5_1)
	}

	blocksPerRow := cols / quant.QK5_1
	want := rows * blocksPerRow * quant.BlockQ5_1Size
	if len(raw) < want {
		return fmt.Errorf("ops: Q5_1 matrix too short")
	}

	if len(out) < rows {
		return fmt.Errorf("ops: out too short")
	}

	parallelForRows(rows, func(rowStart, rowEnd int) {
		matMulVecQ5_1Rows(raw, vec, out, rowStart, rowEnd, blocksPerRow)
	})

	return nil
}

func matMulVecQ5_1Rows(raw []byte, vec, out []float32, rowStart, rowEnd, blocksPerRow int) {
	for r := rowStart; r < rowEnd; r++ {
		var sum float32
		rowOff := r * blocksPerRow * quant.BlockQ5_1Size
		for b := range blocksPerRow {
			block := raw[rowOff+b*quant.BlockQ5_1Size:]
			vecOff := b * quant.QK5_1
			dot, _ := quant.DotBlockQ5_1(block, vec[vecOff:vecOff+quant.QK5_1])
			sum += dot
		}
		out[r] = sum
	}
}

// EmbeddingQ4_1Into dequantizes an embedding row into dst
func EmbeddingQ4_1Into(dst []float32, raw []byte, dim, tokenID int) error {
	off, rowBytes, err := embeddingRowOffset(dim, tokenID, quant.BlockQ4_1Size, quant.QK4_1)
	if err != nil {
		return err
	}

	if off+rowBytes > len(raw) {
		return fmt.Errorf("ops: tokenID=%d out of range", tokenID)
	}

	return quant.DequantQ4_1Into(dst, raw[off:off+rowBytes], dim)
}

// EmbeddingQ5_0Into dequantizes an embedding row into dst
func EmbeddingQ5_0Into(dst []float32, raw []byte, dim, tokenID int) error {
	off, rowBytes, err := embeddingRowOffset(dim, tokenID, quant.BlockQ5_0Size, quant.QK5_0)
	if err != nil {
		return err
	}

	if off+rowBytes > len(raw) {
		return fmt.Errorf("ops: tokenID=%d out of range", tokenID)
	}

	return quant.DequantQ5_0Into(dst, raw[off:off+rowBytes], dim)
}

// EmbeddingQ5_1Into dequantizes an embedding row into dst
func EmbeddingQ5_1Into(dst []float32, raw []byte, dim, tokenID int) error {
	off, rowBytes, err := embeddingRowOffset(dim, tokenID, quant.BlockQ5_1Size, quant.QK5_1)
	if err != nil {
		return err
	}

	if off+rowBytes > len(raw) {
		return fmt.Errorf("ops: tokenID=%d out of range", tokenID)
	}

	return quant.DequantQ5_1Into(dst, raw[off:off+rowBytes], dim)
}
