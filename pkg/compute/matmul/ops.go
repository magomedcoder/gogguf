package matmul

import (
	"fmt"
	cvec "github.com/magomedcoder/gogguf/pkg/compute/vec"

	"github.com/magomedcoder/gogguf/pkg/quant"
)

const parallelMatMulMinRows = 64

// MatMulVec multiplies matrix [rows*cols] by vector [cols].
// Allocating wrapper - prefer MatMulVecInto on hot paths.
func MatMulVec(matrix []float32, rows, cols int, vec []float32) ([]float32, error) {
	if len(vec) != cols {
		return nil, fmt.Errorf("compute: len(vec)=%d, cols=%d", len(vec), cols)
	}

	if len(matrix) < rows*cols {
		return nil, fmt.Errorf("compute: matrix too short")
	}

	out := make([]float32, rows)
	if err := MatMulVecInto(matrix, rows, cols, vec, out); err != nil {
		return nil, err
	}
	return out, nil
}

// MatMulVecInto writes matmul to out [rows]
func MatMulVecInto(matrix []float32, rows, cols int, vec, out []float32) error {
	if len(vec) != cols {
		return fmt.Errorf("compute: len(vec)=%d, cols=%d", len(vec), cols)
	}

	if len(matrix) < rows*cols {
		return fmt.Errorf("compute: matrix too short")
	}

	if len(out) < rows {
		return fmt.Errorf("compute: out too short")
	}

	cvec.ParallelForRows(rows, func(rowStart, rowEnd int) {
		matMulVecRows(matrix, vec, out, rowStart, rowEnd, cols)
	})
	return nil
}

func matMulVecRows(matrix, vec, out []float32, rowStart, rowEnd, cols int) {
	for r := rowStart; r < rowEnd; r++ {
		out[r] = cvec.Dot(matrix[r*cols:(r+1)*cols], vec)
	}
}

// MatMulVecQ8_0 multiplies Q8_0 matrix [rows*cols] by float32 vector [cols]
func MatMulVecQ8_0(raw []byte, rows, cols int, vec []float32) ([]float32, error) {
	if len(vec) != cols {
		return nil, fmt.Errorf("compute: len(vec)=%d, cols=%d", len(vec), cols)
	}

	if cols%quant.QK8_0 != 0 {
		return nil, fmt.Errorf("compute: cols=%d not divisible by %d", cols, quant.QK8_0)
	}

	blocksPerRow := cols / quant.QK8_0
	want := rows * blocksPerRow * quant.BlockQ8_0Size
	if len(raw) < want {
		return nil, fmt.Errorf("compute: Q8_0 matrix too short")
	}

	out := make([]float32, rows)
	if err := MatMulVecQ8_0Into(raw, rows, cols, vec, out); err != nil {
		return nil, err
	}

	return out, nil
}

// MatMulVecQ8_0Into writes Q8_0 matmul to out [rows]
func MatMulVecQ8_0Into(raw []byte, rows, cols int, vec, out []float32) error {
	if len(vec) != cols {
		return fmt.Errorf("compute: len(vec)=%d, cols=%d", len(vec), cols)
	}

	if cols%quant.QK8_0 != 0 {
		return fmt.Errorf("compute: cols=%d not divisible by %d", cols, quant.QK8_0)
	}

	blocksPerRow := cols / quant.QK8_0
	want := rows * blocksPerRow * quant.BlockQ8_0Size
	if len(raw) < want {
		return fmt.Errorf("compute: Q8_0 matrix too short")
	}

	if len(out) < rows {
		return fmt.Errorf("compute: out too short")
	}

	return cvec.ParallelForRowsErr(rows, func(rowStart, rowEnd int) error {
		return matMulVecQ8_0Rows(raw, vec, out, rowStart, rowEnd, blocksPerRow)
	})
}

func matMulVecQ8_0Rows(raw []byte, vec, out []float32, rowStart, rowEnd, blocksPerRow int) error {
	for r := rowStart; r < rowEnd; r++ {
		var sum float32
		rowOff := r * blocksPerRow * quant.BlockQ8_0Size
		for b := range blocksPerRow {
			block := raw[rowOff+b*quant.BlockQ8_0Size:]
			vecOff := b * quant.QK8_0
			dot, errDot := quant.DotBlockQ8_0(block, vec[vecOff:vecOff+quant.QK8_0])
			if errDot != nil {
				return errDot
			}
			sum += dot
		}
		out[r] = sum
	}

	return nil
}

// MatMulVecQ4_0 multiplies Q4_0 matrix [rows*cols] by float32 vector [cols]
func MatMulVecQ4_0(raw []byte, rows, cols int, vec []float32) ([]float32, error) {
	if len(vec) != cols {
		return nil, fmt.Errorf("compute: len(vec)=%d, cols=%d", len(vec), cols)
	}

	if cols%quant.QK4_0 != 0 {
		return nil, fmt.Errorf("compute: cols=%d not divisible by %d", cols, quant.QK4_0)
	}

	blocksPerRow := cols / quant.QK4_0
	want := rows * blocksPerRow * quant.BlockQ4_0Size
	if len(raw) < want {
		return nil, fmt.Errorf("compute: Q4_0 matrix too short")
	}

	out := make([]float32, rows)
	if err := MatMulVecQ4_0Into(raw, rows, cols, vec, out); err != nil {
		return nil, err
	}

	return out, nil
}

// MatMulVecQ4_0Into writes Q4_0 matmul to out [rows]
func MatMulVecQ4_0Into(raw []byte, rows, cols int, vec, out []float32) error {
	if len(vec) != cols {
		return fmt.Errorf("compute: len(vec)=%d, cols=%d", len(vec), cols)
	}

	if cols%quant.QK4_0 != 0 {
		return fmt.Errorf("compute: cols=%d not divisible by %d", cols, quant.QK4_0)
	}

	blocksPerRow := cols / quant.QK4_0
	want := rows * blocksPerRow * quant.BlockQ4_0Size
	if len(raw) < want {
		return fmt.Errorf("compute: Q4_0 matrix too short")
	}

	if len(out) < rows {
		return fmt.Errorf("compute: out too short")
	}

	return cvec.ParallelForRowsErr(rows, func(rowStart, rowEnd int) error {
		return matMulVecQ4_0Rows(raw, vec, out, rowStart, rowEnd, blocksPerRow)
	})
}

func matMulVecQ4_0Rows(raw []byte, vec, out []float32, rowStart, rowEnd, blocksPerRow int) error {
	for r := rowStart; r < rowEnd; r++ {
		var sum float32
		rowOff := r * blocksPerRow * quant.BlockQ4_0Size
		for b := range blocksPerRow {
			block := raw[rowOff+b*quant.BlockQ4_0Size:]
			vecOff := b * quant.QK4_0
			dot, errDot := quant.DotBlockQ4_0(block, vec[vecOff:vecOff+quant.QK4_0])
			if errDot != nil {
				return errDot
			}
			sum += dot
		}
		out[r] = sum
	}

	return nil
}

// MatMulVecQ4_K multiplies Q4_K matrix [rows*cols] by float32 vector [cols]
func MatMulVecQ4_K(raw []byte, rows, cols int, vec []float32) ([]float32, error) {
	if len(vec) != cols {
		return nil, fmt.Errorf("compute: len(vec)=%d, cols=%d", len(vec), cols)
	}

	if cols%quant.QK_K != 0 {
		return nil, fmt.Errorf("compute: cols=%d not divisible by %d", cols, quant.QK_K)
	}

	blocksPerRow := cols / quant.QK_K
	want := rows * blocksPerRow * quant.BlockQ4_KSize
	if len(raw) < want {
		return nil, fmt.Errorf("compute: Q4_K matrix too short")
	}

	out := make([]float32, rows)
	if err := MatMulVecQ4_KInto(raw, rows, cols, vec, out); err != nil {
		return nil, err
	}

	return out, nil
}

// MatMulVecQ4_KInto writes Q4_K matmul to out [rows]
func MatMulVecQ4_KInto(raw []byte, rows, cols int, vec, out []float32) error {
	if len(vec) != cols {
		return fmt.Errorf("compute: len(vec)=%d, cols=%d", len(vec), cols)
	}

	if cols%quant.QK_K != 0 {
		return fmt.Errorf("compute: cols=%d not divisible by %d", cols, quant.QK_K)
	}

	blocksPerRow := cols / quant.QK_K
	want := rows * blocksPerRow * quant.BlockQ4_KSize
	if len(raw) < want {
		return fmt.Errorf("compute: Q4_K matrix too short")
	}

	if len(out) < rows {
		return fmt.Errorf("compute: out too short")
	}

	return cvec.ParallelForRowsErr(rows, func(rowStart, rowEnd int) error {
		return matMulVecQ4_KRows(raw, vec, out, rowStart, rowEnd, blocksPerRow)
	})
}

func matMulVecQ4_KRows(raw []byte, vec, out []float32, rowStart, rowEnd, blocksPerRow int) error {
	for r := rowStart; r < rowEnd; r++ {
		var sum float32
		rowOff := r * blocksPerRow * quant.BlockQ4_KSize
		for b := range blocksPerRow {
			block := raw[rowOff+b*quant.BlockQ4_KSize:]
			vecOff := b * quant.QK_K
			dot, errDot := quant.DotBlockQ4_K(block, vec[vecOff:vecOff+quant.QK_K])
			if errDot != nil {
				return errDot
			}
			sum += dot
		}
		out[r] = sum
	}

	return nil
}

// MatMulVecQ5_K multiplies Q5_K matrix [rows*cols] by float32 vector [cols]
func MatMulVecQ5_K(raw []byte, rows, cols int, vec []float32) ([]float32, error) {
	out := make([]float32, rows)
	if err := MatMulVecQ5_KInto(raw, rows, cols, vec, out); err != nil {
		return nil, err
	}

	return out, nil
}

// MatMulVecQ5_KInto writes Q5_K matmul to out [rows]
func MatMulVecQ5_KInto(raw []byte, rows, cols int, vec, out []float32) error {
	if len(vec) != cols {
		return fmt.Errorf("compute: len(vec)=%d, cols=%d", len(vec), cols)
	}

	if cols%quant.QK_K != 0 {
		return fmt.Errorf("compute: cols=%d not divisible by %d", cols, quant.QK_K)
	}

	blocksPerRow := cols / quant.QK_K
	want := rows * blocksPerRow * quant.BlockQ5_KSize
	if len(raw) < want {
		return fmt.Errorf("compute: Q5_K matrix too short")
	}

	if len(out) < rows {
		return fmt.Errorf("compute: out too short")
	}

	return cvec.ParallelForRowsErr(rows, func(rowStart, rowEnd int) error {
		return matMulVecQ5_KRows(raw, vec, out, rowStart, rowEnd, blocksPerRow)
	})
}

func matMulVecQ5_KRows(raw []byte, vec, out []float32, rowStart, rowEnd, blocksPerRow int) error {
	for r := rowStart; r < rowEnd; r++ {
		var sum float32
		rowOff := r * blocksPerRow * quant.BlockQ5_KSize
		for b := range blocksPerRow {
			block := raw[rowOff+b*quant.BlockQ5_KSize:]
			vecOff := b * quant.QK_K
			dot, errDot := quant.DotBlockQ5_K(block, vec[vecOff:vecOff+quant.QK_K])
			if errDot != nil {
				return errDot
			}
			sum += dot
		}
		out[r] = sum
	}

	return nil
}

// MatMulVecQ6_K multiplies Q6_K matrix [rows*cols] by float32 vector [cols]
func MatMulVecQ6_K(raw []byte, rows, cols int, vec []float32) ([]float32, error) {
	out := make([]float32, rows)
	if err := MatMulVecQ6_KInto(raw, rows, cols, vec, out); err != nil {
		return nil, err
	}

	return out, nil
}

// MatMulVecQ6_KInto writes Q6_K matmul to out [rows]
func MatMulVecQ6_KInto(raw []byte, rows, cols int, vec, out []float32) error {
	if len(vec) != cols {
		return fmt.Errorf("compute: len(vec)=%d, cols=%d", len(vec), cols)
	}

	if cols%quant.QK_K != 0 {
		return fmt.Errorf("compute: cols=%d not divisible by %d", cols, quant.QK_K)
	}

	blocksPerRow := cols / quant.QK_K
	want := rows * blocksPerRow * quant.BlockQ6_KSize
	if len(raw) < want {
		return fmt.Errorf("compute: Q6_K matrix too short")
	}

	if len(out) < rows {
		return fmt.Errorf("compute: out too short")
	}

	return cvec.ParallelForRowsErr(rows, func(rowStart, rowEnd int) error {
		return matMulVecQ6_KRows(raw, vec, out, rowStart, rowEnd, blocksPerRow)
	})
}

func matMulVecQ6_KRows(raw []byte, vec, out []float32, rowStart, rowEnd, blocksPerRow int) error {
	for r := rowStart; r < rowEnd; r++ {
		var sum float32
		rowOff := r * blocksPerRow * quant.BlockQ6_KSize
		for b := range blocksPerRow {
			block := raw[rowOff+b*quant.BlockQ6_KSize:]
			vecOff := b * quant.QK_K
			dot, errDot := quant.DotBlockQ6_K(block, vec[vecOff:vecOff+quant.QK_K])
			if errDot != nil {
				return errDot
			}
			sum += dot
		}
		out[r] = sum
	}

	return nil
}
