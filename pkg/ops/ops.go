package ops

import (
	"fmt"
	"math"

	"github.com/magomedcoder/gogguf/pkg/quant"
)

const parallelMatMulMinRows = 64

// MatMulVec умножает матрицу [rows*cols] на вектор [cols]
func MatMulVec(matrix []float32, rows, cols int, vec []float32) ([]float32, error) {
	if len(vec) != cols {
		return nil, fmt.Errorf("ops: len(vec)=%d, cols=%d", len(vec), cols)
	}

	if len(matrix) < rows*cols {
		return nil, fmt.Errorf("ops: matrix слишком короткая")
	}

	out := make([]float32, rows)
	if err := MatMulVecInto(matrix, rows, cols, vec, out); err != nil {
		return nil, err
	}
	return out, nil
}

// MatMulVecInto записывает matmul в out [rows]
func MatMulVecInto(matrix []float32, rows, cols int, vec, out []float32) error {
	if len(vec) != cols {
		return fmt.Errorf("ops: len(vec)=%d, cols=%d", len(vec), cols)
	}

	if len(matrix) < rows*cols {
		return fmt.Errorf("ops: matrix слишком короткая")
	}

	if len(out) < rows {
		return fmt.Errorf("ops: out слишком короткий")
	}

	parallelForRows(rows, func(rowStart, rowEnd int) {
		matMulVecRows(matrix, vec, out, rowStart, rowEnd, cols)
	})
	return nil
}

func matMulVecRows(matrix, vec, out []float32, rowStart, rowEnd, cols int) {
	for r := rowStart; r < rowEnd; r++ {
		out[r] = dot(matrix[r*cols:(r+1)*cols], vec)
	}
}

// MatMulVecQ8_0 умножает Q8_0-матрицу [rows*cols] на float32-вектор [cols]
func MatMulVecQ8_0(raw []byte, rows, cols int, vec []float32) ([]float32, error) {
	if len(vec) != cols {
		return nil, fmt.Errorf("ops: len(vec)=%d, cols=%d", len(vec), cols)
	}

	if cols%quant.QK8_0 != 0 {
		return nil, fmt.Errorf("ops: cols=%d не кратно %d", cols, quant.QK8_0)
	}

	blocksPerRow := cols / quant.QK8_0
	want := rows * blocksPerRow * quant.BlockQ8_0Size
	if len(raw) < want {
		return nil, fmt.Errorf("ops: Q8_0 matrix слишком короткая")
	}

	out := make([]float32, rows)
	if err := MatMulVecQ8_0Into(raw, rows, cols, vec, out); err != nil {
		return nil, err
	}

	return out, nil
}

// MatMulVecQ8_0Into записывает Q8_0 matmul в out [rows]
func MatMulVecQ8_0Into(raw []byte, rows, cols int, vec, out []float32) error {
	if len(vec) != cols {
		return fmt.Errorf("ops: len(vec)=%d, cols=%d", len(vec), cols)
	}

	if cols%quant.QK8_0 != 0 {
		return fmt.Errorf("ops: cols=%d не кратно %d", cols, quant.QK8_0)
	}

	blocksPerRow := cols / quant.QK8_0
	want := rows * blocksPerRow * quant.BlockQ8_0Size
	if len(raw) < want {
		return fmt.Errorf("ops: Q8_0 matrix слишком короткая")
	}

	if len(out) < rows {
		return fmt.Errorf("ops: out слишком короткий")
	}

	parallelForRows(rows, func(rowStart, rowEnd int) {
		matMulVecQ8_0Rows(raw, vec, out, rowStart, rowEnd, blocksPerRow)
	})

	return nil
}

func matMulVecQ8_0Rows(raw []byte, vec, out []float32, rowStart, rowEnd, blocksPerRow int) {
	for r := rowStart; r < rowEnd; r++ {
		var sum float32
		rowOff := r * blocksPerRow * quant.BlockQ8_0Size
		for b := range blocksPerRow {
			block := raw[rowOff+b*quant.BlockQ8_0Size:]
			vecOff := b * quant.QK8_0
			dot, _ := quant.DotBlockQ8_0(block, vec[vecOff:vecOff+quant.QK8_0])
			sum += dot
		}
		out[r] = sum
	}
}

// MatMulVecQ4_0 умножает Q4_0-матрицу [rows*cols] на float32-вектор [cols]
func MatMulVecQ4_0(raw []byte, rows, cols int, vec []float32) ([]float32, error) {
	if len(vec) != cols {
		return nil, fmt.Errorf("ops: len(vec)=%d, cols=%d", len(vec), cols)
	}

	if cols%quant.QK4_0 != 0 {
		return nil, fmt.Errorf("ops: cols=%d не кратно %d", cols, quant.QK4_0)
	}

	blocksPerRow := cols / quant.QK4_0
	want := rows * blocksPerRow * quant.BlockQ4_0Size
	if len(raw) < want {
		return nil, fmt.Errorf("ops: Q4_0 matrix слишком короткая")
	}

	out := make([]float32, rows)
	if err := MatMulVecQ4_0Into(raw, rows, cols, vec, out); err != nil {
		return nil, err
	}

	return out, nil
}

// MatMulVecQ4_0Into записывает Q4_0 matmul в out [rows]
func MatMulVecQ4_0Into(raw []byte, rows, cols int, vec, out []float32) error {
	if len(vec) != cols {
		return fmt.Errorf("ops: len(vec)=%d, cols=%d", len(vec), cols)
	}

	if cols%quant.QK4_0 != 0 {
		return fmt.Errorf("ops: cols=%d не кратно %d", cols, quant.QK4_0)
	}

	blocksPerRow := cols / quant.QK4_0
	want := rows * blocksPerRow * quant.BlockQ4_0Size
	if len(raw) < want {
		return fmt.Errorf("ops: Q4_0 matrix слишком короткая")
	}

	if len(out) < rows {
		return fmt.Errorf("ops: out слишком короткий")
	}

	parallelForRows(rows, func(rowStart, rowEnd int) {
		matMulVecQ4_0Rows(raw, vec, out, rowStart, rowEnd, blocksPerRow)
	})

	return nil
}

func matMulVecQ4_0Rows(raw []byte, vec, out []float32, rowStart, rowEnd, blocksPerRow int) {
	for r := rowStart; r < rowEnd; r++ {
		var sum float32
		rowOff := r * blocksPerRow * quant.BlockQ4_0Size
		for b := range blocksPerRow {
			block := raw[rowOff+b*quant.BlockQ4_0Size:]
			vecOff := b * quant.QK4_0
			dot, _ := quant.DotBlockQ4_0(block, vec[vecOff:vecOff+quant.QK4_0])
			sum += dot
		}
		out[r] = sum
	}
}

// MatMulVecQ4_K умножает Q4_K-матрицу [rows*cols] на float32-вектор [cols]
func MatMulVecQ4_K(raw []byte, rows, cols int, vec []float32) ([]float32, error) {
	if len(vec) != cols {
		return nil, fmt.Errorf("ops: len(vec)=%d, cols=%d", len(vec), cols)
	}

	if cols%quant.QK_K != 0 {
		return nil, fmt.Errorf("ops: cols=%d не кратно %d", cols, quant.QK_K)
	}

	blocksPerRow := cols / quant.QK_K
	want := rows * blocksPerRow * quant.BlockQ4_KSize
	if len(raw) < want {
		return nil, fmt.Errorf("ops: Q4_K matrix слишком короткая")
	}

	out := make([]float32, rows)
	if err := MatMulVecQ4_KInto(raw, rows, cols, vec, out); err != nil {
		return nil, err
	}

	return out, nil
}

// MatMulVecQ4_KInto записывает Q4_K matmul в out [rows]
func MatMulVecQ4_KInto(raw []byte, rows, cols int, vec, out []float32) error {
	if len(vec) != cols {
		return fmt.Errorf("ops: len(vec)=%d, cols=%d", len(vec), cols)
	}

	if cols%quant.QK_K != 0 {
		return fmt.Errorf("ops: cols=%d не кратно %d", cols, quant.QK_K)
	}

	blocksPerRow := cols / quant.QK_K
	want := rows * blocksPerRow * quant.BlockQ4_KSize
	if len(raw) < want {
		return fmt.Errorf("ops: Q4_K matrix слишком короткая")
	}

	if len(out) < rows {
		return fmt.Errorf("ops: out слишком короткий")
	}

	parallelForRows(rows, func(rowStart, rowEnd int) {
		matMulVecQ4_KRows(raw, vec, out, rowStart, rowEnd, blocksPerRow)
	})

	return nil
}

func matMulVecQ4_KRows(raw []byte, vec, out []float32, rowStart, rowEnd, blocksPerRow int) {
	for r := rowStart; r < rowEnd; r++ {
		var sum float32
		rowOff := r * blocksPerRow * quant.BlockQ4_KSize
		for b := range blocksPerRow {
			block := raw[rowOff+b*quant.BlockQ4_KSize:]
			vecOff := b * quant.QK_K
			dot, _ := quant.DotBlockQ4_K(block, vec[vecOff:vecOff+quant.QK_K])
			sum += dot
		}
		out[r] = sum
	}
}

// MatMulVecQ6_KInto записывает Q6_K matmul в out [rows]
func MatMulVecQ6_KInto(raw []byte, rows, cols int, vec, out []float32) error {
	if len(vec) != cols {
		return fmt.Errorf("ops: len(vec)=%d, cols=%d", len(vec), cols)
	}

	if cols%quant.QK_K != 0 {
		return fmt.Errorf("ops: cols=%d не кратно %d", cols, quant.QK_K)
	}

	blocksPerRow := cols / quant.QK_K
	want := rows * blocksPerRow * quant.BlockQ6_KSize
	if len(raw) < want {
		return fmt.Errorf("ops: Q6_K matrix слишком короткая")
	}

	if len(out) < rows {
		return fmt.Errorf("ops: out слишком короткий")
	}

	parallelForRows(rows, func(rowStart, rowEnd int) {
		matMulVecQ6_KRows(raw, vec, out, rowStart, rowEnd, blocksPerRow)
	})

	return nil
}

func matMulVecQ6_KRows(raw []byte, vec, out []float32, rowStart, rowEnd, blocksPerRow int) {
	for r := rowStart; r < rowEnd; r++ {
		var sum float32
		rowOff := r * blocksPerRow * quant.BlockQ6_KSize
		for b := range blocksPerRow {
			block := raw[rowOff+b*quant.BlockQ6_KSize:]
			vecOff := b * quant.QK_K
			dot, _ := quant.DotBlockQ6_K(block, vec[vecOff:vecOff+quant.QK_K])
			sum += dot
		}
		out[r] = sum
	}
}

// RMSNorm применяет RMS-нормализацию: x * weight / RMS(x)
func RMSNorm(x, weight []float32, eps float32) ([]float32, error) {
	return rmsnorm(x, weight, eps)
}

// Add поэлементно складывает a и b
func Add(a, b []float32) []float32 {
	out := make([]float32, len(a))
	for i := range a {
		out[i] = a[i] + b[i]
	}

	return out
}

// Scale умножает вектор на скаляр
func Scale(x []float32, s float32) []float32 {
	out := make([]float32, len(x))
	for i, v := range x {
		out[i] = v * s
	}
	return out
}

// SiLU: x * sigmoid(x)
func SiLU(x float32) float32 {
	return x / (1 + float32(math.Exp(float64(-x))))
}

// VectorRMS возвращает sqrt(mean(x²)) - метрика для сверки hidden states
func VectorRMS(x []float32) float32 {
	if len(x) == 0 {
		return 0
	}

	var sumSq float64
	for _, v := range x {
		sumSq += float64(v) * float64(v)
	}

	return float32(math.Sqrt(sumSq / float64(len(x))))
}
