package ops

import (
	"fmt"
	"math"

	"github.com/magomedcoder/gguf.go/pkg/quant"
)

// MatMulVec умножает матрицу [rows*cols] на вектор [cols]
func MatMulVec(matrix []float32, rows, cols int, vec []float32) ([]float32, error) {
	if len(vec) != cols {
		return nil, fmt.Errorf("ops: len(vec)=%d, cols=%d", len(vec), cols)
	}

	if len(matrix) < rows*cols {
		return nil, fmt.Errorf("ops: matrix слишком короткая")
	}

	out := make([]float32, rows)
	for r := range rows {
		out[r] = dot(matrix[r*cols:(r+1)*cols], vec)
	}

	return out, nil
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
	for r := range rows {
		var sum float32
		rowOff := r * blocksPerRow * quant.BlockQ8_0Size
		for b := range blocksPerRow {
			block := raw[rowOff+b*quant.BlockQ8_0Size:]
			vecOff := b * quant.QK8_0
			dot, err := quant.DotBlockQ8_0(block, vec[vecOff:vecOff+quant.QK8_0])
			if err != nil {
				return nil, err
			}
			sum += dot
		}
		out[r] = sum
	}

	return out, nil
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
	for r := range rows {
		var sum float32
		rowOff := r * blocksPerRow * quant.BlockQ4_0Size
		for b := range blocksPerRow {
			block := raw[rowOff+b*quant.BlockQ4_0Size:]
			vecOff := b * quant.QK4_0
			dot, err := quant.DotBlockQ4_0(block, vec[vecOff:vecOff+quant.QK4_0])
			if err != nil {
				return nil, err
			}
			sum += dot
		}
		out[r] = sum
	}

	return out, nil
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
	for r := range rows {
		var sum float32
		rowOff := r * blocksPerRow * quant.BlockQ4_KSize
		for b := range blocksPerRow {
			block := raw[rowOff+b*quant.BlockQ4_KSize:]
			vecOff := b * quant.QK_K
			dot, err := quant.DotBlockQ4_K(block, vec[vecOff:vecOff+quant.QK_K])
			if err != nil {
				return nil, err
			}

			sum += dot
		}

		out[r] = sum
	}

	return out, nil
}

// RMSNorm применяет RMS-нормализацию: x * weight / RMS(x)
func RMSNorm(x, weight []float32, eps float32) ([]float32, error) {
	if len(x) != len(weight) {
		return nil, fmt.Errorf("ops: x и weight разной длины")
	}

	var sumSq float32
	for _, v := range x {
		sumSq += v * v
	}
	scale := float32(1) / float32(math.Sqrt(float64(sumSq/float32(len(x))+eps)))

	out := make([]float32, len(x))
	for i := range x {
		out[i] = x[i] * scale * weight[i]
	}

	return out, nil
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
