package ops

import (
	"fmt"

	"github.com/magomedcoder/gogguf/pkg/quant"
)

// MatMulMatInto: W [rows*cols] * X [batch*cols] -> out [batch*rows] X и out - подряд идущие векторы (токен b с offset b*cols / b*rows)
func MatMulMatInto(matrix []float32, rows, cols int, x []float32, batch int, out []float32) error {
	if batch < 1 {
		return fmt.Errorf("ops: batch=%d", batch)
	}

	if len(x) < batch*cols {
		return fmt.Errorf("ops: x слишком короткий для batch=%d", batch)
	}

	if len(out) < batch*rows {
		return fmt.Errorf("ops: out слишком короткий для batch=%d", batch)
	}

	if len(matrix) < rows*cols {
		return fmt.Errorf("ops: matrix слишком короткая")
	}

	parallelForRows(rows, func(rowStart, rowEnd int) {
		for r := rowStart; r < rowEnd; r++ {
			row := matrix[r*cols : (r+1)*cols]
			for b := range batch {
				out[b*rows+r] = dot(row, x[b*cols:(b+1)*cols])
			}
		}
	})
	return nil
}

// MatMulMatQ8_0Into - то же для Q8_0-матрицы.
func MatMulMatQ8_0Into(raw []byte, rows, cols int, x []float32, batch int, out []float32) error {
	if batch < 1 {
		return fmt.Errorf("ops: batch=%d", batch)
	}

	if len(x) < batch*cols {
		return fmt.Errorf("ops: x слишком короткий для batch=%d", batch)
	}

	if len(out) < batch*rows {
		return fmt.Errorf("ops: out слишком короткий для batch=%d", batch)
	}

	if cols%quant.QK8_0 != 0 {
		return fmt.Errorf("ops: cols=%d не кратно %d", cols, quant.QK8_0)
	}

	blocksPerRow := cols / quant.QK8_0
	want := rows * blocksPerRow * quant.BlockQ8_0Size
	if len(raw) < want {
		return fmt.Errorf("ops: Q8_0 matrix слишком короткая")
	}

	parallelForRows(rows, func(rowStart, rowEnd int) {
		for r := rowStart; r < rowEnd; r++ {
			rowOff := r * blocksPerRow * quant.BlockQ8_0Size
			for b := range batch {
				var sum float32
				xb := x[b*cols : (b+1)*cols]
				for blk := range blocksPerRow {
					block := raw[rowOff+blk*quant.BlockQ8_0Size:]
					vecOff := blk * quant.QK8_0
					d, _ := quant.DotBlockQ8_0(block, xb[vecOff:vecOff+quant.QK8_0])
					sum += d
				}
				out[b*rows+r] = sum
			}
		}
	})

	return nil
}

// MatMulMatViaVecInto - batch через MatMulVecInto (любой уже готовый vec-kernel)
func MatMulMatViaVecInto(batch int, rows, cols int, x, out []float32, mul func(vec, dst []float32) error) error {
	if batch < 1 {
		return fmt.Errorf("ops: batch=%d", batch)
	}

	if len(x) < batch*cols || len(out) < batch*rows {
		return fmt.Errorf("ops: буферы слишком короткие для batch=%d", batch)
	}

	for b := range batch {
		if err := mul(x[b*cols:(b+1)*cols], out[b*rows:(b+1)*rows]); err != nil {
			return err
		}
	}

	return nil
}
