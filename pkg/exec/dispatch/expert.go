package dispatch

import (
	"fmt"
	"github.com/magomedcoder/gogguf/pkg/dtype"

	"github.com/magomedcoder/gogguf/pkg/compute"
	"github.com/magomedcoder/gogguf/pkg/quant"
	"github.com/magomedcoder/gogguf/pkg/weights"
)

// MatMulExpertInto multiplies one MoE expert slice [rows*cols] from a packed
// [n_exp][rows][cols] tensor without dequantizing the full pack on host.
func MatMulExpertInto(store *weights.Store, name string, expert, rows, cols int, vec, out []float32) error {
	info, err := store.Info(name)
	if err != nil {
		return err
	}

	switch info.Type {
	case dtype.Q8_0,
		dtype.Q4_0,
		dtype.Q4_1,
		dtype.Q5_0,
		dtype.Q5_1,
		dtype.Q4_K,
		dtype.Q5_K,
		dtype.Q6_K,
		dtype.Q2_K,
		dtype.Q3_K,
		dtype.Q8_K:
		nbytes, err := expertQuantBytes(info.Type, rows, cols)
		if err != nil {
			return fmt.Errorf("dispatch: %s: %w", name, err)
		}

		raw, err := store.Raw(name)
		if err != nil {
			return err
		}

		off := expert * nbytes
		if off < 0 || off+nbytes > len(raw) {
			return fmt.Errorf("dispatch: %s: expert %d out of bounds", name, expert)
		}

		return MatMulTypedInto(info.Type, raw[off:off+nbytes], rows, cols, vec, out, nil, name)
	default:
		f32, err := store.Floats(name)
		if err != nil {
			return err
		}

		elems := rows * cols
		off := expert * elems
		if off < 0 || off+elems > len(f32) {
			return fmt.Errorf("dispatch: %s: expert %d out of bounds", name, expert)
		}

		return compute.MatMulVecInto(f32[off:off+elems], rows, cols, vec, out)
	}
}

func expertQuantBytes(t dtype.DType, rows, cols int) (int, error) {
	if rows <= 0 || cols <= 0 {
		return 0, fmt.Errorf("rows=%d cols=%d", rows, cols)
	}

	var blk, size int
	switch t {
	case dtype.Q8_0:
		blk, size = quant.QK8_0, quant.BlockQ8_0Size
	case dtype.Q4_0:
		blk, size = quant.QK4_0, quant.BlockQ4_0Size
	case dtype.Q4_1:
		blk, size = quant.QK4_1, quant.BlockQ4_1Size
	case dtype.Q5_0:
		blk, size = quant.QK5_0, quant.BlockQ5_0Size
	case dtype.Q5_1:
		blk, size = quant.QK5_1, quant.BlockQ5_1Size
	case dtype.Q4_K:
		blk, size = quant.QK_K, quant.BlockQ4_KSize
	case dtype.Q5_K:
		blk, size = quant.QK_K, quant.BlockQ5_KSize
	case dtype.Q6_K:
		blk, size = quant.QK_K, quant.BlockQ6_KSize
	case dtype.Q2_K:
		blk, size = quant.QK_K, quant.BlockQ2_KSize
	case dtype.Q3_K:
		blk, size = quant.QK_K, quant.BlockQ3_KSize
	case dtype.Q8_K:
		blk, size = quant.QK_K, quant.BlockQ8_KSize
	default:
		return 0, fmt.Errorf("type %s not supported", t)
	}

	if cols%blk != 0 {
		return 0, fmt.Errorf("cols=%d not divisible by %d", cols, blk)
	}

	return rows * (cols / blk) * size, nil
}
