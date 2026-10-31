package moe

import (
	"github.com/magomedcoder/gogguf/pkg/compute"
	"github.com/magomedcoder/gogguf/pkg/exec/dispatch"
	"github.com/magomedcoder/gogguf/pkg/weights"
)

// ExpertsSwiGLUInto runs selected experts: gate/up SwiGLU down, weighted into acc.
// acc must be cleared by caller. gateBuf/upBuf/tmp are scratch [ffn]/[ffn]/[embd].
func ExpertsSwiGLUInto(store *weights.Store, gateName, upName, downName string, idxs []int, weights []float32, h, gateBuf, upBuf, tmp, acc []float32, embd, ffn int) error {
	for i, ei := range idxs {
		if err := dispatch.MatMulExpertInto(store, gateName, ei, ffn, embd, h, gateBuf); err != nil {
			return err
		}

		if err := dispatch.MatMulExpertInto(store, upName, ei, ffn, embd, h, upBuf); err != nil {
			return err
		}

		compute.SwiGLUInPlace(gateBuf, upBuf)
		if err := dispatch.MatMulExpertInto(store, downName, ei, embd, ffn, gateBuf, tmp); err != nil {
			return err
		}

		w := weights[i]
		for j := range embd {
			acc[j] += tmp[j] * w
		}
	}

	return nil
}
