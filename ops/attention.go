package ops

import (
	"fmt"
	"math"
)

// AttentionScores вычисляет scaled dot-product attention для одной позиции
// q: [nHeads×headDim]
// k: [seqLen×nKVHeads×headDim]
// v: [seqLen×nKVHeads×headDim]
func AttentionScores(q, k, v []float32, seqLen, nHeads, nKVHeads, headDim int) ([]float32, error) {
	if nHeads%nKVHeads != 0 {
		return nil, fmt.Errorf("ops: nHeads=%d не кратно nKVHeads=%d", nHeads, nKVHeads)
	}

	groupSize := nHeads / nKVHeads
	scale := float32(1 / math.Sqrt(float64(headDim)))
	out := make([]float32, nHeads*headDim)

	for h := 0; h < nHeads; h++ {
		kvHead := h / groupSize
		qOff := h * headDim
		scores := make([]float32, seqLen)

		for t := 0; t < seqLen; t++ {
			kOff := t*nKVHeads*headDim + kvHead*headDim
			var dot float32
			for i := 0; i < headDim; i++ {
				dot += q[qOff+i] * k[kOff+i]
			}
			
			scores[t] = dot * scale
		}

		SoftmaxInPlace(scores)

		outOff := h * headDim
		for i := 0; i < headDim; i++ {
			var sum float32
			for t := 0; t < seqLen; t++ {
				vOff := t*nKVHeads*headDim + kvHead*headDim
				sum += scores[t] * v[vOff+i]
			}
			out[outOff+i] = sum
		}
	}

	return out, nil
}
