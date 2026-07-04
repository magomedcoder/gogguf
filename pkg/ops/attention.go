package ops

import (
	"fmt"
	"math"
)

// AttentionScoresInto записывает attention в dst [nHeads*headDim]
func AttentionScoresInto(dst, q, k, v []float32, seqLen, nHeads, nKVHeads, headDim int) error {
	if len(dst) < nHeads*headDim {
		return fmt.Errorf("ops: dst слишком короткий")
	}

	if nHeads%nKVHeads != 0 {
		return fmt.Errorf("ops: nHeads=%d не кратно nKVHeads=%d", nHeads, nKVHeads)
	}

	groupSize := nHeads / nKVHeads
	scale := float32(1 / math.Sqrt(float64(headDim)))
	scores := make([]float32, seqLen)

	for h := range nHeads {
		kvHead := h / groupSize
		qOff := h * headDim

		for t := range seqLen {
			kOff := t*nKVHeads*headDim + kvHead*headDim
			var dotVal float32
			for i := range headDim {
				dotVal += q[qOff+i] * k[kOff+i]
			}

			scores[t] = dotVal * scale
		}

		SoftmaxInPlace(scores)

		outOff := h * headDim
		for i := range headDim {
			var sum float32
			for t := range seqLen {
				vOff := t*nKVHeads*headDim + kvHead*headDim
				sum += scores[t] * v[vOff+i]
			}
			dst[outOff+i] = sum
		}
	}

	return nil
}

// AttentionScores вычисляет scaled dot-product attention для одной позиции
// q: [nHeads*headDim]
// k: [seqLen*nKVHeads*headDim]
// v: [seqLen*nKVHeads*headDim]
func AttentionScores(q, k, v []float32, seqLen, nHeads, nKVHeads, headDim int) ([]float32, error) {
	out := make([]float32, nHeads*headDim)
	if err := AttentionScoresInto(out, q, k, v, seqLen, nHeads, nKVHeads, headDim); err != nil {
		return nil, err
	}

	return out, nil
}
