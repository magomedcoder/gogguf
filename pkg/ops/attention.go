package ops

import (
	"fmt"
	"math"
)

// AttentionScoresInto writes attention to dst [nHeads*headDim]
// scores - buffer length >= seqLen for softmax weights (reused across heads)
func AttentionScoresInto(dst, q, k, v, scores []float32, seqLen, nHeads, nKVHeads, headDim int) error {
	return AttentionScoresIntoSoftcap(dst, q, k, v, scores, seqLen, nHeads, nKVHeads, headDim, 0)
}

// AttentionScoresIntoSoftcap like AttentionScoresInto; softcap>0 -> softcap*tanh(score/softcap) (Gemma2)
func AttentionScoresIntoSoftcap(dst, q, k, v, scores []float32, seqLen, nHeads, nKVHeads, headDim int, softcap float32) error {
	if len(dst) < nHeads*headDim {
		return fmt.Errorf("ops: dst too short")
	}

	if len(scores) < seqLen {
		return fmt.Errorf("ops: scores too short")
	}

	if nHeads%nKVHeads != 0 {
		return fmt.Errorf("ops: nHeads=%d not divisible by nKVHeads=%d", nHeads, nKVHeads)
	}

	groupSize := nHeads / nKVHeads
	scale := float32(1 / math.Sqrt(float64(headDim)))
	headScores := scores[:seqLen]
	useCap := softcap != 0
	capF := float64(softcap)

	for h := range nHeads {
		kvHead := h / groupSize
		qOff := h * headDim

		for t := range seqLen {
			kOff := t*nKVHeads*headDim + kvHead*headDim
			s := float64(dot(q[qOff:qOff+headDim], k[kOff:kOff+headDim]) * scale)
			if useCap {
				s = capF * math.Tanh(s/capF)
			}
			headScores[t] = float32(s)
		}

		SoftmaxInPlace(headScores)

		outOff := h * headDim
		vStride := nKVHeads * headDim
		vBase := kvHead * headDim
		for i := range headDim {
			dst[outOff+i] = dotStride(headScores, v, vBase+i, vStride, seqLen)
		}
	}

	return nil
}

// AttentionScores computes scaled dot-product attention for one position
// q: [nHeads*headDim]
// k: [seqLen*nKVHeads*headDim]
// v: [seqLen*nKVHeads*headDim]
func AttentionScores(q, k, v []float32, seqLen, nHeads, nKVHeads, headDim int) ([]float32, error) {
	out := make([]float32, nHeads*headDim)
	scores := make([]float32, seqLen)
	if err := AttentionScoresInto(out, q, k, v, scores, seqLen, nHeads, nKVHeads, headDim); err != nil {
		return nil, err
	}

	return out, nil
}
