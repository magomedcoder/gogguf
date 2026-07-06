package ops

import (
	"fmt"
	"math"
)

// AttentionScoresInto записывает attention в dst [nHeads*headDim]
// scores - буфер длины >= seqLen для softmax weights (переиспользуется между головами)
func AttentionScoresInto(dst, q, k, v, scores []float32, seqLen, nHeads, nKVHeads, headDim int) error {
	if len(dst) < nHeads*headDim {
		return fmt.Errorf("ops: dst слишком короткий")
	}

	if len(scores) < seqLen {
		return fmt.Errorf("ops: scores слишком короткий")
	}

	if nHeads%nKVHeads != 0 {
		return fmt.Errorf("ops: nHeads=%d не кратно nKVHeads=%d", nHeads, nKVHeads)
	}

	groupSize := nHeads / nKVHeads
	scale := float32(1 / math.Sqrt(float64(headDim)))
	headScores := scores[:seqLen]

	for h := range nHeads {
		kvHead := h / groupSize
		qOff := h * headDim

		for t := range seqLen {
			kOff := t*nKVHeads*headDim + kvHead*headDim
			headScores[t] = dot(q[qOff:qOff+headDim], k[kOff:kOff+headDim]) * scale
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

// AttentionScores вычисляет scaled dot-product attention для одной позиции
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
