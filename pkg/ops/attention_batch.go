package ops

import (
	"fmt"
	"math"
)

// AttentionScoresBatchCausalInto - causal attention for a batch of requests (prefill n_batch).
//
// q, dst: [batch * nHeads * headDim]
// k, v:   [(pastLen+batch) * nKVHeads * headDim] - already appended with new tokens scores: buffer length >= pastLen+batch (reused)
//
// Query i (position pastLen+i) sees keys 0..pastLen+i inclusive.
func AttentionScoresBatchCausalInto(dst, q, k, v, scores []float32, pastLen, batch, nHeads, nKVHeads, headDim int) error {
	if batch < 1 {
		return fmt.Errorf("ops: batch=%d", batch)
	}

	qDim := nHeads * headDim
	if len(dst) < batch*qDim || len(q) < batch*qDim {
		return fmt.Errorf("ops: q/dst too short")
	}

	total := pastLen + batch
	kvDim := nKVHeads * headDim
	if len(k) < total*kvDim || len(v) < total*kvDim {
		return fmt.Errorf("ops: k/v too short for seq=%d", total)
	}

	if len(scores) < total {
		return fmt.Errorf("ops: scores too short")
	}

	if nHeads%nKVHeads != 0 {
		return fmt.Errorf("ops: nHeads=%d not divisible by nKVHeads=%d", nHeads, nKVHeads)
	}

	groupSize := nHeads / nKVHeads
	scale := float32(1 / math.Sqrt(float64(headDim)))

	for b := range batch {
		causalEnd := pastLen + b + 1 // number of keys visible to query b
		headScores := scores[:causalEnd]
		qBase := b * qDim
		dstBase := b * qDim

		for h := range nHeads {
			kvHead := h / groupSize
			qOff := qBase + h*headDim

			for t := range causalEnd {
				kOff := t*kvDim + kvHead*headDim
				headScores[t] = dot(q[qOff:qOff+headDim], k[kOff:kOff+headDim]) * scale
			}

			SoftmaxInPlace(headScores)

			outOff := dstBase + h*headDim
			vBase := kvHead * headDim
			for i := range headDim {
				dst[outOff+i] = dotStride(headScores, v, vBase+i, kvDim, causalEnd)
			}
		}
	}

	return nil
}
