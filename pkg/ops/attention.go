package ops

import (
	"fmt"
	"math"
	"sync"
)

var headScoresPool = sync.Pool{New: func() any { return new([]float32) }}

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
	useCap := softcap != 0
	capF := float64(softcap)

	headScores := scores[:seqLen]
	if nHeads < 4 || seqLen*headDim < 256 {
		for h := range nHeads {
			attentionOneHead(dst, q, k, v, headScores, h, groupSize, seqLen, nKVHeads, headDim, scale, useCap, capF)
		}

		return nil
	}

	parallelForWork(nHeads, 4, func(start, end int) {
		scratchPtr := headScoresPool.Get().(*[]float32)
		local := *scratchPtr
		if cap(local) < seqLen {
			local = make([]float32, seqLen)
			*scratchPtr = local
		} else {
			local = local[:seqLen]
		}
		defer headScoresPool.Put(scratchPtr)

		for h := start; h < end; h++ {
			attentionOneHead(dst, q, k, v, local, h, groupSize, seqLen, nKVHeads, headDim, scale, useCap, capF)
		}
	})

	return nil
}

func attentionOneHead(dst, q, k, v, headScores []float32, h, groupSize, seqLen, nKVHeads, headDim int, scale float32, useCap bool, capF float64) {
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
