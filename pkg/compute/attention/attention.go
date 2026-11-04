package attention

import (
	"fmt"
	"github.com/magomedcoder/gogguf/pkg/compute/vec"
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
		return fmt.Errorf("compute: dst too short")
	}

	if len(scores) < seqLen {
		return fmt.Errorf("compute: scores too short")
	}

	if nHeads%nKVHeads != 0 {
		return fmt.Errorf("compute: nHeads=%d not divisible by nKVHeads=%d", nHeads, nKVHeads)
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

	vec.ParallelForWork(nHeads, 4, func(start, end int) {
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
	attentionOneHeadHV(dst, q, k, v, headScores, h, groupSize, seqLen, nKVHeads, headDim, headDim, scale, useCap, capF)
}

// AttentionScoresHVInto is MHA with separate key/value head dims and explicit scale (DeepSeek2 legacy attn_kv_b: kDim = qk_nope+rope, vDim = v_head, scale = YaRN kq_scale).
// q: [nHeads*kDim], k: [seqLen*nKVHeads*kDim], v: [seqLen*nKVHeads*vDim], dst: [nHeads*vDim].
// scale == 0 -> 1/sqrt(kDim).
func AttentionScoresHVInto(dst, q, k, v, scores []float32, seqLen, nHeads, nKVHeads, kDim, vDim int, scale float32) error {
	if nHeads <= 0 || nKVHeads <= 0 || kDim <= 0 || vDim <= 0 || seqLen <= 0 {
		return fmt.Errorf("compute: AttentionScoresHVInto: invalid dimensions")
	}

	if nHeads%nKVHeads != 0 {
		return fmt.Errorf("compute: nHeads=%d not divisible by nKVHeads=%d", nHeads, nKVHeads)
	}

	if len(dst) < nHeads*vDim {
		return fmt.Errorf("compute: dst too short")
	}

	if len(scores) < seqLen {
		return fmt.Errorf("compute: scores too short")
	}

	if len(q) < nHeads*kDim || len(k) < seqLen*nKVHeads*kDim || len(v) < seqLen*nKVHeads*vDim {
		return fmt.Errorf("compute: q/k/v too short")
	}

	if scale == 0 {
		scale = float32(1 / math.Sqrt(float64(kDim)))
	}

	groupSize := nHeads / nKVHeads
	headScores := scores[:seqLen]
	if nHeads < 4 || seqLen*kDim < 256 {
		for h := range nHeads {
			attentionOneHeadHV(dst, q, k, v, headScores, h, groupSize, seqLen, nKVHeads, kDim, vDim, scale, false, 0)
		}

		return nil
	}

	vec.ParallelForWork(nHeads, 4, func(start, end int) {
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
			attentionOneHeadHV(dst, q, k, v, local, h, groupSize, seqLen, nKVHeads, kDim, vDim, scale, false, 0)
		}
	})

	return nil
}

func attentionOneHeadHV(dst, q, k, v, headScores []float32, h, groupSize, seqLen, nKVHeads, kDim, vDim int, scale float32, useCap bool, capF float64) {
	kvHead := h / groupSize
	qOff := h * kDim

	for t := range seqLen {
		kOff := t*nKVHeads*kDim + kvHead*kDim
		s := float64(vec.Dot(q[qOff:qOff+kDim], k[kOff:kOff+kDim]) * scale)
		if useCap {
			s = capF * math.Tanh(s/capF)
		}
		headScores[t] = float32(s)
	}

	SoftmaxInPlace(headScores)

	outOff := h * vDim
	vStride := nKVHeads * vDim
	vBase := kvHead * vDim
	for i := range vDim {
		dst[outOff+i] = vec.DotStride(headScores, v, vBase+i, vStride, seqLen)
	}
}
