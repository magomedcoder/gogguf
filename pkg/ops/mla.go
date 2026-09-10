package ops

import (
	"fmt"
	"math"
)

// AttentionMLAAbsorbedInto - поглощённая MLA (DeepSeek-V2/V3):
//
//	размер Q/K = kvLora + ropeDim (MQA: одна KV-голова);
//	размер V = kvLora; после softmax wv_b поднимает V до vHead на голову.
//
// q: [nHeads*(kvLora+ropeDim)]
// k: [seqLen*(kvLora+ropeDim)]  (1 KV-голова)
// v: [seqLen*kvLora]
// wvB: [nHeads][vHead][kvLora] в раскладке ggml {kvLora, vHead, nHeads}
// dst: [nHeads*vHead]
func AttentionMLAAbsorbedInto(dst, q, k, v, scores, wvB []float32, seqLen, nHeads, kvLora, ropeDim, vHead int, scale float32) error {
	qkDim := kvLora + ropeDim
	if nHeads <= 0 || kvLora <= 0 || ropeDim < 0 || vHead <= 0 || seqLen <= 0 {
		return fmt.Errorf("ops: MLA sizes invalid")
	}

	if len(dst) < nHeads*vHead {
		return fmt.Errorf("ops: MLA dst слишком короткий")
	}

	if len(q) < nHeads*qkDim || len(k) < seqLen*qkDim || len(v) < seqLen*kvLora {
		return fmt.Errorf("ops: MLA q/k/v слишком короткие")
	}

	if len(scores) < seqLen {
		return fmt.Errorf("ops: MLA scores слишком короткий")
	}

	if len(wvB) < nHeads*vHead*kvLora {
		return fmt.Errorf("ops: MLA wv_b слишком короткий")
	}

	if scale == 0 {
		scale = float32(1 / math.Sqrt(float64(qkDim)))
	}

	headScores := scores[:seqLen]
	ctx := make([]float32, kvLora)

	for h := range nHeads {
		qOff := h * qkDim
		for t := range seqLen {
			kOff := t * qkDim
			headScores[t] = dot(q[qOff:qOff+qkDim], k[kOff:kOff+qkDim]) * scale
		}

		SoftmaxInPlace(headScores)

		for i := range kvLora {
			ctx[i] = dotStride(headScores, v, i, kvLora, seqLen)
		}

		// wv_b[h]: out[v] = Σ_k W[k + kvLora*v + kvLora*vHead*h] * ctx[k]
		base := h * kvLora * vHead
		outOff := h * vHead
		for vh := range vHead {
			var s float32
			wOff := base + vh*kvLora
			for i := range kvLora {
				s += wvB[wOff+i] * ctx[i]
			}
			dst[outOff+vh] = s
		}
	}

	return nil
}

// MatMulColMajorInto: out[m] = Σ_k W[k + kDim*m] * vec[k] (раскладка ggml_mul_mat для wk_b: {kDim, mDim})
func MatMulColMajorInto(w []float32, kDim, mDim int, vec, out []float32) error {
	if len(vec) < kDim || len(out) < mDim || len(w) < kDim*mDim {
		return fmt.Errorf("ops: MatMulColMajor размеры")
	}

	for m := range mDim {
		var s float32
		off := m * kDim
		for k := range kDim {
			s += w[off+k] * vec[k]
		}
		out[m] = s
	}

	return nil
}
