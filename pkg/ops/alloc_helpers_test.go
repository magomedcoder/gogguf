package ops

// Test-only allocating helpers (not part of the public ops API).

func testAttentionScores(q, k, v []float32, seqLen, nHeads, nKVHeads, headDim int) ([]float32, error) {
	out := make([]float32, nHeads*headDim)
	scores := make([]float32, seqLen)
	if err := AttentionScoresInto(out, q, k, v, scores, seqLen, nHeads, nKVHeads, headDim); err != nil {
		return nil, err
	}

	return out, nil
}

func testSoftmax(x []float32) []float32 {
	out := make([]float32, len(x))
	copy(out, x)
	SoftmaxInPlace(out)

	return out
}

func testAdd(a, b []float32) []float32 {
	out := make([]float32, len(a))
	for i := range a {
		out[i] = a[i] + b[i]
	}

	return out
}

func testScale(x []float32, s float32) []float32 {
	out := make([]float32, len(x))
	for i, v := range x {
		out[i] = v * s
	}

	return out
}

func testSwiGLU(gate, up []float32) []float32 {
	out := make([]float32, len(gate))
	copy(out, gate)
	for i := range out {
		out[i] = SiLU(out[i])
	}

	vecMulInPlace(out, up)

	return out
}
