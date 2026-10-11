package attention

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
