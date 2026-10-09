package moe

import (
	"math"
	"slices"
)

// TopKSoftmax selects top-k experts after softmax on router logits.
// normWeights: true - renorm top-k weights (Mixtral); false - raw softmax probs (DeepSeek V1).
// wScale scales weights (expert_weights_scale); 0 and 1 mean no scaling.
func TopKSoftmax(logits []float32, k int, normWeights bool, wScale float32) (idxs []int, weights []float32) {
	n := len(logits)
	if n == 0 || k <= 0 {
		return nil, nil
	}

	if k > n {
		k = n
	}

	probs := append([]float32(nil), logits...)
	softmaxInPlace(probs)

	type pair struct {
		i int
		p float32
	}

	ranked := make([]pair, n)
	for i, p := range probs {
		ranked[i] = pair{i: i, p: p}
	}

	slices.SortFunc(ranked, func(a, b pair) int {
		if a.p > b.p {
			return -1
		}

		if a.p < b.p {
			return 1
		}

		return a.i - b.i
	})

	idxs = make([]int, k)
	weights = make([]float32, k)
	var sum float32
	for i := 0; i < k; i++ {
		idxs[i] = ranked[i].i
		weights[i] = ranked[i].p
		sum += weights[i]
	}

	if normWeights && sum > 0 {
		inv := 1 / sum
		for i := range weights {
			weights[i] *= inv
		}
	}

	if wScale != 0 && wScale != 1 {
		for i := range weights {
			weights[i] *= wScale
		}
	}

	return idxs, weights
}

func softmaxInPlace(x []float32) {
	if len(x) == 0 {
		return
	}

	maxV := x[0]
	for _, v := range x[1:] {
		if v > maxV {
			maxV = v
		}
	}

	var sum float64
	for i, v := range x {
		e := math.Exp(float64(v - maxV))
		x[i] = float32(e)
		sum += e
	}

	inv := float32(1 / sum)
	for i := range x {
		x[i] *= inv
	}
}
