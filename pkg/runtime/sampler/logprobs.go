package sampler

import "math"

// TopToken is one vocabulary entry with its log-probability under softmax
type TopToken struct {
	ID      int
	Logprob float64
}

// SoftmaxLogprob returns ln(p[id]) under temperature-1 softmax of logits
func SoftmaxLogprob(logits []float32, id int) float64 {
	if id < 0 || id >= len(logits) {
		return math.Inf(-1)
	}

	max := maxLogit(logits)
	var sum float64
	for _, v := range logits {
		sum += math.Exp(float64(v - max))
	}

	if sum <= 0 {
		return math.Inf(-1)
	}

	return float64(logits[id]-max) - math.Log(sum)
}

// TopLogprobs returns up to k tokens with highest softmax probability (descending)
func TopLogprobs(logits []float32, k int) []TopToken {
	if k < 1 || len(logits) == 0 {
		return nil
	}

	if k > len(logits) {
		k = len(logits)
	}

	max := maxLogit(logits)
	var sum float64
	for _, v := range logits {
		sum += math.Exp(float64(v - max))
	}

	if sum <= 0 {
		return nil
	}
	logSum := math.Log(sum)

	out := make([]TopToken, 0, k)
	for id, v := range logits {
		lp := float64(v-max) - logSum
		if len(out) < k {
			out = append(out, TopToken{
				ID:      id,
				Logprob: lp,
			})
			insertSortDesc(out)

			continue
		}

		if lp <= out[k-1].Logprob {
			continue
		}

		out[k-1] = TopToken{
			ID:      id,
			Logprob: lp,
		}
		insertSortDesc(out)
	}
	return out
}

func insertSortDesc(a []TopToken) {
	for i := len(a) - 1; i > 0; i-- {
		if a[i].Logprob <= a[i-1].Logprob {
			return
		}

		a[i], a[i-1] = a[i-1], a[i]
	}
}
