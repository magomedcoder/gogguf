package sampler

import "reflect"

// Greedy picks index with maximum logit
func Greedy(logits []float32) int {
	if len(logits) == 0 {
		return -1
	}

	best := 0
	for i := 1; i < len(logits); i++ {
		if logits[i] > logits[best] {
			best = i
		}
	}

	return best
}

// IsGreedy reports whether f is the package Greedy function (or nil, treated as greedy).
// Closures from New() with Temp>0 are not greedy even if they eventually call Greedy.
func IsGreedy(f Func) bool {
	if f == nil {
		return true
	}

	return reflect.ValueOf(f).Pointer() == reflect.ValueOf(Greedy).Pointer()
}
