package debug

import (
	"cmp"
	"fmt"
	"math"
)

// AssertClose reports whether |got-want| <= tol for float values
func AssertClose[T ~float32 | ~float64](got, want T, tol T) error {
	d := got - want
	if d < 0 {
		d = -d
	}

	if d > tol {
		return fmt.Errorf("got %v want %v (tol %v)", got, want, tol)
	}

	return nil
}

// TopKIndices returns indices of the k largest values in xs (descending)
func TopKIndices[T cmp.Ordered](xs []T, k int) []int {
	if k <= 0 || len(xs) == 0 {
		return nil
	}

	if k > len(xs) {
		k = len(xs)
	}

	type pair struct {
		i int
		v T
	}
	items := make([]pair, len(xs))
	for i, v := range xs {
		items[i] = pair{i: i, v: v}
	}

	// Partial selection: keep top-k via simple sort of copy (debug/tools only).
	for i := 0; i < k; i++ {
		best := i
		for j := i + 1; j < len(items); j++ {
			if items[j].v > items[best].v {
				best = j
			}
		}
		items[i], items[best] = items[best], items[i]
	}

	out := make([]int, k)
	for i := 0; i < k; i++ {
		out[i] = items[i].i
	}

	return out
}

// RelErr returns |got-want|/max(|want|, eps).
func RelErr(got, want, eps float64) float64 {
	den := math.Abs(want)
	if den < eps {
		den = eps
	}

	return math.Abs(got-want) / den
}
