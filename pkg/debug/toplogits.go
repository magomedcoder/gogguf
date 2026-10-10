package debug

import "container/heap"

// Logit is a token id and logit value.
type Logit struct {
	ID    int     `json:"id"`
	Logit float32 `json:"logit"`
}

type logitMinHeap []Logit

func (h logitMinHeap) Len() int {
	return len(h)
}
func (h logitMinHeap) Less(i, j int) bool {
	return h[i].Logit < h[j].Logit
}
func (h logitMinHeap) Swap(i, j int) {
	h[i], h[j] = h[j], h[i]
}
func (h *logitMinHeap) Push(x any) {
	*h = append(*h, x.(Logit))
}
func (h *logitMinHeap) Pop() any {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]

	return x
}

// TopLogits returns top-N logits sorted by descending value (O(n log k) heap).
func TopLogits(logits []float32, n int) []Logit {
	if n <= 0 || len(logits) == 0 {
		return nil
	}

	if n > len(logits) {
		n = len(logits)
	}

	h := make(logitMinHeap, 0, n)
	for id, v := range logits {
		if len(h) < n {
			heap.Push(&h, Logit{ID: id, Logit: v})
			continue
		}
		if v > h[0].Logit {
			h[0] = Logit{ID: id, Logit: v}
			heap.Fix(&h, 0)
		}
	}

	out := make([]Logit, len(h))
	for i := len(h) - 1; i >= 0; i-- {
		out[i] = heap.Pop(&h).(Logit)
	}
	return out
}
