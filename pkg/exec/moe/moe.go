package moe

import (
	"container/heap"
	"slices"

	"github.com/magomedcoder/gogguf/pkg/compute"
)

// TopKSoftmax selects top-k experts after softmax on router logits.
// logits are overwritten with softmax probabilities.
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

	probs := logits
	compute.SoftmaxInPlace(probs)

	idxs = topKIndices(probs, k)

	weights = make([]float32, k)
	var sum float32
	for i, id := range idxs {
		weights[i] = probs[id]
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

func topKIndices(probs []float32, k int) []int {
	n := len(probs)
	if k >= n {
		idxs := make([]int, n)
		for i := range idxs {
			idxs[i] = i
		}
		sortExpertIndices(idxs, probs)
		return idxs
	}

	// Full sort when k is a large fraction of n; otherwise partial heap select.
	if k*4 >= n {
		idxs := make([]int, n)
		for i := range idxs {
			idxs[i] = i
		}
		sortExpertIndices(idxs, probs)
		return idxs[:k]
	}

	h := expertRankHeap{probs: probs}
	for i := range n {
		if len(h.ids) < k {
			heap.Push(&h, i)
			continue
		}

		if compareExpertRank(probs, i, h.ids[0]) < 0 {
			h.ids[0] = i
			heap.Fix(&h, 0)
		}
	}

	idxs := make([]int, len(h.ids))
	for i := len(h.ids) - 1; i >= 0; i-- {
		idxs[i] = heap.Pop(&h).(int)
	}
	sortExpertIndices(idxs, probs)

	return idxs
}

func sortExpertIndices(idxs []int, probs []float32) {
	slices.SortFunc(idxs, func(a, b int) int {
		return compareExpertRank(probs, a, b)
	})
}

// Higher prob first; ties broken by lower expert index.
func compareExpertRank(probs []float32, a, b int) int {
	if probs[a] > probs[b] {
		return -1
	}

	if probs[a] < probs[b] {
		return 1
	}

	return a - b
}

type expertRankHeap struct {
	probs []float32
	ids   []int
}

func (h *expertRankHeap) Len() int {
	return len(h.ids)
}

// Root = worst kept expert (lowest rank).
func (h *expertRankHeap) Less(i, j int) bool {
	return compareExpertRank(h.probs, h.ids[i], h.ids[j]) > 0
}

func (h *expertRankHeap) Swap(i, j int) {
	h.ids[i], h.ids[j] = h.ids[j], h.ids[i]
}

func (h *expertRankHeap) Push(x any) {
	h.ids = append(h.ids, x.(int))
}

func (h *expertRankHeap) Pop() any {
	old := h.ids
	n := len(old)
	x := old[n-1]
	h.ids = old[:n-1]
	return x
}
