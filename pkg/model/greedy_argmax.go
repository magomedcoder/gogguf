package model

// GreedyArgmaxModel optionally computes lm_head argmax on device (DtoH only idx+val).
// Used by decode when sampling is pure greedy (no penalties / logprobs).
type GreedyArgmaxModel interface {
	SetGreedyArgmax(on bool)

	TakeGreedyArgmax() (idx int, ok bool)
}
