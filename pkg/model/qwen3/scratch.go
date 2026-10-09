package qwen3

import "github.com/magomedcoder/gogguf/pkg/mempool"

// scratch - reusable forward pass buffers from one Arena
type scratch struct {
	x      []float32
	h      []float32
	q      []float32
	k      []float32
	v      []float32
	attn   []float32
	scores []float32
	gate   []float32
	up     []float32
	router []float32
	moeAcc []float32
	tmp    []float32
	logits []float32
	out    []float32
}

func newScratch(cfg Config, nBatch int) scratch {
	if nBatch < 1 {
		nBatch = 1
	}

	qDim := cfg.NumHeads * cfg.HeadDim
	kvDim := cfg.NumKVHeads * cfg.HeadDim
	ffn := cfg.maxFFN()
	nExp := max(cfg.ExpertCount, 1)
	embd := cfg.EmbeddingDim

	need := nBatch*(4*embd+2*qDim+2*kvDim+2*ffn) + cfg.ContextLength + nExp + embd + 2*cfg.VocabSize
	a := mempool.NewArena(need)

	return scratch{
		x:      a.Alloc(nBatch * embd),
		h:      a.Alloc(nBatch * embd),
		q:      a.Alloc(nBatch * qDim),
		k:      a.Alloc(nBatch * kvDim),
		v:      a.Alloc(nBatch * kvDim),
		attn:   a.Alloc(nBatch * qDim),
		scores: a.Alloc(cfg.ContextLength),
		gate:   a.Alloc(nBatch * ffn),
		up:     a.Alloc(nBatch * ffn),
		router: a.Alloc(nExp),
		moeAcc: a.Alloc(embd),
		tmp:    a.Alloc(embd),
		logits: a.Alloc(cfg.VocabSize),
		out:    a.Alloc(cfg.VocabSize),
	}
}
