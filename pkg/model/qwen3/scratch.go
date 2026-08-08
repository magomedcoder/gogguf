package qwen3

import "github.com/magomedcoder/gogguf/pkg/mempool"

// scratch - переиспользуемые буферы forward pass из одного Arena
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
	logits []float32
	out    []float32 // копия logits для возврата из Forward
}

func newScratch(cfg Config) scratch {
	qDim := cfg.NumHeads * cfg.HeadDim
	kvDim := cfg.NumKVHeads * cfg.HeadDim
	need := 2*cfg.EmbeddingDim + 2*qDim + 2*kvDim + cfg.ContextLength + 2*cfg.FFNHidden + 2*cfg.VocabSize
	a := mempool.NewArena(need)

	return scratch{
		x:      a.Alloc(cfg.EmbeddingDim),
		h:      a.Alloc(cfg.EmbeddingDim),
		q:      a.Alloc(qDim),
		k:      a.Alloc(kvDim),
		v:      a.Alloc(kvDim),
		attn:   a.Alloc(qDim),
		scores: a.Alloc(cfg.ContextLength),
		gate:   a.Alloc(cfg.FFNHidden),
		up:     a.Alloc(cfg.FFNHidden),
		logits: a.Alloc(cfg.VocabSize),
		out:    a.Alloc(cfg.VocabSize),
	}
}
