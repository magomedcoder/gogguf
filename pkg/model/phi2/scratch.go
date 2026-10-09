package phi2

import "github.com/magomedcoder/gogguf/pkg/mempool"

// scratch - reusable forward pass buffers
type scratch struct {
	x       []float32
	h       []float32
	attnOut []float32
	qkv     []float32
	q       []float32
	k       []float32
	v       []float32
	attn    []float32
	scores  []float32
	up      []float32
	logits  []float32
	out     []float32
}

func newScratch(cfg Config) scratch {
	qDim := cfg.NumHeads * cfg.HeadDim
	kvDim := cfg.NumKVHeads * cfg.HeadDim
	need := 3*cfg.EmbeddingDim + (qDim + 2*kvDim) + qDim + 2*kvDim + qDim + cfg.ContextLength + cfg.FFNHidden + 2*cfg.VocabSize
	a := mempool.NewArena(need)

	return scratch{
		x:       a.Alloc(cfg.EmbeddingDim),
		h:       a.Alloc(cfg.EmbeddingDim),
		attnOut: a.Alloc(cfg.EmbeddingDim),
		qkv:     a.Alloc(qDim + 2*kvDim),
		q:       a.Alloc(qDim),
		k:       a.Alloc(kvDim),
		v:       a.Alloc(kvDim),
		attn:    a.Alloc(qDim),
		scores:  a.Alloc(cfg.ContextLength),
		up:      a.Alloc(cfg.FFNHidden),
		logits:  a.Alloc(cfg.VocabSize),
		out:     a.Alloc(cfg.VocabSize),
	}
}
