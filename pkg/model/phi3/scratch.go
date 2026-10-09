package phi3

import "github.com/magomedcoder/gogguf/pkg/mempool"

// scratch - reusable forward pass buffers from one Arena
type scratch struct {
	x      []float32
	h      []float32
	qkv    []float32
	q      []float32
	k      []float32
	v      []float32
	attn   []float32
	scores []float32
	gateUp []float32
	gate   []float32
	up     []float32
	logits []float32
	out    []float32
}

func newScratch(cfg Config) scratch {
	qDim := cfg.NumHeads * cfg.HeadDim
	kvDim := cfg.NumKVHeads * cfg.HeadDim
	need := 2*cfg.EmbeddingDim + (qDim + 2*kvDim) + qDim + 2*kvDim + qDim + cfg.ContextLength + 2*cfg.FFNHidden + 2*cfg.FFNHidden + 2*cfg.VocabSize
	a := mempool.NewArena(need)

	return scratch{
		x:      a.Alloc(cfg.EmbeddingDim),
		h:      a.Alloc(cfg.EmbeddingDim),
		qkv:    a.Alloc(qDim + 2*kvDim),
		q:      a.Alloc(qDim),
		k:      a.Alloc(kvDim),
		v:      a.Alloc(kvDim),
		attn:   a.Alloc(qDim),
		scores: a.Alloc(cfg.ContextLength),
		gateUp: a.Alloc(2 * cfg.FFNHidden),
		gate:   a.Alloc(cfg.FFNHidden),
		up:     a.Alloc(cfg.FFNHidden),
		logits: a.Alloc(cfg.VocabSize),
		out:    a.Alloc(cfg.VocabSize),
	}
}
