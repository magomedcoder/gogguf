package llama

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
}

func newScratch(cfg Config) scratch {
	qDim := cfg.NumHeads * cfg.HeadDim
	kvDim := cfg.NumKVHeads * cfg.HeadDim
	return scratch{
		x:      make([]float32, cfg.EmbeddingDim),
		h:      make([]float32, cfg.EmbeddingDim),
		q:      make([]float32, qDim),
		k:      make([]float32, kvDim),
		v:      make([]float32, kvDim),
		attn:   make([]float32, qDim),
		scores: make([]float32, cfg.ContextLength),
		gate:   make([]float32, cfg.FFNHidden),
		up:     make([]float32, cfg.FFNHidden),
		logits: make([]float32, cfg.VocabSize),
	}
}
