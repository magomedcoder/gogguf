package qwen3

// scratch - переиспользуемые буферы forward pass (без аллокаций на слой)
type scratch struct {
	x      []float32 // embedding dim - текущий hidden state
	h      []float32 // embedding dim - временный
	q      []float32 // numHeads * headDim
	k      []float32 // numKVHeads * headDim
	v      []float32 // numKVHeads * headDim
	attn   []float32 // numHeads * headDim
	scores []float32 // context_length - softmax scores в attention
	gate   []float32 // ffn hidden
	up     []float32 // ffn hidden
	logits []float32 // vocab
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
