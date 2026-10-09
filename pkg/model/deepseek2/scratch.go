package deepseek2

import "github.com/magomedcoder/gogguf/pkg/mempool"

type scratch struct {
	x      []float32
	h      []float32
	qFull  []float32 // nHeads*(qkNope+rope) before absorb
	qAbs   []float32 // nHeads*(kvLora+rope)
	kvPe   []float32 // kvLora + rope
	kCache []float32 // K for one token
	vCache []float32 // V for one token
	attn   []float32 // nHeads*vHead
	scores []float32
	gate   []float32
	up     []float32
	router []float32
	moeAcc []float32
	tmp    []float32
	qLora  []float32
	logits []float32
	out    []float32
}

func newScratch(cfg Config) scratch {
	qFull := cfg.NumHeads * (cfg.QKNopeDim + cfg.RopeDim)
	qAbs := cfg.NumHeads * cfg.qkDim()
	attn := cfg.NumHeads * cfg.VHeadDim
	ffn := cfg.maxFFN()
	nExp := max(cfg.ExpertCount, 1)

	qLora := max(cfg.QLoraRank, 1)

	need := 4*cfg.EmbeddingDim + qFull + qAbs + cfg.qkDim() + cfg.KVLoraRank + attn + cfg.ContextLength + 2*ffn + nExp + qLora + 2*cfg.VocabSize + cfg.qkDim()
	a := mempool.NewArena(need)
	return scratch{
		x:      a.Alloc(cfg.EmbeddingDim),
		h:      a.Alloc(cfg.EmbeddingDim),
		qFull:  a.Alloc(qFull),
		qAbs:   a.Alloc(qAbs),
		kvPe:   a.Alloc(cfg.qkDim()),
		kCache: a.Alloc(cfg.qkDim()),
		vCache: a.Alloc(cfg.KVLoraRank),
		attn:   a.Alloc(attn),
		scores: a.Alloc(cfg.ContextLength),
		gate:   a.Alloc(ffn),
		up:     a.Alloc(ffn),
		router: a.Alloc(nExp),
		moeAcc: a.Alloc(cfg.EmbeddingDim),
		tmp:    a.Alloc(cfg.EmbeddingDim),
		qLora:  a.Alloc(qLora),
		logits: a.Alloc(cfg.VocabSize),
		out:    a.Alloc(cfg.VocabSize),
	}
}
