package deepseek2

import "github.com/magomedcoder/gogguf/pkg/model/mempool"

type scratch struct {
	x      []float32
	h      []float32
	qFull  []float32 // nHeads*(qkNope+rope) before absorb
	qAbs   []float32 // nHeads*(kvLora+rope)
	qRope  []float32 // nHeads*(qkNope+rope): q_nope raw + RoPE'd q_pe (device MLA input)
	kvPe   []float32 // kvLora + rope
	kCache []float32 // K for one token (absorbed MLA)
	vCache []float32 // V for one token (absorbed MLA / kv_lora)
	kMHA   []float32 // legacy: nHeads*(qkNope+rope) for one token
	vMHA   []float32 // legacy: nHeads*vHead for one token
	pePack []float32 // nHeads*rope: packed q_pe for device YaRN RoPE
	attn   []float32 // nHeads*vHead
	scores []float32
	gate   []float32
	up     []float32
	router []float32
	moeAcc []float32
	tmp    []float32
	qLora  []float32
	qPe    []float32 // RoPE scratch for one Q head
	kvExp  []float32 // legacy: per-head kv_b expand (qkNope+vHead)
	logits []float32
}

func newScratch(cfg Config) scratch {
	headK := cfg.QKNopeDim + cfg.RopeDim
	qFull := cfg.NumHeads * headK
	qAbs := cfg.NumHeads * cfg.qkDim()
	attn := cfg.NumHeads * cfg.VHeadDim
	pePack := cfg.NumHeads * cfg.RopeDim
	kMHA := qFull
	vMHA := attn
	kvExp := cfg.QKNopeDim + cfg.VHeadDim
	ffn := cfg.maxFFN()
	nExp := max(cfg.ExpertCount, 1)

	qLora := max(cfg.QLoraRank, 1)

	need := 4*cfg.EmbeddingDim + 2*qFull + qAbs + cfg.qkDim() + cfg.KVLoraRank + kMHA + vMHA + pePack + attn + cfg.ContextLength + 2*ffn + nExp + qLora + cfg.VocabSize + cfg.qkDim() + kvExp
	a := mempool.NewArena(need)
	return scratch{
		x:      a.Alloc(cfg.EmbeddingDim),
		h:      a.Alloc(cfg.EmbeddingDim),
		qFull:  a.Alloc(qFull),
		qAbs:   a.Alloc(qAbs),
		qRope:  a.Alloc(qFull),
		kvPe:   a.Alloc(cfg.qkDim()),
		kCache: a.Alloc(cfg.qkDim()),
		vCache: a.Alloc(cfg.KVLoraRank),
		kMHA:   a.Alloc(kMHA),
		vMHA:   a.Alloc(vMHA),
		pePack: a.Alloc(pePack),
		attn:   a.Alloc(attn),
		scores: a.Alloc(cfg.ContextLength),
		gate:   a.Alloc(ffn),
		up:     a.Alloc(ffn),
		router: a.Alloc(nExp),
		moeAcc: a.Alloc(cfg.EmbeddingDim),
		tmp:    a.Alloc(cfg.EmbeddingDim),
		qLora:  a.Alloc(qLora),
		qPe:    a.Alloc(cfg.RopeDim),
		kvExp:  a.Alloc(kvExp),
		logits: a.Alloc(cfg.VocabSize),
	}
}
