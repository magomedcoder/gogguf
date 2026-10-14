package deepseek

import "github.com/magomedcoder/gogguf/pkg/model/mempool"

// x/h/q/k/v/attn/gate/up are single-token views (first row) of the nBatch buffers bx/bh/...
type scratch struct {
	// batch buffers [nBatch*dim] (prefill chunk; dense models only)
	bx    []float32
	bh    []float32
	bq    []float32
	bk    []float32
	bv    []float32
	battn []float32
	bgate []float32
	bup   []float32

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
	tmp    []float32 // embd: expert down / shexp out
	logits []float32
}

func newScratch(cfg Config, nBatch int) scratch {
	if nBatch < 1 {
		nBatch = 1
	}

	qDim := cfg.NumHeads * cfg.HeadDim
	kvDim := cfg.NumKVHeads * cfg.HeadDim
	embd := cfg.EmbeddingDim
	ffn := cfg.maxFFN()
	nExp := max(cfg.ExpertCount, 1)

	need := nBatch*(2*embd+2*qDim+2*kvDim+2*ffn) + 2*embd + cfg.ContextLength + nExp + 2*cfg.VocabSize
	a := mempool.NewArena(need)

	bx := a.Alloc(nBatch * embd)
	bh := a.Alloc(nBatch * embd)
	bq := a.Alloc(nBatch * qDim)
	bk := a.Alloc(nBatch * kvDim)
	bv := a.Alloc(nBatch * kvDim)
	battn := a.Alloc(nBatch * qDim)
	bgate := a.Alloc(nBatch * ffn)
	bup := a.Alloc(nBatch * ffn)

	return scratch{
		bx:     bx,
		bh:     bh,
		bq:     bq,
		bk:     bk,
		bv:     bv,
		battn:  battn,
		bgate:  bgate,
		bup:    bup,
		x:      bx[:embd],
		h:      bh[:embd],
		q:      bq[:qDim],
		k:      bk[:kvDim],
		v:      bv[:kvDim],
		attn:   battn[:qDim],
		scores: a.Alloc(cfg.ContextLength),
		gate:   bgate[:ffn],
		up:     bup[:ffn],
		router: a.Alloc(nExp),
		moeAcc: a.Alloc(embd),
		tmp:    a.Alloc(embd),
		logits: a.Alloc(cfg.VocabSize),
	}
}
