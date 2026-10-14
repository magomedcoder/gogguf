package phi3

import "github.com/magomedcoder/gogguf/pkg/model/mempool"

// scratch - reusable forward pass buffers from one Arena.
// x/h/qkv/q/k/v/attn/gateUp/gate/up are single-token views (first row) of the nBatch buffers bx/bh/...
type scratch struct {
	// batch buffers [nBatch*dim] (prefill chunk)
	bx      []float32
	bh      []float32
	bqkv    []float32
	bq      []float32
	bk      []float32
	bv      []float32
	battn   []float32
	bgateUp []float32
	bgate   []float32
	bup     []float32

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
}

func newScratch(cfg Config, nBatch int) scratch {
	if nBatch < 1 {
		nBatch = 1
	}

	qDim := cfg.NumHeads * cfg.HeadDim
	kvDim := cfg.NumKVHeads * cfg.HeadDim
	embd := cfg.EmbeddingDim
	ffn := cfg.FFNHidden
	qkvDim := qDim + 2*kvDim
	need := nBatch*(2*embd+qkvDim+qDim+2*kvDim+qDim+4*ffn) + cfg.ContextLength + 2*cfg.VocabSize
	a := mempool.NewArena(need)

	bx := a.Alloc(nBatch * embd)
	bh := a.Alloc(nBatch * embd)
	bqkv := a.Alloc(nBatch * qkvDim)
	bq := a.Alloc(nBatch * qDim)
	bk := a.Alloc(nBatch * kvDim)
	bv := a.Alloc(nBatch * kvDim)
	battn := a.Alloc(nBatch * qDim)
	bgateUp := a.Alloc(nBatch * 2 * ffn)
	bgate := a.Alloc(nBatch * ffn)
	bup := a.Alloc(nBatch * ffn)

	return scratch{
		bx:      bx,
		bh:      bh,
		bqkv:    bqkv,
		bq:      bq,
		bk:      bk,
		bv:      bv,
		battn:   battn,
		bgateUp: bgateUp,
		bgate:   bgate,
		bup:     bup,
		x:       bx[:embd],
		h:       bh[:embd],
		qkv:     bqkv[:qkvDim],
		q:       bq[:qDim],
		k:       bk[:kvDim],
		v:       bv[:kvDim],
		attn:    battn[:qDim],
		scores:  a.Alloc(cfg.ContextLength),
		gateUp:  bgateUp[:2*ffn],
		gate:    bgate[:ffn],
		up:      bup[:ffn],
		logits:  a.Alloc(cfg.VocabSize),
	}
}
