package mempool

// KV - fixed CPU KV-cache: maxSeq*kvDim buffers, append-free writes.
// Per-layer token counter (like append); Len/Advance are separate
type KV struct {
	layers []kvLayer
	kvDim  int
	maxSeq int
	length int
}

type kvLayer struct {
	k []float32
	v []float32
	n int // number of written tokens
}

// NewKV allocates KV for numLayers layers.
// arena may be nil - then separate make
func NewKV(numLayers, maxSeq, kvDim int, arena *Arena) *KV {
	if numLayers < 0 {
		numLayers = 0
	}

	if maxSeq < 1 {
		maxSeq = 1
	}

	if kvDim < 0 {
		kvDim = 0
	}

	layerBytes := maxSeq * kvDim
	layers := make([]kvLayer, numLayers)
	for i := range layers {
		if arena != nil {
			layers[i].k = arena.Alloc(layerBytes)
			layers[i].v = arena.Alloc(layerBytes)
		} else {
			layers[i].k = make([]float32, layerBytes)
			layers[i].v = make([]float32, layerBytes)
		}
	}

	return &KV{
		layers: layers,
		kvDim:  kvDim,
		maxSeq: maxSeq,
	}
}

// Len - number of completed tokens (Advance counter).
func (c *KV) Len() int {
	return c.length
}

// Append appends K/V of one token to layer.
func (c *KV) Append(layer int, k, v []float32) {
	c.AppendN(layer, k, v, 1)
}

// AppendN appends K/V of n tokens to layer (k/v: [n*kvDim]).
func (c *KV) AppendN(layer int, k, v []float32, n int) {
	if n < 1 || layer < 0 || layer >= len(c.layers) {
		return
	}

	l := &c.layers[layer]
	for i := range n {
		if l.n >= c.maxSeq {
			return
		}

		off := l.n * c.kvDim
		src := i * c.kvDim
		copy(l.k[off:off+c.kvDim], k[src:src+c.kvDim])
		copy(l.v[off:off+c.kvDim], v[src:src+c.kvDim])
		l.n++
	}
}

// Advance marks token completion.
func (c *KV) Advance() {
	c.AdvanceN(1)
}

// AdvanceN marks completion of n tokens (n_batch prefill).
func (c *KV) AdvanceN(n int) {
	if n < 1 {
		return
	}

	c.length += n
}

// KLayer returns layer K [n*kvDim].
func (c *KV) KLayer(layer int) []float32 {
	l := &c.layers[layer]
	return l.k[:l.n*c.kvDim]
}

// VLayer returns layer V.
func (c *KV) VLayer(layer int) []float32 {
	l := &c.layers[layer]
	return l.v[:l.n*c.kvDim]
}

// Reset clears length and layer counters.
func (c *KV) Reset() {
	c.length = 0
	for i := range c.layers {
		c.layers[i].n = 0
	}
}
