package mempool

import "fmt"

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
func (c *KV) Append(layer int, k, v []float32) error {
	return c.AppendN(layer, k, v, 1)
}

// Bump reserves n token slots on a layer without writing K/V.
// Used when GPU KV is the source of truth and the host mirror is skipped.
func (c *KV) Bump(layer, n int) error {
	if n < 1 || layer < 0 || layer >= len(c.layers) {
		return fmt.Errorf("mempool: invalid KV Bump layer=%d n=%d", layer, n)
	}
	l := &c.layers[layer]
	if l.n+n > c.maxSeq {
		return fmt.Errorf("mempool: KV overflow layer=%d have=%d add=%d maxSeq=%d", layer, l.n, n, c.maxSeq)
	}
	l.n += n
	return nil
}

// AppendN appends K/V of n tokens to layer (k/v: [n*kvDim]).
func (c *KV) AppendN(layer int, k, v []float32, n int) error {
	if n < 1 || layer < 0 || layer >= len(c.layers) {
		return fmt.Errorf("mempool: invalid KV AppendN layer=%d n=%d", layer, n)
	}

	l := &c.layers[layer]
	if l.n+n > c.maxSeq {
		return fmt.Errorf("mempool: KV overflow layer=%d have=%d add=%d maxSeq=%d", layer, l.n, n, c.maxSeq)
	}

	for i := range n {
		off := l.n * c.kvDim
		src := i * c.kvDim
		copy(l.k[off:off+c.kvDim], k[src:src+c.kvDim])
		copy(l.v[off:off+c.kvDim], v[src:src+c.kvDim])
		l.n++
	}
	return nil
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

// DropOldest removes the oldest n tokens from every layer (sliding-window reclaim).
// Callers that use absolute RoPE positions must keep that contract themselves.
func (c *KV) DropOldest(n int) {
	if n < 1 {
		return
	}

	for i := range c.layers {
		c.dropOldestLayer(i, n)
	}

	if c.length > n {
		c.length -= n
	} else {
		c.length = 0
	}
}

// DropOldestLayer removes the oldest n tokens from one layer without changing Len().
// Used for hybrid SWA (e.g. Gemma2) where only some layers reclaim.
func (c *KV) DropOldestLayer(layer, n int) {
	if n < 1 || layer < 0 || layer >= len(c.layers) {
		return
	}

	c.dropOldestLayer(layer, n)
}

func (c *KV) dropOldestLayer(layer, n int) {
	l := &c.layers[layer]
	if l.n == 0 {
		return
	}

	drop := n
	if drop > l.n {
		drop = l.n
	}

	keep := l.n - drop
	if keep > 0 {
		copy(l.k[:keep*c.kvDim], l.k[drop*c.kvDim:l.n*c.kvDim])
		copy(l.v[:keep*c.kvDim], l.v[drop*c.kvDim:l.n*c.kvDim])
	}
	l.n = keep
}

// LayerLen returns how many tokens are stored in one layer.
func (c *KV) LayerLen(layer int) int {
	if layer < 0 || layer >= len(c.layers) {
		return 0
	}

	return c.layers[layer].n
}

// TokensStored returns the max per-layer token count (occupancy before Advance sync).
func (c *KV) TokensStored() int {
	max := 0
	for i := range c.layers {
		if c.layers[i].n > max {
			max = c.layers[i].n
		}
	}

	return max
}

// EnsureRoom drops oldest tokens across all layers so TokensStored()+n fits in MaxSeq.
// Returns how many tokens were dropped (0 if already enough room).
func (c *KV) EnsureRoom(n int) int {
	if n < 1 {
		return 0
	}

	have := c.TokensStored()
	if have+n <= c.maxSeq {
		return 0
	}

	drop := have + n - c.maxSeq
	c.DropOldest(drop)

	return drop
}

// MaxSeq returns the allocated token capacity per layer.
func (c *KV) MaxSeq() int {
	return c.maxSeq
}

// Reset clears length and layer counters.
func (c *KV) Reset() {
	c.length = 0
	for i := range c.layers {
		c.layers[i].n = 0
	}
}
