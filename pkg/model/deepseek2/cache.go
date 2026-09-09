package deepseek2

// MLACache хранит K и V разной ширины (поглощённая MLA: K=lora+rope, V=lora)
type MLACache struct {
	layers []mlaLayer
	kDim   int
	vDim   int
	maxSeq int
	length int
}

type mlaLayer struct {
	k []float32
	v []float32
	n int
}

func NewMLACache(cfg Config) *MLACache {
	kDim := cfg.qkDim()
	vDim := cfg.KVLoraRank
	layers := make([]mlaLayer, cfg.NumLayers)
	for i := range layers {
		layers[i].k = make([]float32, cfg.ContextLength*kDim)
		layers[i].v = make([]float32, cfg.ContextLength*vDim)
	}

	return &MLACache{
		layers: layers, 
		kDim: kDim,
		vDim: vDim,
		maxSeq: cfg.ContextLength,
	}
}

func (c *MLACache) Len() int {
	return c.length
}

func (c *MLACache) Append(layer int, k, v []float32) {
	l := &c.layers[layer]
	if l.n >= c.maxSeq {
		return
	}
	copy(l.k[l.n*c.kDim:(l.n+1)*c.kDim], k)
	copy(l.v[l.n*c.vDim:(l.n+1)*c.vDim], v)
	l.n++
}

func (c *MLACache) Advance() {
	c.length++
}

func (c *MLACache) KLayer(layer int) []float32 {
	l := &c.layers[layer]

	return l.k[:l.n*c.kDim]
}

func (c *MLACache) VLayer(layer int) []float32 {
	l := &c.layers[layer]

	return l.v[:l.n*c.vDim]
}

func (c *MLACache) Reset() {
	c.length = 0
	for i := range c.layers {
		c.layers[i].n = 0
	}
}
