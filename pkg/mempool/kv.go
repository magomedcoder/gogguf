package mempool

// KV - фиксированный CPU KV-cache: буферы maxSeq*kvDim, запись без append.
// На каждый слой свой счётчик токенов (как у append); Len/Advance - отдельно
type KV struct {
	layers []kvLayer
	kvDim  int
	maxSeq int
	length int
}

type kvLayer struct {
	k []float32
	v []float32
	n int // число записанных токенов
}

// NewKV выделяет KV на numLayers слоёв.
// arena может быть nil - тогда отдельные make
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

// Len - число завершённых токенов (счётчик Advance).
func (c *KV) Len() int {
	return c.length
}

// Append дописывает K/V одного токена в слой.
func (c *KV) Append(layer int, k, v []float32) {
	if layer < 0 || layer >= len(c.layers) {
		return
	}

	l := &c.layers[layer]
	if l.n >= c.maxSeq {
		return
	}

	off := l.n * c.kvDim
	copy(l.k[off:off+c.kvDim], k)
	copy(l.v[off:off+c.kvDim], v)
	l.n++
}

// Advance отмечает завершение токена.
func (c *KV) Advance() {
	c.length++
}

// KLayer возвращает K слоя [n*kvDim].
func (c *KV) KLayer(layer int) []float32 {
	l := &c.layers[layer]
	return l.k[:l.n*c.kvDim]
}

// VLayer возвращает V слоя.
func (c *KV) VLayer(layer int) []float32 {
	l := &c.layers[layer]
	return l.v[:l.n*c.kvDim]
}

// Reset очищает длину и счётчики слоёв.
func (c *KV) Reset() {
	c.length = 0
	for i := range c.layers {
		c.layers[i].n = 0
	}
}
