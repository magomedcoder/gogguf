package qwen3

// KVCache хранит K/V для autoregressive decode
type KVCache struct {
	cfg    Config
	layers []layerKV
	length int
}

type layerKV struct {
	k []float32
	v []float32
}

// NewKVCache создаёт пустой KV-cache
func NewKVCache(cfg Config) *KVCache {
	layers := make([]layerKV, cfg.NumLayers)
	kvDim := cfg.NumKVHeads * cfg.HeadDim
	for i := range layers {
		cap := cfg.ContextLength * kvDim
		layers[i].k = make([]float32, 0, cap)
		layers[i].v = make([]float32, 0, cap)
	}

	return &KVCache{
		cfg: cfg,
		layers: layers,
	}
}

// Len возвращает длину последовательности в cache
func (c *KVCache) Len() int {
	return c.length
}

// Append добавляет K/V одного токена для слоя
func (c *KVCache) Append(layer int, k, v []float32) {
	c.layers[layer].k = append(c.layers[layer].k, k...)
	c.layers[layer].v = append(c.layers[layer].v, v...)
}

// Advance отмечает завершение обработки одного токена
func (c *KVCache) Advance() {
	c.length++
}

// KLayer возвращает K слоя [seqLen×kvDim]
func (c *KVCache) KLayer(layer int) []float32 {
	return c.layers[layer].k
}

// VLayer возвращает V слоя
func (c *KVCache) VLayer(layer int) []float32 {
	return c.layers[layer].v
}

// Reset очищает cache
func (c *KVCache) Reset() {
	for i := range c.layers {
		c.layers[i].k = c.layers[i].k[:0]
		c.layers[i].v = c.layers[i].v[:0]
	}
	c.length = 0
}
