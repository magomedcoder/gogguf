package mistral

import "github.com/magomedcoder/gogguf/pkg/mempool"

// KVCache stores K/V for autoregressive decode (fixed buffers, no append)
type KVCache struct {
	*mempool.KV
}

// NewKVCache creates an empty KV-cache with capacity ContextLength
func NewKVCache(cfg Config) *KVCache {
	kvDim := cfg.NumKVHeads * cfg.HeadDim
	return &KVCache{
		KV: mempool.NewKV(cfg.NumLayers, cfg.ContextLength, kvDim, nil),
	}
}
