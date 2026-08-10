package phi2

import "github.com/magomedcoder/gogguf/pkg/mempool"

// KVCache хранит K/V для autoregressive decode
type KVCache struct {
	*mempool.KV
}

// NewKVCache создаёт пустой KV-cache ёмкостью ContextLength
func NewKVCache(cfg Config) *KVCache {
	kvDim := cfg.NumKVHeads * cfg.HeadDim
	return &KVCache{
		KV: mempool.NewKV(cfg.NumLayers, cfg.ContextLength, kvDim, nil),
	}
}
