package mistral

import "github.com/magomedcoder/gogguf/pkg/model/mempool"

// KVCache stores K/V for autoregressive decode (fixed buffers, no append)
type KVCache struct {
	*mempool.KV
}

// NewKVCache creates an empty KV-cache (lazy init; see mempool.NewKV).
// SWA models use SlidingWindow as maxSeq and reclaim via EnsureRoom/DropOldest; otherwise maxSeq is ContextLength.
func NewKVCache(cfg Config) *KVCache {
	kvDim := cfg.NumKVHeads * cfg.HeadDim
	return &KVCache{
		KV: mempool.NewKV(cfg.NumLayers, kvMaxSeq(cfg), kvDim, nil),
	}
}

func kvMaxSeq(cfg Config) int {
	if cfg.SlidingWindow > 0 && cfg.SlidingWindow < cfg.ContextLength {
		return cfg.SlidingWindow
	}

	return cfg.ContextLength
}
