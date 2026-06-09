package qwen3

import "testing"

func TestKVCacheAdvance(t *testing.T) {
	cfg := Config{
		NumLayers: 2,
		NumKVHeads: 1,
		HeadDim: 4,
		ContextLength: 8,
	}
	c := NewKVCache(cfg)

	k := []float32{1, 2, 3, 4}
	v := []float32{5, 6, 7, 8}
	c.Append(0, k, v)
	c.Append(1, k, v)
	if c.Len() != 0 {
		t.Fatalf("Len before advance = %d", c.Len())
	}

	c.Advance()

	if c.Len() != 1 {
		t.Fatalf("Len after advance = %d", c.Len())
	}

	if len(c.KLayer(0)) != 4 {
		t.Fatalf("k layer size = %d", len(c.KLayer(0)))
	}
}
