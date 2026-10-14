//go:build cuda

package cuda

import "testing"

// newCacheTestBackend returns a Backend with no device (frees are no-ops on a zeroed driver).
func newCacheTestBackend() *Backend {
	return &Backend{
		matrices:    make(map[string]gpuMatrix),
		matricesQ8:  make(map[string]gpuQ8Matrix),
		matricesQ4:  make(map[string]gpuQ8Matrix),
		matricesQ4K: make(map[string]gpuQ8Matrix),
		matricesQ5K: make(map[string]gpuQ8Matrix),
		matricesQ6K: make(map[string]gpuQ8Matrix),
	}
}

func (b *Backend) fakeUpload(kind, name string, bytes int) {
	switch kind {
	case kindF32:
		b.matrices[name] = gpuMatrix{ptr: 1, rows: 1, cols: bytes / 4}
	case kindQ8:
		b.matricesQ8[name] = gpuQ8Matrix{ptr: 1, rows: 1, cols: 1, bytes: bytes}
	case kindQ4K:
		b.matricesQ4K[name] = gpuQ8Matrix{ptr: 1, rows: 1, cols: 1, bytes: bytes}
	}

	b.trackWeight(kind, name, bytes)
}

func TestWeightCacheCapLRU(t *testing.T) {
	b := newCacheTestBackend()
	b.SetWeightCacheLimits(10, 0)

	names := []string{"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l"}
	for _, n := range names {
		b.fakeUpload(kindQ8, n, 100)
	}

	entries, bytes, ev := b.WeightCacheStats()
	if entries != 10 || bytes != 1000 || ev != 2 {
		t.Fatalf("entries=%d bytes=%d evictions=%d", entries, bytes, ev)
	}

	if _, ok := b.matricesQ8["a"]; ok {
		t.Fatal("oldest 'a' should be evicted")
	}

	if _, ok := b.matricesQ8["l"]; !ok {
		t.Fatal("newest must stay")
	}

	// a hit refreshes recency: 'c' survives next eviction round
	b.touchWeight(kindQ8, "c")
	b.fakeUpload(kindQ8, "m", 100)
	if _, ok := b.matricesQ8["c"]; !ok {
		t.Fatal("touched 'c' evicted")
	}

	if _, ok := b.matricesQ8["d"]; ok {
		t.Fatal("'d' should be next LRU victim")
	}
}

func TestWeightCacheProtectsWorkingSet(t *testing.T) {
	b := newCacheTestBackend()
	b.SetWeightCacheLimits(2, 0) // below weightProtect

	for _, n := range []string{"a", "b", "c", "d", "e", "f", "g", "h", "i", "j"} {
		b.fakeUpload(kindQ4K, n, 10)
	}

	entries, _, _ := b.WeightCacheStats()
	if entries != weightProtect {
		t.Fatalf("entries=%d want %d (protected set)", entries, weightProtect)
	}
}

func TestWeightCacheOOMRetryEvicts(t *testing.T) {
	b := newCacheTestBackend()
	for i := 0; i < 12; i++ {
		b.fakeUpload(kindQ8, string(rune('a'+i)), 64)
	}

	// first two attempts "OOM", third succeeds after two evictions
	attempts := 0
	rc := b.evictRetry(func() int {
		attempts++
		if attempts <= 2 {
			return -1
		}
		return 0
	})
	if rc != 0 || attempts != 3 {
		t.Fatalf("rc=%d attempts=%d", rc, attempts)
	}

	entries, _, ev := b.WeightCacheStats()
	if entries != 10 || ev != 2 {
		t.Fatalf("entries=%d evictions=%d", entries, ev)
	}

	// nothing evictable (<= protect) -> OOM surfaces
	small := newCacheTestBackend()
	for i := 0; i < 3; i++ {
		small.fakeUpload(kindQ8, string(rune('a'+i)), 64)
	}

	if rc := small.evictRetry(func() int { return -1 }); rc != -1 {
		t.Fatalf("rc=%d want -1", rc)
	}
}

func TestEvictQuantMatrixAndPrefix(t *testing.T) {
	b := newCacheTestBackend()
	b.fakeUpload(kindQ8, "blk.0.attn_q.weight", 10)
	b.fakeUpload(kindQ4K, "blk.0.ffn_up.weight", 10)
	b.fakeUpload(kindF32, "blk.1.norm.weight", 8)

	if !b.EvictQuantMatrix("blk.0.attn_q.weight") {
		t.Fatal("expected evict")
	}

	if b.EvictQuantMatrix("missing") {
		t.Fatal("missing must report false")
	}

	if n := b.EvictMatricesWithPrefix("blk.0."); n != 1 {
		t.Fatalf("prefix evicted %d", n)
	}

	entries, bytes, _ := b.WeightCacheStats()
	if entries != 1 || bytes != 8 || len(b.matrices) != 1 {
		t.Fatalf("entries=%d bytes=%d", entries, bytes)
	}
}
