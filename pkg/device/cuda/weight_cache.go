//go:build cuda

package cuda

/*
#include "driver.h"
*/
import "C"

import (
	"container/list"
	"os"
	"strconv"
	"strings"

	"github.com/magomedcoder/gogguf/pkg/dtype"
)

// GPU weight cache: LRU bookkeeping + eviction
//
// Uploaded weight matrices (FP32 and quantized kinds) live in per-kind maps keyed by tensor name.
// This file adds an LRU order over all of them so VRAM can be reclaimed:
//
//   - OOM retry: every weight upload goes through withEvict; on cuMemAlloc failure (code -1) the least recently used matrices are freed one by one and the upload retried.
//   - Optional caps: SetWeightCacheLimits(maxMatrices, maxBytes) evicts after each upload (0 = unlimited). Env at Open: GOGGUF_CUDA_MAX_MATRICES, GOGGUF_CUDA_WEIGHT_CACHE_MB.
//   - Explicit: EvictQuantMatrix(name), EvictMatrix(name), EvictMatricesWithPrefix("blk.3.").
//
// The weightProtect most recently touched matrices are never auto-evicted: fused ops (QKV+WO, gate/up/down) hold several device pointers at once and a matrix is touched when it is ensured, so the working set of the current op stays valid.
// Caps below weightProtect therefore act as weightProtect. Evicting also drops captured CUDA graphs (they embed weight pointers).
// Evicted matrices are re-uploaded transparently on next use (host bytes come from mmap).
// If the most recent matrices alone exceed VRAM, the upload still fails with errUploadOOM.

const weightProtect = 8

type weightKey struct {
	kind string
	name string
}

type weightEntry struct {
	key   weightKey
	bytes int64
}

// weightCache is the LRU order (front = most recently used)
type weightCache struct {
	order      *list.List
	idx        map[weightKey]*list.Element
	bytes      int64
	maxEntries int
	maxBytes   int64
	evictions  int
}

func (c *weightCache) init() {
	if c.order == nil {
		c.order = list.New()
		c.idx = make(map[weightKey]*list.Element)
	}
}

// weight cache kinds (match the per-kind maps in Backend)
const (
	kindF32 = "f32"
	kindQ8  = "q8_0"
	kindQ4  = "q4_0"
	kindQ4K = "q4_k"
	kindQ5K = "q5_k"
	kindQ6K = "q6_k"
	kindQ41 = "q4_1"
	kindQ50 = "q5_0"
	kindQ51 = "q5_1"
	kindQ2K = "q2_k"
	kindQ3K = "q3_k"
	kindQ8K = "q8_k"
)

// quantKind maps a quant dtype to its weight cache kind
func quantKind(t dtype.DType) string {
	switch t {
	case dtype.Q8_0:
		return kindQ8
	case dtype.Q4_0:
		return kindQ4
	case dtype.Q4_K:
		return kindQ4K
	case dtype.Q5_K:
		return kindQ5K
	case dtype.Q6_K:
		return kindQ6K
	case dtype.Q4_1:
		return kindQ41
	case dtype.Q5_0:
		return kindQ50
	case dtype.Q5_1:
		return kindQ51
	case dtype.Q2_K:
		return kindQ2K
	case dtype.Q3_K:
		return kindQ3K
	case dtype.Q8_K:
		return kindQ8K
	}

	return ""
}

var allWeightKinds = []string{kindF32, kindQ8, kindQ4, kindQ4K, kindQ5K, kindQ6K, kindQ41, kindQ50, kindQ51, kindQ2K, kindQ3K, kindQ8K}

// loadWeightCacheEnv applies optional env limits (called from OpenDevice)
func (b *Backend) loadWeightCacheEnv() {
	if v, err := strconv.Atoi(strings.TrimSpace(os.Getenv("GOGGUF_CUDA_MAX_MATRICES"))); err == nil && v > 0 {
		b.wc.maxEntries = v
	}

	if v, err := strconv.ParseInt(strings.TrimSpace(os.Getenv("GOGGUF_CUDA_WEIGHT_CACHE_MB")), 10, 64); err == nil && v > 0 {
		b.wc.maxBytes = v << 20
	}
}

// SetWeightCacheLimits caps uploaded weight matrices: at most maxMatrices entries and maxBytes of device memory (0 = unlimited).
// Enforced after each upload; see weightProtect.
func (b *Backend) SetWeightCacheLimits(maxMatrices int, maxBytes int64) {
	b.mu.Lock()
	defer b.mu.Unlock()

	b.wc.maxEntries = maxMatrices
	b.wc.maxBytes = maxBytes
	b.enforceWeightCaps()
}

// WeightCacheStats returns number of cached device matrices, their bytes and total LRU evictions.
func (b *Backend) WeightCacheStats() (entries int, bytes int64, evictions int) {
	b.mu.Lock()
	defer b.mu.Unlock()

	b.wc.init()
	return b.wc.order.Len(), b.wc.bytes, b.wc.evictions
}

// trackWeight records (or refreshes) an uploaded matrix as most recently used, then enforces caps.
func (b *Backend) trackWeight(kind, name string, bytes int) {
	b.wc.init()

	k := weightKey{kind, name}
	if el, ok := b.wc.idx[k]; ok {
		e := el.Value.(*weightEntry)
		b.wc.bytes += int64(bytes) - e.bytes
		e.bytes = int64(bytes)
		b.wc.order.MoveToFront(el)
	} else {
		b.wc.idx[k] = b.wc.order.PushFront(&weightEntry{key: k, bytes: int64(bytes)})
		b.wc.bytes += int64(bytes)
	}

	b.enforceWeightCaps()
}

// touchWeight marks a cache hit as most recently used.
func (b *Backend) touchWeight(kind, name string) {
	b.wc.init()

	if el, ok := b.wc.idx[weightKey{kind, name}]; ok {
		b.wc.order.MoveToFront(el)
	}
}

// untrackWeight removes bookkeeping for a matrix freed elsewhere.
func (b *Backend) untrackWeight(kind, name string) {
	b.wc.init()

	k := weightKey{kind, name}
	if el, ok := b.wc.idx[k]; ok {
		b.wc.bytes -= el.Value.(*weightEntry).bytes
		b.wc.order.Remove(el)
		delete(b.wc.idx, k)
	}
}

// freeWeight frees device memory of one cached matrix and drops it from map and LRU.
func (b *Backend) freeWeight(kind, name string) bool {
	var ptr C.CUdeviceptr
	found := false

	if kind == kindF32 {
		if m, ok := b.matrices[name]; ok {
			ptr, found = m.ptr, true
			delete(b.matrices, name)
		}
	} else {
		var cache map[string]gpuQ8Matrix
		switch kind {
		case kindQ8:
			cache = b.matricesQ8
		case kindQ4:
			cache = b.matricesQ4
		case kindQ4K:
			cache = b.matricesQ4K
		case kindQ5K:
			cache = b.matricesQ5K
		case kindQ6K:
			cache = b.matricesQ6K
		case kindQ41:
			cache = b.matricesQ41
		case kindQ50:
			cache = b.matricesQ50
		case kindQ51:
			cache = b.matricesQ51
		case kindQ2K:
			cache = b.matricesQ2K
		case kindQ3K:
			cache = b.matricesQ3K
		case kindQ8K:
			cache = b.matricesQ8K
		}

		if m, ok := cache[name]; ok {
			ptr, found = m.ptr, true
			delete(cache, name)
		}
	}

	b.untrackWeight(kind, name)
	if !found {
		return false
	}

	// captured graphs embed weight pointers
	C.gguf_cuda_matmul_pool_clear_graphs(&b.drv, &b.matmulPool)
	C.gguf_cuda_free(&b.drv, ptr)
	b.lastVecAddr = 0
	b.lastVecLen = 0

	return true
}

// evictOldestWeight frees the least recently used matrix outside the protected working set.
func (b *Backend) evictOldestWeight() bool {
	b.wc.init()

	if b.wc.order.Len() <= weightProtect {
		return false
	}

	e := b.wc.order.Back().Value.(*weightEntry)
	if !b.freeWeight(e.key.kind, e.key.name) {
		return false
	}

	b.wc.evictions++
	return true
}

// enforceWeightCaps evicts LRU entries while over the configured caps.
func (b *Backend) enforceWeightCaps() {
	b.wc.init()

	for (b.wc.maxEntries > 0 && b.wc.order.Len() > b.wc.maxEntries) || (b.wc.maxBytes > 0 && b.wc.bytes > b.wc.maxBytes) {
		if !b.evictOldestWeight() {
			return
		}
	}
}

// withEvict runs an upload; on VRAM exhaustion (rc == -1) evicts LRU matrices and retries.
// Caller holds b.mu.
func (b *Backend) withEvict(upload func() C.int) C.int {
	return C.int(b.evictRetry(func() int {
		return int(upload())
	}))
}

// evictRetry is withEvict on plain ints (rc -1 == out of VRAM)
func (b *Backend) evictRetry(upload func() int) int {
	rc := upload()
	for rc == -1 {
		if !b.evictOldestWeight() {
			break
		}

		rc = upload()
	}

	return rc
}

// EvictQuantMatrix frees the device copy of weight `name` (any quant kind or FP32).
// The matrix is re-uploaded on next use.
// Returns true if something was freed.
func (b *Backend) EvictQuantMatrix(name string) bool {
	b.mu.Lock()
	defer b.mu.Unlock()

	freed := false
	for _, kind := range allWeightKinds {
		if b.freeWeight(kind, name) {
			freed = true
			b.wc.evictions++
		}
	}

	return freed
}

// EvictMatrix is an alias of EvictQuantMatrix.
func (b *Backend) EvictMatrix(name string) bool {
	return b.EvictQuantMatrix(name)
}

// EvictMatricesWithPrefix frees all cached matrices whose name starts with prefix (e.g. "blk.3." after layer 3 finished).
// Returns the number of matrices freed.
func (b *Backend) EvictMatricesWithPrefix(prefix string) int {
	b.mu.Lock()
	defer b.mu.Unlock()

	b.wc.init()

	var victims []weightKey
	for k := range b.wc.idx {
		if strings.HasPrefix(k.name, prefix) {
			victims = append(victims, k)
		}
	}

	n := 0
	for _, k := range victims {
		if b.freeWeight(k.kind, k.name) {
			n++
			b.wc.evictions++
		}
	}

	return n
}
