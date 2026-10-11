package vec

import (
	"sync"
	"unsafe"
)

// Per-goroutine parallel matmul depth. A global nest counter would serialize
// unrelated Forwards on different goroutines; only nested calls on the same
// goroutine should run serially.
var (
	parallelDepth sync.Map // uintptr -> *int32
	matmulWorkers sync.Map // goroutine stack key -> struct{}
)

func goroutineKey() uintptr {
	var marker byte
	return uintptr(unsafe.Pointer(&marker))
}

func onMatmulWorker() bool {
	_, ok := matmulWorkers.Load(goroutineKey())
	return ok
}

func parallelDepthSlot() (depth *int32, key uintptr) {
	key = goroutineKey()
	if v, ok := parallelDepth.Load(key); ok {
		return v.(*int32), key
	}

	d := new(int32)
	parallelDepth.Store(key, d)

	return d, key
}
