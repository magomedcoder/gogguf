package ops

import (
	"runtime"
	"sync"
	"sync/atomic"
)

// Persistent worker pool for parallel matmul. Nested calls on the same goroutine
// and jobs running on pool workers stay serial to avoid deadlock.
var (
	poolOnce    sync.Once
	poolJobs    chan func()
	poolWorkers int
)

func initMatmulPool() {
	poolWorkers = max(runtime.GOMAXPROCS(0), 1)

	poolJobs = make(chan func(), poolWorkers*4)
	for range poolWorkers {
		go func() {
			for fn := range poolJobs {
				key := goroutineKey()
				matmulWorkers.Store(key, struct{}{})
				fn()
				matmulWorkers.Delete(key)
			}
		}()
	}
}

func parallelForRows(rows int, fn func(rowStart, rowEnd int)) {
	parallelForWork(rows, parallelMatMulMinRows, fn)
}

// parallelForWork like parallelForRows with a caller-chosen minimum item count.
func parallelForWork(items, minItems int, fn func(start, end int)) {
	depth, depthKey := parallelDepthSlot()
	if items < minItems || *depth > 0 || onMatmulWorker() {
		fn(0, items)
		return
	}

	poolOnce.Do(initMatmulPool)

	workers := min(poolWorkers, items)
	if workers <= 1 {
		fn(0, items)
		return
	}

	*depth++
	defer func() {
		*depth--
		if *depth == 0 {
			parallelDepth.Delete(depthKey)
		}
	}()

	chunk := (items + workers - 1) / workers
	var wg sync.WaitGroup
	for start := 0; start < items; start += chunk {
		end := min(start+chunk, items)
		wg.Add(1)
		s, e := start, end
		poolJobs <- func() {
			defer wg.Done()
			fn(s, e)
		}
	}
	wg.Wait()
}

func parallelForRowsErr(rows int, fn func(rowStart, rowEnd int) error) error {
	var first atomic.Pointer[error]
	parallelForRows(rows, func(rowStart, rowEnd int) {
		if err := fn(rowStart, rowEnd); err != nil {
			first.CompareAndSwap(nil, &err)
		}
	})
	if p := first.Load(); p != nil {
		return *p
	}

	return nil
}
