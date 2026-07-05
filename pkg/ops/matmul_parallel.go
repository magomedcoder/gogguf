package ops

import (
	"runtime"
	"sync"
)

func parallelForRows(rows int, fn func(rowStart, rowEnd int)) {
	if rows < parallelMatMulMinRows {
		fn(0, rows)
		return
	}

	workers := min(runtime.GOMAXPROCS(0), rows)

	chunk := (rows + workers - 1) / workers
	var wg sync.WaitGroup
	for start := 0; start < rows; start += chunk {
		end := min(start+chunk, rows)

		wg.Add(1)
		go func(rowStart, rowEnd int) {
			defer wg.Done()
			fn(rowStart, rowEnd)
		}(start, end)
	}

	wg.Wait()
}
