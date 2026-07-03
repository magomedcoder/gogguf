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

	workers := runtime.GOMAXPROCS(0)
	if workers > rows {
		workers = rows
	}

	chunk := (rows + workers - 1) / workers
	var wg sync.WaitGroup
	for start := 0; start < rows; start += chunk {
		end := start + chunk
		if end > rows {
			end = rows
		}

		wg.Add(1)
		go func(rowStart, rowEnd int) {
			defer wg.Done()
			fn(rowStart, rowEnd)
		}(start, end)
	}

	wg.Wait()
}
