//go:build !cgo || !darwin

package llama2

import (
	"runtime"
	"sync"
)

var matmulConcurrency = runtime.GOMAXPROCS(0) * 2

func matmul(xout, x, w []float32) {
	var wg sync.WaitGroup
	wg.Add(matmulConcurrency)

	rowsPerThread := len(xout) / matmulConcurrency
	for thread := 0; thread < matmulConcurrency; thread++ {
		start := thread * rowsPerThread
		end := start + rowsPerThread
		if thread == matmulConcurrency-1 {
			end = len(xout)
		}

		if end > len(xout) {
			return
		}

		go func(xout, w []float32) {
			for i := range xout {
				var val float32
				in := i * len(x)

				j := 0
				for ; j < len(x)-8; j += 8 {
					w := w[in+j : in+j+8]
					x := x[j : j+8] // bce

					val += w[0] * x[0]
					val += w[1] * x[1]
					val += w[2] * x[2]
					val += w[3] * x[3]
					val += w[4] * x[4]
					val += w[5] * x[5]
					val += w[6] * x[6]
					val += w[7] * x[7]
				}

				for ; j < len(x)-4; j += 4 {
					w := w[in+j : in+j+4]
					x := x[j : j+4] // bce

					val += w[0] * x[0]
					val += w[1] * x[1]
					val += w[2] * x[2]
					val += w[3] * x[3]
				}

				for ; j < len(x); j++ {
					val += w[in+j] * x[j]
				}

				xout[i] = val
			}
			wg.Done()
		}(xout[start:end], w[len(x)*start:len(x)*end])
	}

	wg.Wait()
}
