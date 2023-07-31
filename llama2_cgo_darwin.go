//go:build cgo && darwin

package llama2

// #cgo LDFLAGS: -framework Accelerate
// #include <Accelerate/Accelerate.h>
import "C"

func matmul(xout, x, w []float32) {
	C.cblas_sgemv(
		C.CblasRowMajor,
		C.CblasNoTrans,
		C.int(len(xout)),
		C.int(len(x)),
		C.float(1),
		(*C.float)(&w[0]),
		C.int(len(x)),
		(*C.float)(&x[0]),
		C.int(1),
		C.float(0),
		(*C.float)(&xout[0]),
		C.int(1))
}
