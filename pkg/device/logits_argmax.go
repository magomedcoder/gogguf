package device

import "github.com/magomedcoder/gogguf/pkg/dtype"

// LogitsArgmaxBackend optionally computes RMSNorm+lm_head+argmax on resident hidden and copies only {idx, value} to host (greedy decode path)
type LogitsArgmaxBackend interface {
	LogitsArgmaxFromDevice(t dtype.DType, normName string, norm []float32, headName string, headRaw []byte, headF32 []float32, vocab, embd int, eps float32) (idx int, val float32, err error)
}

// LogitsArgmaxFromDevice calls LogitsArgmaxBackend when available
func LogitsArgmaxFromDevice(b Backend, t dtype.DType, normName string, norm []float32, headName string, headRaw []byte, headF32 []float32, vocab, embd int, eps float32) (int, float32, error) {
	if lb, ok := b.(LogitsArgmaxBackend); ok {
		return lb.LogitsArgmaxFromDevice(t, normName, norm, headName, headRaw, headF32, vocab, embd, eps)
	}

	return 0, 0, ErrHiddenUnavailable
}
