//go:build metal

package device

import (
	"fmt"

	"github.com/magomedcoder/gogguf/pkg/device/metal"
)

// optional capabilities of the Metal backend (compile-time checks): B3 prefill GEMM + batch FFN + batch attention
var (
	_ MatMulMatBackend        = (*metal.Backend)(nil)
	_ FFNSwiGLUMatBackend     = (*metal.Backend)(nil)
	_ KVBatchAttentionBackend = (*metal.Backend)(nil)
)

// OpenMetal returns Metal backend (scaffold: compute calls return error).
// Built only with metal tag, runs only on macOS
func OpenMetal() (Backend, error) {
	b, err := metal.Open()
	if err != nil {
		return nil, fmt.Errorf("%w: %w", ErrMetalUnavailable, err)
	}

	return b, nil
}
