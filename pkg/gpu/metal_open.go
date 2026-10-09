//go:build metal

package gpu

import (
	"fmt"

	"github.com/magomedcoder/gogguf/pkg/gpu/metal"
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
