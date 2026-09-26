//go:build metal

package gpu

import (
	"fmt"

	"github.com/magomedcoder/gogguf/pkg/gpu/metal"
)

// OpenMetal возвращает Metal-backend (scaffold: compute-вызовы отдают ошибку).
// Собирается только с тегом metal, работает только на macOS
func OpenMetal() (Backend, error) {
	b, err := metal.Open()
	if err != nil {
		return nil, fmt.Errorf("%w: %w", ErrMetalUnavailable, err)
	}

	return b, nil
}
