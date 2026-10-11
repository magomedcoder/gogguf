//go:build !metal

package device

// OpenMetal without metal tag unavailable (does not affect CUDA path)
func OpenMetal() (Backend, error) {
	return nil, ErrMetalUnavailable
}
