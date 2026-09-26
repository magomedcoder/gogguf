//go:build !metal

package gpu

// OpenMetal без тега metal недоступен (CUDA-путь это не затрагивает)
func OpenMetal() (Backend, error) {
	return nil, ErrMetalUnavailable
}
