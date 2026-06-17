//go:build !cuda

package gpu

// OpenCUDA возвращает CUDA-backend или ErrUnavailable если сборка без тега cuda
func OpenCUDA() (Backend, error) {
	return nil, ErrUnavailable
}
