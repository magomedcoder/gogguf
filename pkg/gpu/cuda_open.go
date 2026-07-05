//go:build cuda

package gpu

import "github.com/magomedcoder/gogguf/pkg/gpu/cuda"

// OpenCUDA инициализирует CUDA через Driver API (libcuda)
func OpenCUDA() (Backend, error) {
	return cuda.Open()
}
