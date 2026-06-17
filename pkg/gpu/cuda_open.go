//go:build cuda

package gpu

import "github.com/magomedcoder/gguf.go/pkg/gpu/cuda"

// OpenCUDA инициализирует CUDA через Driver API (libcuda)
func OpenCUDA() (Backend, error) {
	return cuda.Open()
}
