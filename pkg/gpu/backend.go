package gpu

// Backend выполняет вычисления на GPU (CUDA)
type Backend interface {
	// Name возвращает имя устройства, например "CUDA:0 NVIDIA ..."
	Name() string

	// MatMulVec умножает matrix[rows*cols] на vec[cols]
	MatMulVec(matrix []float32, rows, cols int, vec []float32) ([]float32, error)

	// MatMulVecCached как MatMulVec, но matrix загружается на GPU один раз по name
	MatMulVecCached(name string, matrix []float32, rows, cols int, vec []float32) ([]float32, error)

	// MatMulVecQ8_0Cached matmul Q8_0-матрицы без деквантизации в FP32
	MatMulVecQ8_0Cached(name string, raw []byte, rows, cols int, vec []float32) ([]float32, error)

	// RMSNormInto записывает RMS-нормализацию в dst (GPU или CPU)
	RMSNormInto(dst, x, weight []float32, eps float32) error

	// ApplyRoPEHeads применяет RoPE к nHeads головам в v (in-place)
	ApplyRoPEHeads(v []float32, nHeads, headDim, pos int, freqBase float32) error

	Close() error
}

// LayerOnGPU возвращает true, если transformer-слой layer должен выполняться на GPU
// layer: 0..totalLayers-1
// ngl: число слоёв для offload (как -ngl в llama.cpp)
func LayerOnGPU(layer, ngl, totalLayers int) bool {
	if ngl <= 0 || layer < 0 {
		return false
	}

	if layer >= totalLayers {
		return false
	}

	return layer < ngl
}
