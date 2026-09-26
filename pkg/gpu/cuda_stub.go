//go:build !cuda

package gpu

// OpenCUDA возвращает CUDA-backend или ErrUnavailable если сборка без тега cuda
func OpenCUDA() (Backend, error) {
	return nil, ErrUnavailable
}

// OpenCUDADevice: без тега cuda устройство выбрать нельзя
func OpenCUDADevice(device int) (Backend, error) {
	if device < 0 {
		return nil, ErrInvalidDevice
	}

	return nil, ErrUnavailable
}

// OpenCUDADevices: без тега cuda multi-GPU недоступен
func OpenCUDADevices(devices []int) (Backend, error) {
	return OpenCUDADevicesSplit(devices, nil)
}

// OpenCUDADevicesSplit: без тега cuda multi-GPU недоступен
func OpenCUDADevicesSplit(devices []int, _ []float64) (Backend, error) {
	if _, err := normalizeDeviceList(devices); err != nil {
		return nil, err
	}

	return nil, ErrUnavailable
}
