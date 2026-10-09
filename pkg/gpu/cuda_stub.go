//go:build !cuda

package gpu

// OpenCUDA returns CUDA backend or ErrUnavailable if built without cuda tag
func OpenCUDA() (Backend, error) {
	return nil, ErrUnavailable
}

// OpenCUDADevice: without cuda tag device cannot be selected
func OpenCUDADevice(device int) (Backend, error) {
	if device < 0 {
		return nil, ErrInvalidDevice
	}

	return nil, ErrUnavailable
}

// OpenCUDADevices: without cuda tag multi-GPU unavailable
func OpenCUDADevices(devices []int) (Backend, error) {
	return OpenCUDADevicesSplit(devices, nil)
}

// OpenCUDADevicesSplit: without cuda tag multi-GPU unavailable
func OpenCUDADevicesSplit(devices []int, _ []float64) (Backend, error) {
	if _, err := normalizeDeviceList(devices); err != nil {
		return nil, err
	}

	return nil, ErrUnavailable
}
