//go:build darwin

package metal

import "fmt"

// Open creates the Metal device on macOS.
// Built with -tags metal (cgo) it compiles the MSL kernels and returns a Backend with a live runtime (Ready() == true); otherwise it returns the placeholder scaffold whose compute calls return ErrUnavailable
func Open() (*Backend, error) {
	rt, name, err := openGPU()
	if err != nil {
		return nil, fmt.Errorf("metal: open: %w", err)
	}

	return &Backend{name: name, rt: rt}, nil
}
