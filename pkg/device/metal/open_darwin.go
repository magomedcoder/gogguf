//go:build darwin

package metal

// Open returns a placeholder Metal device: interface fully implemented but any compute call returns ErrUnavailable (kernels not written yet)
func Open() (*Backend, error) {
	return &Backend{name: "Metal (scaffold, macOS)"}, nil
}
