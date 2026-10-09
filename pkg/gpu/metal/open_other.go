//go:build !darwin

package metal

import "errors"

// ErrNotDarwin: Metal exists only on macOS
var ErrNotDarwin = errors.New("metal: доступен только на macOS (darwin)")

// Open outside macOS always returns an error
func Open() (*Backend, error) {
	return nil, ErrNotDarwin
}
