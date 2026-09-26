//go:build !darwin

package metal

import "errors"

// ErrNotDarwin: Metal существует только на macOS
var ErrNotDarwin = errors.New("metal: доступен только на macOS (darwin)")

// Open вне macOS всегда возвращает ошибку
func Open() (*Backend, error) {
	return nil, ErrNotDarwin
}
