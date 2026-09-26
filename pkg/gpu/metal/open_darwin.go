//go:build darwin

package metal

// Open возвращает placeholder-устройство Metal: интерфейс реализован полностью, но любой compute-вызов отдаёт ErrUnavailable (kernels ещё не написаны)
func Open() (*Backend, error) {
	return &Backend{name: "Metal (scaffold, macOS)"}, nil
}
