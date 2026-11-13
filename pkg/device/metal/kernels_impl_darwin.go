//go:build metal && darwin && cgo

package metal

// KernelsImplemented is true: the MSL kernels in msl.go are compiled by runtime_darwin.go (Backend.Ready still needs an opened device)
const KernelsImplemented = true
