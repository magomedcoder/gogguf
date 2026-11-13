//go:build !(metal && darwin && cgo)

package metal

// KernelsImplemented is false outside a darwin + cgo build with -tags metal: no MSL runtime exists, so darwin -ngl>0 is rejected
const KernelsImplemented = false
