//go:build !(metal && darwin && cgo)

package metal

// openGPU without the darwin + cgo + metal build has no runtime: Open (macOS only) returns the scaffold Backend and Ready() stays false
func openGPU() (gpu, string, error) {
	return nil, "Metal (scaffold, macOS)", nil
}
