package gpuresid

import "github.com/magomedcoder/gogguf/pkg/device"

// CanEnableResidency reports whether full hidden residency is allowed for this backend and layer quants.
// Callers still apply arch-specific gates (MoE, SWA, softcap, ExpertCount, ngl coverage).
func CanEnableResidency(g device.Backend, layerQuantOK []bool) bool {
	if g == nil || !g.HiddenResident() {
		return false
	}
	if !HiddenResidencyEnabled() || !QKVResidencyEnabled() || !AttnFFNResidencyEnabled() {
		return false
	}
	for _, ok := range layerQuantOK {
		if !ok {
			return false
		}
	}
	return true
}

// AllLayersOffloaded is true when ngl covers every transformer layer.
func AllLayersOffloaded(ngl, numLayers int) bool {
	return ngl > 0 && numLayers > 0 && ngl >= numLayers
}
