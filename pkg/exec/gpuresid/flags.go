package gpuresid

// DecodeFlags tracks host/device mirror coherence during residency decode
type DecodeFlags struct {
	ResidDevice bool
	LogitsOnGPU bool
	GPUKVStale  bool
	CPUKVHollow bool
}

// Clear resets all residency decode flags to their zero values
func (f *DecodeFlags) Clear() {
	*f = DecodeFlags{}
}
