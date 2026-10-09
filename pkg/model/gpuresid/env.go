package gpuresid

import "os"

// HiddenResidencyEnabled - device-resident hidden state (disable: GGUF_HIDDEN_RESIDENCY=0)
func HiddenResidencyEnabled() bool {
	return envEnabled("GGUF_HIDDEN_RESIDENCY")
}

// QKVResidencyEnabled - fused QKV+RoPE+attention (disable: GGUF_QKV_RESIDENCY=0)
func QKVResidencyEnabled() bool {
	return envEnabled("GGUF_QKV_RESIDENCY")
}

// AttnFFNResidencyEnabled - fused WO+FFN+residual (disable: GGUF_ATTN_FFN_RESIDENCY=0)
func AttnFFNResidencyEnabled() bool {
	return envEnabled("GGUF_ATTN_FFN_RESIDENCY")
}

// envEnabled: on by default; disabled only with explicit 0/false/off
func envEnabled(name string) bool {
	switch os.Getenv(name) {
	case "0", "false", "off":
		return false
	default:
		return true
	}
}
