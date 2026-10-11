package gpuresid

// Optional per-Engine overrides (set via Configure from model.Options).
// nil = enabled (default). No process environment variables.
var (
	hiddenOverride  *bool
	qkvOverride     *bool
	attnFFNOverride *bool
)

// Configure sets residency overrides for subsequent Enabled() checks.
// Pass nil for a field to keep the default (enabled).
func Configure(hidden, qkv, attnFFN *bool) {
	hiddenOverride = hidden
	qkvOverride = qkv
	attnFFNOverride = attnFFN
}

// HiddenResidencyEnabled - device-resident hidden state.
// Options override wins; else enabled.
func HiddenResidencyEnabled() bool {
	if hiddenOverride != nil {
		return *hiddenOverride
	}

	return true
}

// QKVResidencyEnabled - fused QKV+RoPE+attention.
func QKVResidencyEnabled() bool {
	if qkvOverride != nil {
		return *qkvOverride
	}

	return true
}

// AttnFFNResidencyEnabled - fused WO+FFN+residual.
func AttnFFNResidencyEnabled() bool {
	if attnFFNOverride != nil {
		return *attnFFNOverride
	}

	return true
}
