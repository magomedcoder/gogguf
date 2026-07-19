package gpu

// CapMaxSeq ограничивает длину GPU KV-cache (0 requested = default 4096)
func CapMaxSeq(contextLength, requested int) int {
	if contextLength <= 0 {
		contextLength = 2048
	}

	cap := 4096
	if requested > 0 {
		cap = requested
	}

	if contextLength < cap {
		return contextLength
	}

	return cap
}
