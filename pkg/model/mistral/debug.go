package mistral

// DebugHooks - колбэки для пошаговой отладки forward pass
type DebugHooks struct {
	OnEmbed       func(x []float32)
	OnLayer       func(layer int, x []float32)
	OnLayerLogits func(layer int, logits []float32)
	OnLogits      func(logits []float32)
}
