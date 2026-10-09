package llama

// DebugHooks - callbacks for step-by-step forward pass debugging (compare with llama.cpp)
type DebugHooks struct {
	OnEmbed       func(x []float32)
	OnLayer       func(layer int, x []float32)
	OnLayerLogits func(layer int, logits []float32)
	OnLogits      func(logits []float32)
}
