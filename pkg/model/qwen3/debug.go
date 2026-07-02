package qwen3

// DebugHooks - колбэки для пошаговой отладки forward pass (сверка с llama.cpp)
type DebugHooks struct {
	OnEmbed  func(x []float32)
	OnLayer  func(layer int, x []float32)
	OnLogits func(logits []float32)
}
