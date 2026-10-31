package server

import (
	"github.com/magomedcoder/gogguf/pkg/runtime"
	"github.com/magomedcoder/gogguf/pkg/runtime/sampler"
)

// maxNewTokens picks OpenAI max_completion_tokens / max_tokens with server default.
func maxNewTokens(maxTokens, maxCompletionTokens int) int {
	if maxCompletionTokens > 0 {
		return maxCompletionTokens
	}

	if maxTokens > 0 {
		return maxTokens
	}

	return 128
}

func samplerConfigFrom(topP, temperature *float64, seed *int64) sampler.Config {
	cfg := sampler.Config{TopP: 1}
	if topP != nil {
		cfg.TopP = float32(*topP)
	}

	if temperature != nil {
		cfg.Temp = float32(*temperature)
	}

	if seed != nil {
		cfg.Seed = uint64(*seed)
	}

	return cfg
}

func baseGenerateParams(maxTokens, maxCompletionTokens int, topP, temperature *float64, seed *int64, frequencyPenalty, presencePenalty *float64, bias map[int]float32, stops []string) runtime.GenerateParams {
	return runtime.GenerateParams{
		MaxTokens:       maxNewTokens(maxTokens, maxCompletionTokens),
		Sampler:         sampler.New(samplerConfigFrom(topP, temperature, seed)),
		RepeatPenalty:   frequencyPenaltyToRepeat(frequencyPenalty),
		RepeatLastN:     64,
		PresencePenalty: presencePenaltyValue(presencePenalty),
		LogitBias:       bias,
		Stop:            stops,
	}
}
