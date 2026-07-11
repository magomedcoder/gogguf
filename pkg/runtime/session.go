package runtime

import (
	"fmt"

	"github.com/magomedcoder/gogguf/pkg/sampler"
)

// GenerationSession - пошаговая генерация после prefill
type GenerationSession struct {
	ctx          *Context
	promptTokens []int
	generated    []int
	logits       []float32
}

// StartGeneration кодирует промпт, выполняет prefill и возвращает сессию decode
func (c *Context) StartGeneration(prompt string) (*GenerationSession, error) {
	c.engine.Model.ResetCache()

	promptTokens, err := c.encodeForInference(prompt)
	if err != nil {
		return nil, err
	}

	if ctxLen := c.engine.ContextLength(); ctxLen > 0 && len(promptTokens) > ctxLen {
		return nil, fmt.Errorf("runtime: промпт %d токенов превышает context_length=%d", len(promptTokens), ctxLen)
	}

	logits, err := c.engine.Model.Forward(promptTokens, 0)
	if err != nil {
		return nil, err
	}

	return &GenerationSession{
		ctx:          c,
		promptTokens: promptTokens,
		logits:       logits,
	}, nil
}

// DecodeStep выбирает и декодирует один следующий токен
// Возвращает tokenID < 0 при EOS или если sampler вернул -1
func (s *GenerationSession) DecodeStep(samp sampler.Func) (int, error) {
	return s.DecodeStepWith(GenerateParams{Sampler: samp})
}

// DecodeStepWith выбирает следующий токен с учётом repeat penalty
func (s *GenerationSession) DecodeStepWith(params GenerateParams) (int, error) {
	samp := params.Sampler
	if samp == nil {
		samp = sampler.Greedy
	}

	logits := s.prepareLogits(params)
	next := samp(logits)
	if next < 0 {
		return -1, nil
	}

	eos := s.ctx.tok.EOS()
	if eos >= 0 && next == eos {
		return -1, nil
	}

	s.generated = append(s.generated, next)

	startPos := len(s.promptTokens) + len(s.generated) - 1
	if ctxLen := s.ctx.engine.ContextLength(); ctxLen > 0 && startPos >= ctxLen {
		return -1, fmt.Errorf("runtime: позиция %d >= context_length=%d", startPos, ctxLen)
	}

	logits, err := s.ctx.engine.Model.Forward([]int{next}, startPos)
	if err != nil {
		return -1, err
	}

	s.logits = logits
	return next, nil
}

func (s *GenerationSession) prepareLogits(params GenerateParams) []float32 {
	logits := make([]float32, len(s.logits))
	copy(logits, s.logits)

	if params.RepeatPenalty > 0 && params.RepeatPenalty != 1 {
		lastN := params.RepeatLastN
		if lastN == 0 {
			lastN = 64
		}
		sampler.ApplyRepeatPenalty(logits, s.tokenHistory(), params.RepeatPenalty, lastN)
	}

	return logits
}

func (s *GenerationSession) tokenHistory() []int {
	out := make([]int, len(s.promptTokens)+len(s.generated))
	copy(out, s.promptTokens)
	copy(out[len(s.promptTokens):], s.generated)
	return out
}

// GeneratedTokens возвращает ID сгенерированных токенов
func (s *GenerationSession) GeneratedTokens() []int {
	out := make([]int, len(s.generated))
	copy(out, s.generated)
	return out
}

// GeneratedText возвращает текст сгенерированных токенов
func (s *GenerationSession) GeneratedText() string {
	return s.ctx.tok.Decode(s.generated)
}

// GeneratedCount возвращает число сгенерированных токенов
func (s *GenerationSession) GeneratedCount() int {
	return len(s.generated)
}

// PromptTokenCount возвращает длину промпта в токенах
func (s *GenerationSession) PromptTokenCount() int {
	return len(s.promptTokens)
}

// DecodeToken преобразует token ID в текст
func (s *GenerationSession) DecodeToken(id int) string {
	return s.ctx.DecodeToken(id)
}

// GenerateSteps выполняет до maxTokens шагов decode
func (s *GenerationSession) GenerateSteps(params GenerateParams) error {
	maxTokens := params.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 128
	}

	for i := 0; i < maxTokens; i++ {
		id, err := s.DecodeStepWith(params)
		if err != nil {
			return err
		}

		if id < 0 {
			return nil
		}

		if params.OnToken != nil && !params.OnToken(id) {
			return nil
		}
	}

	return nil
}

// ErrNoSession возвращается, если сессия не инициализирована
var ErrNoSession = fmt.Errorf("runtime: сессия генерации не начата")
