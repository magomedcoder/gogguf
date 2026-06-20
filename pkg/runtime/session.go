package runtime

import (
	"fmt"

	"github.com/magomedcoder/gguf.go/pkg/sampler"
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

	promptTokens, err := c.tok.Encode(prompt)
	if err != nil {
		return nil, err
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
	if samp == nil {
		samp = sampler.Greedy
	}

	next := samp(s.logits)
	if next < 0 {
		return -1, nil
	}

	eos := s.ctx.tok.EOS()
	if eos >= 0 && next == eos {
		return -1, nil
	}

	s.generated = append(s.generated, next)

	startPos := len(s.promptTokens) + len(s.generated) - 1
	logits, err := s.ctx.engine.Model.Forward([]int{next}, startPos)
	if err != nil {
		return -1, err
	}

	s.logits = logits
	return next, nil
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
func (s *GenerationSession) GenerateSteps(maxTokens int, samp sampler.Func, onToken func(int) bool) error {
	if maxTokens <= 0 {
		maxTokens = 128
	}

	for i := 0; i < maxTokens; i++ {
		id, err := s.DecodeStep(samp)
		if err != nil {
			return err
		}

		if id < 0 {
			return nil
		}

		if onToken != nil && !onToken(id) {
			return nil
		}
	}

	return nil
}

// ErrNoSession возвращается, если сессия не инициализирована
var ErrNoSession = fmt.Errorf("runtime: сессия генерации не начата")
