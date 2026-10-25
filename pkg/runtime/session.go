package runtime

import (
	"fmt"
	"strings"

	"github.com/magomedcoder/gogguf/pkg/runtime/sampler"
)

// GenerationSession is step-by-step generation after prefill.
type GenerationSession struct {
	ctx          *Context
	promptTokens []int
	generated    []int
	logits       []float32
	logitsBuf    []float32 // reused for penalty copy
	historyBuf   []int     // reused prompt+generated for repeat penalty
	stopBuf      strings.Builder
	stopReason   string // "stop" or "length"; empty until GenerateSteps
	logprobs     []TokenLogprob
}

// StartGeneration encodes the prompt, runs prefill, and returns a decode session.
func (c *Context) StartGeneration(prompt string) (*GenerationSession, error) {
	c.engine.Model.ResetCache()

	promptTokens, err := c.encodeForInference(prompt)
	if err != nil {
		return nil, err
	}

	if ctxLen := c.engine.ContextLength(); ctxLen > 0 && len(promptTokens) > ctxLen {
		return nil, fmt.Errorf("runtime: prompt %d tokens exceeds context_length=%d", len(promptTokens), ctxLen)
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

// DecodeStep selects and decodes one next token.
// Returns tokenID < 0 on EOS or if the sampler returned -1.
func (s *GenerationSession) DecodeStep(samp sampler.Func) (int, error) {
	return s.DecodeStepWith(GenerateParams{Sampler: samp})
}

// DecodeStepWith selects the next token with repeat penalty applied.
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

	if params.Logprobs {
		s.recordLogprob(next, logits, params.TopLogprobs)
	}

	s.generated = append(s.generated, next)
	if len(params.Stop) > 0 {
		s.stopBuf.WriteString(s.ctx.tok.Decode([]int{next}))
	}

	startPos := len(s.promptTokens) + len(s.generated) - 1
	if ctxLen := s.ctx.engine.ContextLength(); ctxLen > 0 && startPos >= ctxLen {
		return -1, fmt.Errorf("runtime: position %d >= context_length=%d", startPos, ctxLen)
	}

	logits, err := s.ctx.engine.Model.Forward([]int{next}, startPos)
	if err != nil {
		return -1, err
	}

	s.logits = logits
	return next, nil
}

func (s *GenerationSession) prepareLogits(params GenerateParams) []float32 {
	needRepeat := params.RepeatPenalty > 0 && params.RepeatPenalty != 1
	needPresence := params.PresencePenalty != 0
	needBias := len(params.LogitBias) > 0
	if !needRepeat && !needPresence && !needBias {
		return s.logits
	}

	if cap(s.logitsBuf) < len(s.logits) {
		s.logitsBuf = make([]float32, len(s.logits))
	} else {
		s.logitsBuf = s.logitsBuf[:len(s.logits)]
	}
	copy(s.logitsBuf, s.logits)

	hist := s.tokenHistoryView()
	if needPresence {
		sampler.ApplyPresencePenalty(s.logitsBuf, hist, params.PresencePenalty)
	}

	if needRepeat {
		lastN := params.RepeatLastN
		if lastN == 0 {
			lastN = 64
		}
		sampler.ApplyRepeatPenalty(s.logitsBuf, hist, params.RepeatPenalty, lastN)
	}

	if needBias {
		sampler.ApplyLogitBias(s.logitsBuf, params.LogitBias)
	}

	return s.logitsBuf
}

func (s *GenerationSession) tokenHistoryView() []int {
	n := len(s.promptTokens) + len(s.generated)
	if cap(s.historyBuf) < n {
		s.historyBuf = make([]int, n)
	} else {
		s.historyBuf = s.historyBuf[:n]
	}

	copy(s.historyBuf, s.promptTokens)
	copy(s.historyBuf[len(s.promptTokens):], s.generated)
	return s.historyBuf
}

// GeneratedTokens returns IDs of generated tokens.
func (s *GenerationSession) GeneratedTokens() []int {
	out := make([]int, len(s.generated))
	copy(out, s.generated)
	return out
}

// GeneratedText returns text of generated tokens.
func (s *GenerationSession) GeneratedText() string {
	return s.ctx.tok.Decode(s.generated)
}

// GeneratedCount returns the number of generated tokens.
func (s *GenerationSession) GeneratedCount() int {
	return len(s.generated)
}

// PromptTokenCount returns prompt length in tokens.
func (s *GenerationSession) PromptTokenCount() int {
	return len(s.promptTokens)
}

// DecodeToken converts a token ID to text.
func (s *GenerationSession) DecodeToken(id int) string {
	return s.ctx.DecodeToken(id)
}

// StopReason is why GenerateSteps returned: "stop" (EOS, stop string, or callback) or "length" (max tokens).
func (s *GenerationSession) StopReason() string {
	if s.stopReason == "" {
		return "stop"
	}

	return s.stopReason
}

// Logprobs returns per-generated-token logprobs when GenerateParams.Logprobs was set.
func (s *GenerationSession) Logprobs() []TokenLogprob {
	return s.logprobs
}

func (s *GenerationSession) recordLogprob(id int, logits []float32, topK int) {
	entry := TokenLogprob{
		TokenID: id,
		Logprob: sampler.SoftmaxLogprob(logits, id),
	}

	if topK > 0 {
		entry.Top = sampler.TopLogprobs(logits, topK)
	}

	s.logprobs = append(s.logprobs, entry)
}

// GenerateSteps runs up to maxTokens decode steps.
func (s *GenerationSession) GenerateSteps(params GenerateParams) error {
	maxTokens := params.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 128
	}

	s.stopBuf.Reset()
	s.stopReason = "stop"
	if params.Logprobs {
		s.logprobs = s.logprobs[:0]
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

		if len(params.Stop) > 0 && hitStopSequence(s.stopBuf.String(), params.Stop) {
			return nil
		}
	}

	s.stopReason = "length"

	return nil
}

func hitStopSequence(text string, stops []string) bool {
	for _, stop := range stops {
		stop = strings.TrimSpace(stop)
		if stop != "" && strings.HasSuffix(text, stop) {
			return true
		}
	}

	return false
}

// ErrNoSession is returned when the generation session was not started.
var ErrNoSession = fmt.Errorf("runtime: generation session not started")
