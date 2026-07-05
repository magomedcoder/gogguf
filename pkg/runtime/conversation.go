package runtime

import (
	"fmt"
	"io"
)

// Conversation сохраняет KV-cache между multi-turn запросами с инкрементальным prefill
type Conversation struct {
	ctx    *Context
	tokens []int
}

// NewConversation создаёт сессию диалога с переиспользованием KV-cache
func (c *Context) NewConversation() *Conversation {
	return &Conversation{ctx: c}
}

// Reset сбрасывает историю токенов и KV-cache
func (conv *Conversation) Reset() {
	conv.tokens = conv.tokens[:0]
	conv.ctx.engine.Model.ResetCache()
}

// TokenCount возвращает число токенов в cache (промпт + сгенерированные)
func (conv *Conversation) TokenCount() int {
	return len(conv.tokens)
}

// StartGeneration выполняет инкрементальный prefill и возвращает сессию decode
func (conv *Conversation) StartGeneration(prompt string) (*GenerationSession, error) {
	return conv.startGeneration(prompt)
}

// Commit добавляет сгенерированные токены в историю cache
func (conv *Conversation) Commit(sess *GenerationSession) {
	conv.tokens = append(conv.tokens, sess.generated...)
}

func (conv *Conversation) rollback(tokenLen int) {
	saved := append([]int(nil), conv.tokens[:tokenLen]...)
	conv.ctx.engine.Model.ResetCache()
	conv.tokens = conv.tokens[:0]
	if len(saved) == 0 {
		return
	}

	if _, err := conv.ctx.engine.Model.Forward(saved, 0); err != nil {
		return
	}

	conv.tokens = saved
}

// Rollback откатывает cache к предыдущему числу токенов (при ошибке генерации)
func (conv *Conversation) Rollback(tokenLen int) {
	conv.rollback(tokenLen)
}

func (conv *Conversation) Generate(prompt string, params GenerateParams) (string, error) {
	snap := len(conv.tokens)
	sess, err := conv.startGeneration(prompt)
	if err != nil {
		return "", err
	}

	if err := sess.GenerateSteps(params); err != nil {
		conv.rollback(snap)
		return "", err
	}

	conv.Commit(sess)
	return sess.GeneratedText(), nil
}

// GenerateStream как Generate, но пишет токены в w по мере decode
func (conv *Conversation) GenerateStream(prompt string, params GenerateParams, w io.Writer) error {
	params.OnToken = func(id int) bool {
		_, err := io.WriteString(w, conv.ctx.tok.Decode([]int{id}))
		return err == nil
	}

	_, err := conv.Generate(prompt, params)
	return err
}

func (conv *Conversation) startGeneration(prompt string) (*GenerationSession, error) {
	promptTokens, err := conv.ctx.tok.Encode(prompt)
	if err != nil {
		return nil, err
	}

	if ctxLen := conv.ctx.engine.ContextLength(); ctxLen > 0 && len(promptTokens) > ctxLen {
		return nil, fmt.Errorf("runtime: промпт %d токенов превышает context_length=%d", len(promptTokens), ctxLen)
	}

	cached := len(conv.tokens)
	if cached > 0 {
		if shouldResetConversation(cached, conv.tokens, promptTokens) {
			conv.Reset()
			cached = 0
		}
	}

	prevLen := len(conv.tokens)
	newTokens := promptTokens[cached:]
	startPos := cached

	var logits []float32
	if len(newTokens) > 0 {
		logits, err = conv.ctx.engine.Model.Forward(newTokens, startPos)
		if err != nil {
			conv.rollback(prevLen)
			return nil, err
		}

		conv.tokens = append(conv.tokens, newTokens...)
	} else if cached == 0 {
		return nil, fmt.Errorf("runtime: пустой промпт")
	} else {
		last := conv.tokens[len(conv.tokens)-1]
		logits, err = conv.ctx.engine.Model.Forward([]int{last}, len(conv.tokens)-1)
		if err != nil {
			return nil, err
		}
	}

	return &GenerationSession{
		ctx:          conv.ctx,
		promptTokens: promptTokens,
		logits:       logits,
	}, nil
}

func shouldResetConversation(cached int, convTokens, promptTokens []int) bool {
	return cached > len(promptTokens) || !tokensPrefixEqual(convTokens, promptTokens, cached)
}

func tokensPrefixEqual(a, b []int, n int) bool {
	if n > len(a) || n > len(b) {
		return false
	}

	for i := range n {
		if a[i] != b[i] {
			return false
		}
	}

	return true
}
