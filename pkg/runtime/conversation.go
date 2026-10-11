package runtime

import (
	"fmt"
	"io"
)

// Conversation keeps KV-cache across multi-turn turns with incremental prefill.
//
// The model has a single KV-cache: share one Conversation per dialog, or call LoadKVCache when switching dialogs on the same Engine.
// Only one inference at a time per Engine.
// Prefix reuse requires the new prompt to tokenize to an exact prefix of cached tokens; otherwise Reset() runs.
type Conversation struct {
	ctx        *Context
	tokens     []int
	lastLogits []float32 // logits after last Forward / decode; reused on empty delta
}

// NewConversation creates a dialog session that reuses KV-cache.
func (c *Context) NewConversation() *Conversation {
	return &Conversation{ctx: c}
}

// Reset clears token history and KV-cache.
func (conv *Conversation) Reset() {
	conv.tokens = conv.tokens[:0]
	conv.lastLogits = nil
	conv.ctx.engine.Model.ResetCache()
}

// LoadKVCache resets model KV and replays conv.tokens (multi-dialog on one Engine).
func (conv *Conversation) LoadKVCache() error {
	conv.ctx.engine.Model.ResetCache()
	conv.lastLogits = nil
	if len(conv.tokens) == 0 {
		return nil
	}

	saved := append([]int(nil), conv.tokens...)
	logits, err := conv.ctx.engine.Model.Forward(saved, 0)
	if err != nil {
		return err
	}

	conv.storeLogits(logits)

	return nil
}

// TokenCount returns token count in cache (prompt + generated).
func (conv *Conversation) TokenCount() int {
	return len(conv.tokens)
}

// StartGeneration runs incremental prefill and returns a decode session.
func (conv *Conversation) StartGeneration(prompt string) (*GenerationSession, error) {
	return conv.startGeneration(prompt)
}

// Commit appends generated tokens to cache history.
func (conv *Conversation) Commit(sess *GenerationSession) {
	conv.tokens = append(conv.tokens, sess.generated...)
	conv.storeLogits(sess.logits)
}

func (conv *Conversation) storeLogits(logits []float32) {
	if len(logits) == 0 {
		conv.lastLogits = nil
		return
	}

	if len(conv.lastLogits) == len(logits) && logitsPtrEqual(conv.lastLogits, logits) {
		return
	}

	if cap(conv.lastLogits) < len(logits) {
		conv.lastLogits = make([]float32, len(logits))
	} else {
		conv.lastLogits = conv.lastLogits[:len(logits)]
	}

	copy(conv.lastLogits, logits)
}

func logitsPtrEqual(a, b []float32) bool {
	if len(a) == 0 {
		return len(b) == 0
	}

	return &a[0] == &b[0]
}

func (conv *Conversation) rollback(tokenLen int) {
	saved := append([]int(nil), conv.tokens[:tokenLen]...)
	conv.ctx.engine.Model.ResetCache()
	conv.tokens = conv.tokens[:0]
	conv.lastLogits = nil
	if len(saved) == 0 {
		return
	}

	logits, err := conv.ctx.engine.Model.Forward(saved, 0)
	if err != nil {
		return
	}

	conv.tokens = saved
	conv.storeLogits(logits)
}

// Rollback restores cache to a previous token count (on generation error).
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

// GenerateStream is like Generate but writes tokens to w as they are decoded.
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
		return nil, fmt.Errorf("runtime: prompt %d tokens exceeds context_length=%d", len(promptTokens), ctxLen)
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
		conv.storeLogits(logits)
	} else if cached == 0 {
		return nil, fmt.Errorf("runtime: empty prompt")
	} else {
		// Empty delta: reuse cached logits (do not re-Forward last token - that double-appends KV).
		if len(conv.lastLogits) == 0 {
			return nil, fmt.Errorf("runtime: empty delta without cached logits")
		}
		logits = conv.lastLogits
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
