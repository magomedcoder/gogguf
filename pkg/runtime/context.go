package runtime

import (
	"fmt"
	"io"

	"github.com/magomedcoder/gguf.go/pkg/sampler"
	"github.com/magomedcoder/gguf.go/pkg/tokenizer"
)

// GenerateParams - параметры генерации
type GenerateParams struct {
	MaxTokens int
	Sampler   sampler.Func
	OnToken   func(tokenID int) bool
}

// Context выполняет prefill и autoregressive decode
type Context struct {
	engine *Engine
	tok    *tokenizer.Tokenizer
}

// NewContext создаёт inference-контекст
func (e *Engine) NewContext() (*Context, error) {
	if e.tok == nil {
		return nil, fmt.Errorf("runtime: tokenizer не загружен")
	}

	return &Context{
		engine: e,
		tok:    e.tok,
	}, nil
}

// Encode преобразует текст в token IDs
func (c *Context) Encode(text string) ([]int, error) {
	return c.tok.Encode(text)
}

// DecodeToken преобразует один token ID в текст
func (c *Context) DecodeToken(id int) string {
	return c.tok.Decode([]int{id})
}

// Generate выполняет prefill + decode и возвращает сгенерированный текст
func (c *Context) Generate(prompt string, params GenerateParams) (string, error) {
	if params.Sampler == nil {
		params.Sampler = sampler.Greedy
	}

	if params.MaxTokens <= 0 {
		params.MaxTokens = 128
	}

	sess, err := c.StartGeneration(prompt)
	if err != nil {
		return "", err
	}

	if err := sess.GenerateSteps(params.MaxTokens, params.Sampler, params.OnToken); err != nil {
		return "", err
	}

	return sess.GeneratedText(), nil
}

// GenerateStream пишет сгенерированные token IDs в w по мере decode
func (c *Context) GenerateStream(prompt string, params GenerateParams, w io.Writer) error {
	params.OnToken = func(id int) bool {
		_, err := io.WriteString(w, c.tok.Decode([]int{id}))
		return err == nil
	}

	_, err := c.Generate(prompt, params)
	return err
}
