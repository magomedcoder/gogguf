package runtime

import (
	"fmt"
	"io"

	"github.com/magomedcoder/gogguf/pkg/format"
	"github.com/magomedcoder/gogguf/pkg/sampler"
	"github.com/magomedcoder/gogguf/pkg/tokenizer"
)

// GenerateParams - параметры генерации
type GenerateParams struct {
	MaxTokens     int
	Sampler       sampler.Func
	RepeatPenalty float32 // 1.0 = выключено
	RepeatLastN   int     // окно истории для penalty (0 = 64)
	Stop          []string
	OnToken       func(tokenID int) bool
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

// Encode преобразует текст в token IDs (без автоматического BOS)
func (c *Context) Encode(text string) ([]int, error) {
	return c.tok.Encode(text)
}

// EncodeForInference кодирует текст с BOS для архитектур, которым он нужен (llama)
func (c *Context) EncodeForInference(text string) ([]int, error) {
	return c.encodeForInference(text)
}

func (c *Context) encodeForInference(text string) ([]int, error) {
	ids, err := c.tok.Encode(text)
	if err != nil {
		return nil, err
	}

	if !needsBOSPrefix(c.engine.meta, ids) {
		return ids, nil
	}

	bos := c.tok.BOS()
	if bos < 0 {
		return ids, nil
	}

	return append([]int{bos}, ids...), nil
}

func needsBOSPrefix(meta format.Metadata, ids []int) bool {
	if meta == nil {
		return false
	}

	arch, err := meta.String("general.architecture")
	if err != nil {
		return false
	}

	if arch != "llama" && arch != "mistral" {
		return false
	}

	bos := meta.IntOptional("tokenizer.ggml.bos_token_id", -1)
	if bos < 0 {
		return false
	}

	if len(ids) > 0 && ids[0] == bos {
		return false
	}

	return true
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

	if err := sess.GenerateSteps(params); err != nil {
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
