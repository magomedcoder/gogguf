package chat

import (
	"strings"

	"github.com/magomedcoder/gogguf/pkg/format"
)

// Options задаёт параметры chat template
type Options struct {
	System   string
	Thinking *bool // nil - выключено; true включает размышление
	Metadata format.Metadata
}

// ThinkingEnabled возвращает true, если режим размышления включён
func ThinkingEnabled(opts Options) bool {
	if opts.Thinking == nil {
		return false
	}

	return *opts.Thinking
}

// HasTemplate возвращает true, если в GGUF есть tokenizer.chat_template
func HasTemplate(r *format.Reader) bool {
	return HasTemplateMeta(r.Metadata)
}

// HasTemplateMeta проверяет наличие chat template в метаданных
func HasTemplateMeta(m format.Metadata) bool {
	_, err := m.String("tokenizer.chat_template")
	return err == nil
}

// FormatUser оборачивает пользовательский промпт в chat template (ChatML/Qwen)
// При наличии tokenizer.chat_template в метаданных использует Render, иначе fallback
func FormatUser(user string, opts Options) (string, error) {
	msgs := []Message{
		{
			Role:    "user",
			Content: user,
		},
	}
	if opts.System != "" {
		msgs = append([]Message{
			{
				Role:    "system",
				Content: opts.System,
			},
		}, msgs...)
	}

	if opts.Metadata != nil && HasTemplateMeta(opts.Metadata) {
		prompt, err := Render(opts.Metadata, msgs, true, opts)
		if err == nil && prompt != "" {
			return prompt, nil
		}
	}

	if isLlamaArchitecture(opts.Metadata) {
		return formatLlama3(msgs, opts), nil
	}

	return formatUserFallback(user, opts), nil
}

// FormatUserMust как FormatUser, но panic при ошибке рендера (для CLI)
func FormatUserMust(user string, opts Options) string {
	s, err := FormatUser(user, opts)
	if err != nil {
		return formatUserFallback(user, opts)
	}

	return s
}

func formatUserFallback(user string, opts Options) string {
	var b strings.Builder

	if opts.System != "" {
		writeBlock(&b, "system", opts.System)
	}

	writeBlock(&b, "user", user)

	writeAssistantPrompt(&b, ThinkingEnabled(opts), opts.Metadata)

	return b.String()
}

func applyThinkingMode(prompt string, enableThinking bool, meta format.Metadata) string {
	const marker = imStart + "assistant\n"
	idx := strings.LastIndex(prompt, marker)
	if idx < 0 {
		return prompt
	}

	prefix := prompt[:idx+len(marker)]
	if enableThinking {
		return prefix
	}

	return prefix + EmptyThinkingBlock(meta)
}
