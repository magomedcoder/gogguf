package chat

import (
	"strings"

	"github.com/magomedcoder/gogguf/pkg/format"
)

// Options configures chat template parameters.
type Options struct {
	System            string
	Thinking          *bool // nil = off; true enables thinking mode
	Metadata          format.Metadata
	Tools             []Tool
	ToolChoice        any  // "auto"|"none"|"required" or {"type":"function",...}
	ParallelToolCalls bool // pass parallel_tool_calls into Jinja
}

// ThinkingEnabled reports whether thinking mode is enabled.
func ThinkingEnabled(opts Options) bool {
	if opts.Thinking == nil {
		return false
	}

	return *opts.Thinking
}

// HasTemplate reports whether GGUF contains tokenizer.chat_template.
func HasTemplate(r *format.Reader) bool {
	return HasTemplateMeta(r.Metadata)
}

// HasTemplateMeta checks for a chat template in metadata.
func HasTemplateMeta(m format.Metadata) bool {
	_, err := m.String("tokenizer.chat_template")
	return err == nil
}

// HasToolUseTemplateMeta checks for a tool_use chat template.
func HasToolUseTemplateMeta(m format.Metadata) bool {
	_, err := m.String("tokenizer.chat_template.tool_use")
	return err == nil
}

// SelectChatTemplate picks a Jinja template: tool_use when tools are present, otherwise the default.
func SelectChatTemplate(m format.Metadata, opts Options) (string, error) {
	if HasTools(opts) {
		if tmpl, err := m.String("tokenizer.chat_template.tool_use"); err == nil && tmpl != "" {
			return tmpl, nil
		}
	}

	return m.String("tokenizer.chat_template")
}

// FormatUser wraps the user prompt in a chat template (ChatML/Qwen).
// Uses Render when tokenizer.chat_template is in metadata, otherwise fallback.
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

	if isMistralArchitecture(opts.Metadata) {
		return formatMistralInstruct(msgs, opts), nil
	}

	if isLlamaArchitecture(opts.Metadata) {
		return formatLlamaFallback(msgs, opts), nil
	}

	return formatUserFallback(user, opts), nil
}

// FormatUserMust is like FormatUser but panics on render error (for CLI).
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
