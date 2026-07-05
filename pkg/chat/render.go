package chat

import (
	"fmt"
	"strings"

	"github.com/magomedcoder/gogguf/pkg/chat/jinja"
	"github.com/magomedcoder/gogguf/pkg/format"
)

const (
	imStart = "<|im_start|>"
	imEnd   = "<|im_end|>"

	// Стандартные Qwen3 thinking-токены - при наличии metadata берутся из vocab
	defaultThinkingOpen  = " "
	defaultThinkingClose = " "
)

// Render форматирует диалог через Jinja2 tokenizer.chat_template из метаданных
func Render(meta format.Metadata, messages []Message, addGenerationPrompt bool, opts Options) (string, error) {
	tmpl, err := meta.String("tokenizer.chat_template")
	if err != nil {
		return "", fmt.Errorf("chat: tokenizer.chat_template не найден")
	}

	ctx := jinjaContext(messages, addGenerationPrompt, opts)
	out, err := jinja.Render(tmpl, ctx)
	if err != nil {
		return renderChatML(messages, addGenerationPrompt, opts), nil
	}

	return out, nil
}

// RenderFromReader рендерит prompt из GGUF reader
func RenderFromReader(r *format.Reader, messages []Message, addGenerationPrompt bool, opts Options) (string, error) {
	return Render(r.Metadata, messages, addGenerationPrompt, opts)
}

func renderChatML(messages []Message, addGenerationPrompt bool, opts Options) string {
	var b strings.Builder

	for _, msg := range messages {
		switch msg.Role {
		case "system", "user", "assistant", "tool":
			writeBlock(&b, msg.Role, msg.Content)
		}
	}

	if addGenerationPrompt {
		writeAssistantPrompt(&b, ThinkingEnabled(opts), opts.Metadata)
	}

	return b.String()
}

func writeAssistantPrompt(b *strings.Builder, enableThinking bool, meta format.Metadata) {
	b.WriteString(imStart)
	b.WriteString("assistant\n")
	if !enableThinking {
		b.WriteString(EmptyThinkingBlock(meta))
	}
}

func writeBlock(b *strings.Builder, role, content string) {
	b.WriteString(imStart)
	b.WriteString(role)
	b.WriteByte('\n')
	b.WriteString(content)
	b.WriteString(imEnd)
	b.WriteByte('\n')
}
