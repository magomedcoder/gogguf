package chat

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/magomedcoder/gogguf/pkg/chat/jinja"
	"github.com/magomedcoder/gogguf/pkg/format"
)

const (
	imStart = "<|im_start|>"
	imEnd   = "<|im_end|>"

	// Default Qwen3 thinking tokens; taken from vocab when metadata is present.
	defaultThinkingOpen  = " "
	defaultThinkingClose = " "
)

// Render formats a dialog via Jinja2 tokenizer.chat_template from metadata.
func Render(meta format.Metadata, messages []Message, addGenerationPrompt bool, opts Options) (string, error) {
	tmpl, err := SelectChatTemplate(meta, opts)
	if err != nil {
		return "", fmt.Errorf("chat: tokenizer.chat_template не найден")
	}

	ctx := jinjaContext(messages, addGenerationPrompt, opts)
	out, err := jinja.Render(tmpl, ctx)
	if err != nil {
		if isMistralArchitecture(meta) {
			return formatMistralInstruct(messages, opts), nil
		}

		if isLlamaArchitecture(meta) {
			return formatLlamaFallback(messages, opts), nil
		}

		return renderChatML(messages, addGenerationPrompt, opts), nil
	}

	return out, nil
}

// RenderFromReader renders the prompt from a GGUF reader.
func RenderFromReader(r *format.Reader, messages []Message, addGenerationPrompt bool, opts Options) (string, error) {
	return Render(r.Metadata, messages, addGenerationPrompt, opts)
}

func renderChatML(messages []Message, addGenerationPrompt bool, opts Options) string {
	var b strings.Builder

	if HasTools(opts) {
		writeBlock(&b, "system", toolsSystemPreamble(opts))
	}

	for _, msg := range messages {
		switch msg.Role {
		case "system", "user", "assistant", "tool":
			content := msg.Content
			if msg.Role == "assistant" {
				content = formatAssistantBody(msg)
			}

			if msg.Role == "tool" && msg.ToolCallID != "" && content != "" {
				// ChatML: tool_call_id may be placed at the start of the block.
				content = "tool_call_id=" + msg.ToolCallID + "\n" + content
			}

			writeBlock(&b, msg.Role, content)
		}
	}

	if addGenerationPrompt {
		writeAssistantPrompt(&b, ThinkingEnabled(opts), opts.Metadata)
	}

	return b.String()
}

func toolsSystemPreamble(opts Options) string {
	var b strings.Builder
	// English tool preamble for ChatML fallback.
	b.WriteString("You may call one or more functions to assist with the user query.\n")
	b.WriteString("Available tools:\n")

	for _, t := range opts.Tools {
		name := t.Function.Name
		desc := t.Function.Description
		b.WriteString("- ")
		b.WriteString(name)
		if desc != "" {
			b.WriteString(": ")
			b.WriteString(desc)
		}

		b.WriteByte('\n')
		if t.Function.Parameters != nil {
			raw, err := json.Marshal(t.Function.Parameters)
			if err == nil {
				b.WriteString("  parameters: ")
				b.Write(raw)
				b.WriteByte('\n')
			}
		}
	}

	b.WriteString("When calling a tool, use:\n<tool_call>\n{\"name\":\"...\",\"arguments\":{...}}\n</tool_call>")
	if opts.System != "" {
		b.WriteString("\n\n")
		b.WriteString(opts.System)
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
