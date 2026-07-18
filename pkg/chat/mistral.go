package chat

import (
	"strings"

	"github.com/magomedcoder/gogguf/pkg/format"
)

// formatMistralInstruct форматирует диалог в стиле Mistral Instruct ([INST] ... [/INST])
func formatMistralInstruct(messages []Message, opts Options) string {
	meta := opts.Metadata
	bos := tokenFromVocab(meta, meta.IntOptional("tokenizer.ggml.bos_token_id", 1))
	eos := tokenFromVocab(meta, meta.IntOptional("tokenizer.ggml.eos_token_id", 2))

	var b strings.Builder
	b.WriteString(bos)

	system := opts.System
	for _, msg := range messages {
		if msg.Role == "system" {
			if system != "" {
				system += "\n"
			}

			system += msg.Content
		}
	}
	if HasTools(opts) {
		if system != "" {
			system += "\n\n"
		}
		system += toolsSystemPreamble(opts)
	}

	for _, msg := range messages {
		switch msg.Role {
		case "system":
			continue
		case "user":
			if system != "" {
				b.WriteString("[INST] ")
				b.WriteString(system)
				b.WriteString("\n\n")
				b.WriteString(msg.Content)
				b.WriteString(" [/INST]")
				system = ""
			} else {
				b.WriteString("[INST] ")
				b.WriteString(msg.Content)
				b.WriteString(" [/INST]")
			}
		case "assistant":
			b.WriteString(formatAssistantBody(msg))
			b.WriteString(eos)
		case "tool":
			b.WriteString("[INST] ")
			b.WriteString(msg.Content)
			b.WriteString(" [/INST]")
		}
	}

	if system != "" {
		b.WriteString("[INST] ")
		b.WriteString(system)
		b.WriteString(" [/INST]")
	} else {
		last := lastNonSystem(messages)
		if last == nil || last.Role != "user" {
			b.WriteString("[INST] ")
			b.WriteString(" [/INST]")
		}
	}

	return b.String()
}

func lastNonSystem(messages []Message) *Message {
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role != "system" {
			return &messages[i]
		}
	}

	return nil
}

func isMistralArchitecture(meta format.Metadata) bool {
	return isMistralMeta(meta)
}

func isMistralMeta(meta format.Metadata) bool {
	if meta == nil {
		return false
	}

	arch, err := meta.String("general.architecture")
	if err != nil {
		return false
	}

	if arch == "mistral" {
		return true
	}

	if arch != "llama" {
		return false
	}

	if name, err := meta.String("general.name"); err == nil {
		if strings.Contains(strings.ToLower(name), "mistral") {
			return true
		}
	}

	if _, err := meta.Int("llama.attention.sliding_window"); err == nil {
		return true
	}

	if _, err := meta.Int("mistral.attention.sliding_window"); err == nil {
		return true
	}

	return false
}
