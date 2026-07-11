package chat

import (
	"strings"

	"github.com/magomedcoder/gogguf/pkg/format"
)

const (
	llamaStartHeaderID = 128006
	llamaEndHeaderID   = 128007
	llamaDefaultDate   = "26 Jul 2024"
)

// formatLlama3 форматирует диалог в стиле Llama 3 Instruct (без tools)
func formatLlama3(messages []Message, opts Options) string {
	meta := opts.Metadata
	startHeader := tokenFromVocab(meta, llamaStartHeaderID)
	endHeader := tokenFromVocab(meta, llamaEndHeaderID)
	eot := tokenFromVocab(meta, meta.IntOptional("tokenizer.ggml.eos_token_id", -1))
	bos := tokenFromVocab(meta, meta.IntOptional("tokenizer.ggml.bos_token_id", -1))

	var b strings.Builder
	b.WriteString(bos)

	b.WriteString(startHeader)
	b.WriteString("system")
	b.WriteString(endHeader)
	b.WriteString("\n\n")
	b.WriteString("Cutting Knowledge Date: December 2026\n")
	b.WriteString("Today Date: ")
	b.WriteString(llamaDefaultDate)
	b.WriteString("\n\n")

	system := opts.System
	for _, msg := range messages {
		if msg.Role == "system" {
			if system != "" {
				system += "\n"
			}
			system += msg.Content
		}
	}
	b.WriteString(system)
	b.WriteString(eot)

	for _, msg := range messages {
		switch msg.Role {
		case "system":
			continue
		case "user", "assistant", "tool":
			b.WriteString(startHeader)
			b.WriteString(msg.Role)
			b.WriteString(endHeader)
			b.WriteString("\n\n")
			b.WriteString(msg.Content)
			b.WriteString(eot)
		}
	}

	b.WriteString(startHeader)
	b.WriteString("assistant")
	b.WriteString(endHeader)
	b.WriteString("\n\n")

	return b.String()
}

func tokenFromVocab(meta format.Metadata, id int) string {
	if meta == nil || id < 0 {
		return ""
	}

	tokens, err := meta.StringArray("tokenizer.ggml.tokens")
	if err != nil || id >= len(tokens) {
		return ""
	}

	return tokens[id]
}

func isLlamaArchitecture(meta format.Metadata) bool {
	if meta == nil {
		return false
	}

	arch, err := meta.String("general.architecture")

	return err == nil && arch == "llama"
}
