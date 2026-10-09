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

// formatLlamaFallback picks Llama 3 Instruct or Llama 2 [INST] formatting.
func formatLlamaFallback(messages []Message, opts Options) string {
	if isLlama3Family(opts.Metadata) {
		return formatLlama3(messages, opts)
	}

	return formatLlama2Instruct(messages, opts)
}

// formatLlama3 formats a dialog in Llama 3 Instruct style.
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
	b.WriteString("Cutting Knowledge Date: December 2023\n")
	b.WriteString("Today Date: ")
	b.WriteString(llamaDefaultDate)
	b.WriteString("\n\n")

	system := collectSystemPrompt(messages, opts)
	if HasTools(opts) {
		if system != "" {
			system += "\n\n"
		}

		system += toolsSystemPreamble(Options{
			Tools:      opts.Tools,
			ToolChoice: opts.ToolChoice,
		})
	}

	b.WriteString(system)
	b.WriteString(eot)

	for _, msg := range messages {
		switch msg.Role {
		case "system":
			continue
		case "user", "assistant", "tool":
			content := msg.Content
			if msg.Role == "assistant" {
				content = formatAssistantBody(msg)
			}

			b.WriteString(startHeader)
			b.WriteString(msg.Role)
			b.WriteString(endHeader)
			b.WriteString("\n\n")
			b.WriteString(content)
			b.WriteString(eot)
		}
	}

	b.WriteString(startHeader)
	b.WriteString("assistant")
	b.WriteString(endHeader)
	b.WriteString("\n\n")

	return b.String()
}

// formatLlama2Instruct - Meta Llama 2 chat: [INST] <<SYS>> ... <</SYS>> ... [/INST]
func formatLlama2Instruct(messages []Message, opts Options) string {
	meta := opts.Metadata
	bos := tokenFromVocab(meta, meta.IntOptional("tokenizer.ggml.bos_token_id", 1))
	eos := tokenFromVocab(meta, meta.IntOptional("tokenizer.ggml.eos_token_id", 2))
	if bos == "" {
		bos = "<s>"
	}

	if eos == "" {
		eos = "</s>"
	}

	system := collectSystemPrompt(messages, opts)
	if HasTools(opts) {
		if system != "" {
			system += "\n\n"
		}
		system += toolsSystemPreamble(opts)
	}

	var b strings.Builder
	firstUser := true
	for _, msg := range messages {
		switch msg.Role {
		case "system":
			continue
		case "user":
			b.WriteString(bos)
			b.WriteString("[INST] ")
			if firstUser && system != "" {
				b.WriteString("<<SYS>>\n")
				b.WriteString(system)
				b.WriteString("\n<</SYS>>\n\n")
				system = ""
			}
			b.WriteString(msg.Content)
			b.WriteString(" [/INST]")
			firstUser = false
		case "assistant":
			b.WriteString(" ")
			b.WriteString(formatAssistantBody(msg))
			b.WriteString(" ")
			b.WriteString(eos)
		case "tool":
			b.WriteString(bos)
			b.WriteString("[INST] ")
			b.WriteString(msg.Content)
			b.WriteString(" [/INST]")
			firstUser = false
		}
	}

	if system != "" {
		b.WriteString(bos)
		b.WriteString("[INST] <<SYS>>\n")
		b.WriteString(system)
		b.WriteString("\n<</SYS>>\n\n [/INST]")
	}

	return b.String()
}

// collectSystemPrompt: system from messages; opts.System only when there are no system messages.
func collectSystemPrompt(messages []Message, opts Options) string {
	var parts []string
	for _, msg := range messages {
		if msg.Role == "system" && msg.Content != "" {
			parts = append(parts, msg.Content)
		}
	}

	if len(parts) > 0 {
		return strings.Join(parts, "\n")
	}

	return opts.System
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

// isLlama3Family: header tokens / llama-bpe / large vocab / high rope.freq_base.
func isLlama3Family(meta format.Metadata) bool {
	if !isLlamaArchitecture(meta) {
		return false
	}

	if tok := tokenFromVocab(meta, llamaStartHeaderID); tok != "" && strings.Contains(tok, "start_header") {
		return true
	}

	if pre, err := meta.String("tokenizer.ggml.pre"); err == nil {
		switch pre {
		case "llama-bpe", "llama3":
			return true
		}
	}

	if v, err := format.MetaValue[float32](meta, "llama.rope.freq_base"); err == nil && v >= 100000 {
		return true
	}

	if tokens, err := meta.StringArray("tokenizer.ggml.tokens"); err == nil && len(tokens) >= 100000 {
		return true
	}

	return false
}
