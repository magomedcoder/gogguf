package chat

import "strings"

// FormatMessages форматирует диалог в chat template
func FormatMessages(messages []Message, opts Options) (string, error) {
	if len(messages) == 0 {
		return "", nil
	}

	if opts.Metadata != nil && HasTemplateMeta(opts.Metadata) {
		prompt, err := Render(opts.Metadata, messages, true)
		if err == nil {
			return applyThinkingMode(prompt, ThinkingEnabled(opts), opts.Metadata), nil
		}
	}

	return formatMessagesFallback(messages, opts), nil
}

func formatMessagesFallback(messages []Message, opts Options) string {
	var b strings.Builder

	for _, msg := range messages {
		switch msg.Role {
		case "system",
			"user",
			"assistant",
			"tool":
			writeBlock(&b, msg.Role, msg.Content)
		}
	}

	writeAssistantPrompt(&b, ThinkingEnabled(opts), opts.Metadata)
	return b.String()
}
