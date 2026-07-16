package chat

// FormatMessages форматирует диалог в chat template
func FormatMessages(messages []Message, opts Options) (string, error) {
	if len(messages) == 0 {
		return "", nil
	}

	if opts.Metadata != nil && HasTemplateMeta(opts.Metadata) {
		prompt, err := Render(opts.Metadata, messages, true, opts)
		if err == nil && prompt != "" {
			return prompt, nil
		}
	}

	if isLlamaArchitecture(opts.Metadata) {
		return formatLlama3(messages, opts), nil
	}

	return formatMessagesFallback(messages, opts), nil
}

func formatMessagesFallback(messages []Message, opts Options) string {
	return renderChatML(messages, true, opts)
}
