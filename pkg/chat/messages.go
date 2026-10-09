package chat

// FormatMessages formats a dialog into the chat template.
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
		return formatLlamaFallback(messages, opts), nil
	}

	return formatMessagesFallback(messages, opts), nil
}

func formatMessagesFallback(messages []Message, opts Options) string {
	return renderChatML(messages, true, opts)
}
