package chat

// jinjaContext строит контекст для tokenizer.chat_template
func jinjaContext(messages []Message, addGenerationPrompt bool, opts Options) map[string]any {
	msgs := make([]any, len(messages))
	for i, m := range messages {
		msgs[i] = map[string]any{
			"role":    m.Role,
			"content": m.Content,
		}
	}

	ctx := map[string]any{
		"messages":              msgs,
		"add_generation_prompt": addGenerationPrompt,
		"enable_thinking":       ThinkingEnabled(opts),
	}
	return ctx
}
