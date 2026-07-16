package chat

// jinjaContext строит контекст для tokenizer.chat_template
func jinjaContext(messages []Message, addGenerationPrompt bool, opts Options) map[string]any {
	msgs := make([]any, len(messages))
	for i, m := range messages {
		msg := map[string]any{
			"role":    m.Role,
			"content": m.Content,
		}
		if m.Name != "" {
			msg["name"] = m.Name
		}

		if m.ToolCallID != "" {
			msg["tool_call_id"] = m.ToolCallID
		}

		if len(m.ToolCalls) > 0 {
			calls := make([]any, len(m.ToolCalls))
			for j, tc := range m.ToolCalls {
				call := map[string]any{
					"type": tc.Type,
					"function": map[string]any{
						"name":      tc.Function.Name,
						"arguments": toolCallArgumentsForTemplate(tc.Function.Arguments),
					},
				}
				if tc.ID != "" {
					call["id"] = tc.ID
				}

				if call["type"] == "" {
					call["type"] = "function"
				}
				calls[j] = call
			}
			msg["tool_calls"] = calls
		}
		msgs[i] = msg
	}

	ctx := map[string]any{
		"messages":              msgs,
		"add_generation_prompt": addGenerationPrompt,
		"enable_thinking":       ThinkingEnabled(opts),
	}

	if HasTools(opts) {
		tools := make([]any, len(opts.Tools))
		for i, t := range opts.Tools {
			tools[i] = map[string]any{
				"type": t.Type,
				"function": map[string]any{
					"name":        t.Function.Name,
					"description": t.Function.Description,
					"parameters":  t.Function.Parameters,
				},
			}

			if t.Type == "" {
				tools[i].(map[string]any)["type"] = "function"
			}
		}
		ctx["tools"] = tools
		ctx["tool_choice"] = NormalizeToolChoice(opts.ToolChoice)
		ctx["parallel_tool_calls"] = opts.ParallelToolCalls
	}

	return ctx
}
