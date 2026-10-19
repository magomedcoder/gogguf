package chat

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
)

var (
	toolCallBlockRe = regexp.MustCompile(`(?s)<tool_call>\s*(.*?)\s*</tool_call>`)
	thinkBlockRe    = regexp.MustCompile(`(?s)<think>.*?</think>`)
)

// ParsedAssistant is a parsed assistant reply (text + tool_calls).
type ParsedAssistant struct {
	Content   string
	ToolCalls []ToolCall
}

// HasToolCalls reports whether tool calls are present.
func (p ParsedAssistant) HasToolCalls() bool {
	return len(p.ToolCalls) > 0
}

// FinishReason returns finish_reason for an OpenAI-compatible response.
func (p ParsedAssistant) FinishReason() string {
	if p.HasToolCalls() {
		return "tool_calls"
	}

	return "stop"
}

// ParseAssistantOutput extracts tool_calls from model text (Hermes/Qwen <tool_call> JSON).
func ParseAssistantOutput(text string) ParsedAssistant {
	text = strings.TrimSpace(text)
	if text == "" {
		return ParsedAssistant{}
	}

	matches := toolCallBlockRe.FindAllStringSubmatchIndex(text, -1)
	if len(matches) == 0 {
		if calls, content, ok := parseLooseJSONToolCalls(text); ok {
			return ParsedAssistant{Content: content, ToolCalls: calls}
		}

		return ParsedAssistant{
			Content: text,
		}
	}

	var calls []ToolCall
	var contentParts []string
	last := 0
	for i, m := range matches {
		if m[0] > last {
			contentParts = append(contentParts, text[last:m[0]])
		}

		raw := strings.TrimSpace(text[m[2]:m[3]])
		if tc, ok := parseToolCallJSON(raw, i); ok {
			calls = append(calls, tc)
		}

		last = m[1]
	}
	if last < len(text) {
		contentParts = append(contentParts, text[last:])
	}

	content := stripThinkBlocks(strings.TrimSpace(strings.Join(contentParts, "")))
	return ParsedAssistant{
		Content:   content,
		ToolCalls: calls,
	}
}

func parseToolCallJSON(raw string, index int) (ToolCall, bool) {
	var payload struct {
		Name       string          `json:"name"`
		Arguments  json.RawMessage `json:"arguments"`
		Parameters json.RawMessage `json:"parameters"`
	}
	if err := json.Unmarshal([]byte(raw), &payload); err != nil || payload.Name == "" {
		return ToolCall{}, false
	}

	argsRaw := payload.Arguments
	if len(argsRaw) == 0 {
		argsRaw = payload.Parameters
	}

	args := "{}"
	if len(argsRaw) > 0 {
		args = string(argsRaw)
		// if arguments arrived as a JSON string, unwrap them
		var asString string
		if err := json.Unmarshal(argsRaw, &asString); err == nil {
			args = asString
		}
	}

	return ToolCall{
		ID:   fmt.Sprintf("call_%d", index),
		Type: "function",
		Function: FunctionCall{
			Name:      payload.Name,
			Arguments: args,
		},
	}, true
}

// parseLooseJSONToolCalls accepts Llama-style calls: optional <|python_tag|> plus a JSON object or array with name and arguments/parameters.
// Surrounding prose stays in content.
func parseLooseJSONToolCalls(text string) ([]ToolCall, string, bool) {
	body := strings.TrimSpace(text)
	body = strings.TrimPrefix(body, "<|python_tag|>")
	body = strings.TrimSpace(body)
	if body == "" {
		return nil, "", false
	}

	startObj := strings.IndexByte(body, '{')
	startArr := strings.IndexByte(body, '[')
	start := -1
	if startObj >= 0 && (startArr < 0 || startObj < startArr) {
		start = startObj
	} else if startArr >= 0 {
		start = startArr
	}

	if start < 0 {
		return nil, "", false
	}

	raw := strings.TrimSpace(body[start:])
	var calls []ToolCall
	if raw[0] == '[' {
		var items []json.RawMessage
		if err := json.Unmarshal([]byte(raw), &items); err != nil || len(items) == 0 {
			return nil, "", false
		}

		for i, item := range items {
			tc, ok := parseToolCallJSON(string(item), i)
			if !ok {
				return nil, "", false
			}

			calls = append(calls, tc)
		}
	} else {
		dec := json.NewDecoder(strings.NewReader(raw))
		var obj json.RawMessage
		if err := dec.Decode(&obj); err != nil {
			return nil, "", false
		}

		tc, ok := parseToolCallJSON(string(obj), 0)
		if !ok {
			return nil, "", false
		}
		calls = []ToolCall{tc}
	}

	content := stripThinkBlocks(strings.TrimSpace(body[:start]))
	return calls, content, true
}

func stripThinkBlocks(s string) string {
	s = thinkBlockRe.ReplaceAllString(s, "")
	return strings.TrimSpace(s)
}
