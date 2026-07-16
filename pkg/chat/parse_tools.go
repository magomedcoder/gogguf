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

// ParsedAssistant - разобранный ответ assistant (текст + tool_calls)
type ParsedAssistant struct {
	Content   string
	ToolCalls []ToolCall
}

// HasToolCalls сообщает, есть ли вызовы инструментов
func (p ParsedAssistant) HasToolCalls() bool {
	return len(p.ToolCalls) > 0
}

// FinishReason возвращает finish_reason для OpenAI-совместимого ответа
func (p ParsedAssistant) FinishReason() string {
	if p.HasToolCalls() {
		return "tool_calls"
	}

	return "stop"
}

// ParseAssistantOutput извлекает tool_calls из текста модели (Hermes/Qwen <tool_call> JSON)
func ParseAssistantOutput(text string) ParsedAssistant {
	text = strings.TrimSpace(text)
	if text == "" {
		return ParsedAssistant{}
	}

	matches := toolCallBlockRe.FindAllStringSubmatchIndex(text, -1)
	if len(matches) == 0 {
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
		// если arguments пришли строкой json - разворачиваем
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

func stripThinkBlocks(s string) string {
	s = thinkBlockRe.ReplaceAllString(s, "")
	return strings.TrimSpace(s)
}
