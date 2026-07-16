package chat

import (
	"encoding/json"
	"fmt"
	"strings"
)

// Tool - описание инструмента в стиле OpenAI
type Tool struct {
	Type     string       `json:"type"`
	Function ToolFunction `json:"function"`
}

// ToolFunction - схема функции инструмента
type ToolFunction struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Parameters  any    `json:"parameters,omitempty"`
}

// ToolCall - вызов инструмента в ответе assistant
type ToolCall struct {
	ID       string       `json:"id,omitempty"`
	Type     string       `json:"type"`
	Function FunctionCall `json:"function"`
}

// FunctionCall - имя и аргументы (json-строка, как в OpenAI API)
type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// ToolChoiceAuto / ToolChoiceNone / ToolChoiceRequired - стандартные значения tool_choice
const (
	ToolChoiceAuto     = "auto"
	ToolChoiceNone     = "none"
	ToolChoiceRequired = "required"
)

// NormalizeToolChoice приводит tool_choice к значению для Jinja (string или object)
func NormalizeToolChoice(choice any) any {
	if choice == nil {
		return ToolChoiceAuto
	}

	switch v := choice.(type) {
	case string:
		if v == "" {
			return ToolChoiceAuto
		}
		return v
	default:
		return choice
	}
}

// HasTools возвращает true, если в Options заданы инструменты
func HasTools(opts Options) bool {
	return len(opts.Tools) > 0
}

func toolCallArgumentsForTemplate(args string) any {
	args = strings.TrimSpace(args)
	if args == "" {
		return map[string]any{}
	}

	var obj any
	if err := json.Unmarshal([]byte(args), &obj); err == nil {
		return obj
	}

	return args
}

func formatToolCallXML(tc ToolCall) string {
	args := toolCallArgumentsForTemplate(tc.Function.Arguments)
	payload := map[string]any{
		"name":      tc.Function.Name,
		"arguments": args,
	}

	b, err := json.Marshal(payload)
	if err != nil {
		b = fmt.Appendf(nil, `{"name":%q,"arguments":{}}`, tc.Function.Name)
	}

	return "<tool_call>\n" + string(b) + "\n</tool_call>"
}

func formatAssistantBody(msg Message) string {
	if len(msg.ToolCalls) == 0 {
		return msg.Content
	}

	body := msg.Content
	for _, tc := range msg.ToolCalls {
		if body != "" && !strings.HasSuffix(body, "\n") {
			body += "\n"
		}

		body += formatToolCallXML(tc)
	}

	return body
}
