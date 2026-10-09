package chat

import (
	"encoding/json"
	"fmt"
	"strings"
)

// Tool describes a tool in OpenAI style.
type Tool struct {
	Type     string       `json:"type"`
	Function ToolFunction `json:"function"`
}

// ToolFunction is the schema for a tool function.
type ToolFunction struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Parameters  any    `json:"parameters,omitempty"`
}

// ToolCall is a tool invocation in an assistant reply.
type ToolCall struct {
	ID       string       `json:"id,omitempty"`
	Type     string       `json:"type"`
	Function FunctionCall `json:"function"`
}

// FunctionCall holds name and arguments (JSON string, as in OpenAI API).
type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// ToolChoiceAuto / ToolChoiceNone / ToolChoiceRequired are standard tool_choice values.
const (
	ToolChoiceAuto     = "auto"
	ToolChoiceNone     = "none"
	ToolChoiceRequired = "required"
)

// NormalizeToolChoice converts tool_choice to a Jinja value (string or object).
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

// HasTools reports whether Options defines tools.
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
