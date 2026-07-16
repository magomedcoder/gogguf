package chat

import (
	"strings"
	"testing"
)

func TestParseAssistantOutputToolCall(t *testing.T) {
	text := `I'll check the weather.
<tool_call>
{"name":"get_weather","arguments":{"location":"Moscow"}}
</tool_call>`

	got := ParseAssistantOutput(text)
	if !got.HasToolCalls() {
		t.Fatal("ожидали tool_calls")
	}

	if got.FinishReason() != "tool_calls" {
		t.Fatalf("finish_reason = %q", got.FinishReason())
	}

	if got.Content != "I'll check the weather." {
		t.Fatalf("content = %q", got.Content)
	}

	if len(got.ToolCalls) != 1 {
		t.Fatalf("len(tool_calls) = %d", len(got.ToolCalls))
	}

	tc := got.ToolCalls[0]
	if tc.Function.Name != "get_weather" {
		t.Fatalf("name = %q", tc.Function.Name)
	}

	if !strings.Contains(tc.Function.Arguments, "Moscow") {
		t.Fatalf("arguments = %q", tc.Function.Arguments)
	}

	if tc.Type != "function" || tc.ID == "" {
		t.Fatalf("tool_call = %+v", tc)
	}
}

func TestParseAssistantOutputNoTools(t *testing.T) {
	text := "Hello!"
	got := ParseAssistantOutput(text)
	if got.HasToolCalls() {
		t.Fatal("не ожидали tool_calls")
	}

	if got.Content != text {
		t.Fatalf("content = %q", got.Content)
	}

	if got.FinishReason() != "stop" {
		t.Fatalf("finish_reason = %q", got.FinishReason())
	}
}

func TestParseAssistantOutputArgumentsAsString(t *testing.T) {
	text := `<tool_call>
{"name":"echo","arguments":"{\"x\":1}"}
</tool_call>`
	got := ParseAssistantOutput(text)
	if len(got.ToolCalls) != 1 {
		t.Fatalf("len = %d", len(got.ToolCalls))
	}

	if got.ToolCalls[0].Function.Arguments != `{"x":1}` {
		t.Fatalf("arguments = %q", got.ToolCalls[0].Function.Arguments)
	}
}

func TestFormatMessagesWithToolsFallback(t *testing.T) {
	tools := []Tool{{
		Type: "function",
		Function: ToolFunction{
			Name:        "get_weather",
			Description: "Get weather",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"location": map[string]any{"type": "string"},
				},
			},
		},
	}}

	got, err := FormatMessages([]Message{
		{
			Role:    "user",
			Content: "Weather in Moscow?",
		},
	},
		Options{
			Tools: tools,
		},
	)
	if err != nil {
		t.Fatal(err)
	}

	if !strings.Contains(got, "get_weather") {
		t.Fatalf("ожидали имя инструмента в промпте: %q", got)
	}

	if !strings.Contains(got, "<tool_call>") {
		t.Fatalf("ожидали инструкцию tool_call: %q", got)
	}

	if !strings.Contains(got, "Weather in Moscow?") {
		t.Fatalf("ожидали user message: %q", got)
	}
}

func TestJinjaContextIncludesTools(t *testing.T) {
	ctx := jinjaContext([]Message{
		{
			Role:    "assistant",
			Content: "",
			ToolCalls: []ToolCall{
				{
					ID:   "call_0",
					Type: "function",
					Function: FunctionCall{
						Name:      "get_weather",
						Arguments: `{"location":"Moscow"}`,
					},
				},
			},
		},
		{
			Role:       "tool",
			Content:    "sunny",
			ToolCallID: "call_0",
		},
	}, true, Options{
		Tools: []Tool{
			{
				Type:     "function",
				Function: ToolFunction{Name: "get_weather"},
			},
		},
		ToolChoice:        "auto",
		ParallelToolCalls: true,
	})

	if _, ok := ctx["tools"]; !ok {
		t.Fatal("ожидали tools в контексте")
	}

	if ctx["tool_choice"] != "auto" {
		t.Fatalf("tool_choice = %v", ctx["tool_choice"])
	}

	if ctx["parallel_tool_calls"] != true {
		t.Fatal("ожидали parallel_tool_calls=true")
	}

	msgs := ctx["messages"].([]any)
	asst := msgs[0].(map[string]any)
	if _, ok := asst["tool_calls"]; !ok {
		t.Fatal("ожидали tool_calls в assistant message")
	}

	tool := msgs[1].(map[string]any)
	if tool["tool_call_id"] != "call_0" {
		t.Fatalf("tool_call_id = %v", tool["tool_call_id"])
	}
}

func TestSelectChatTemplatePrefersToolUse(t *testing.T) {
	meta := map[string]any{
		"tokenizer.chat_template":          "BASE",
		"tokenizer.chat_template.tool_use": "TOOLS",
	}
	tmpl, err := SelectChatTemplate(meta, Options{
		Tools: []Tool{
			{
				Type: "function",
				Function: ToolFunction{
					Name: "x",
				},
			},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	if tmpl != "TOOLS" {
		t.Fatalf("tmpl = %q, ожидали TOOLS", tmpl)
	}

	tmpl, err = SelectChatTemplate(meta, Options{})
	if err != nil {
		t.Fatal(err)
	}

	if tmpl != "BASE" {
		t.Fatalf("tmpl = %q, ожидали BASE", tmpl)
	}
}

func TestRenderJinjaWithTools(t *testing.T) {
	meta := map[string]any{
		"tokenizer.chat_template": `{%- if tools %}TOOLS:{{ tools|length }}{%- endif %}{% for m in messages %}{{ m.role }}:{{ m.content }}{% endfor %}`,
	}
	out, err := Render(meta, []Message{
		{
			Role:    "user",
			Content: "hi",
		},
	},
		true,
		Options{
			Tools: []Tool{
				{
					Type: "function",
					Function: ToolFunction{
						Name: "x",
					},
				},
			},
		},
	)
	if err != nil {
		t.Fatal(err)
	}

	if !strings.Contains(out, "TOOLS:1") {
		t.Fatalf("out = %q", out)
	}

	if !strings.Contains(out, "user:hi") {
		t.Fatalf("out = %q", out)
	}
}
