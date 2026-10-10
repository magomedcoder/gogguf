package main

import (
	"strings"
	"testing"

	chattmpl "github.com/magomedcoder/gogguf/pkg/chat"
)

func TestFormatChatHistoryIncludesRoles(t *testing.T) {
	messages := []chattmpl.Message{
		{
			Role:    "user",
			Content: "Hello",
		},
		{
			Role:    "assistant",
			Content: "Hello there",
		},
		{
			Role:    "user",
			Content: "How are you?",
		},
	}

	prompt, err := formatChatHistory(nil, messages, nil)
	if err != nil {
		t.Fatal(err)
	}

	for _, want := range []string{"user", "assistant", "Hello", "Hello there", "How are you?"} {
		if !strings.Contains(prompt, want) {
			t.Fatalf("prompt does not contain %q:\n%s", want, prompt)
		}
	}
}
