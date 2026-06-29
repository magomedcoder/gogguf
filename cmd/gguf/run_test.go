package main

import (
	"strings"
	"testing"

	chattmpl "github.com/magomedcoder/gguf.go/pkg/chat"
)

func TestFormatChatHistoryIncludesRoles(t *testing.T) {
	messages := []chattmpl.Message{
		{
			Role:    "user",
			Content: "Привет",
		},
		{
			Role:    "assistant",
			Content: "Здравствуйте",
		},
		{
			Role:    "user",
			Content: "Как дела?",
		},
	}

	prompt, err := formatChatHistory(nil, messages, nil)
	if err != nil {
		t.Fatal(err)
	}

	for _, want := range []string{"user", "assistant", "Привет", "Здравствуйте", "Как дела?"} {
		if !strings.Contains(prompt, want) {
			t.Fatalf("промпт не содержит %q:\n%s", want, prompt)
		}
	}
}
