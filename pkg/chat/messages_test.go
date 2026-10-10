package chat

import (
	"strings"
	"testing"
)

func TestFormatMessagesUserOnly(t *testing.T) {
	got, err := FormatMessages([]Message{
		{
			Role:    "user",
			Content: "Hi",
		},
	}, Options{})
	if err != nil {
		t.Fatal(err)
	}

	want := formatUserFallback("Hi", Options{})
	if got != want {
		t.Fatalf("got %q, expected %q", got, want)
	}
}

func TestFormatMessagesWithSystem(t *testing.T) {
	got, err := FormatMessages([]Message{
		{
			Role:    "system",
			Content: "You are very helpful",
		},
		{
			Role:    "user",
			Content: "Hi",
		},
	}, Options{})
	if err != nil {
		t.Fatal(err)
	}

	if !strings.Contains(got, "system") || !strings.Contains(got, "user") {
		t.Fatalf("expected system and user blocks: %q", got)
	}
}
