//go:build integration

package integration

import (
	"encoding/json"
	"fmt"
	"os"
	"testing"

	"github.com/magomedcoder/gogguf"
	"github.com/magomedcoder/gogguf/pkg/chat"
)

func fixtureJSONPaths(name string) []string {
	return []string{
		"../fixtures/" + name,
		"test/fixtures/" + name,
	}
}

func loadFixtureJSON(t *testing.T, skipIfMissing bool, dest any, paths ...string) {
	t.Helper()

	data, err := readFirstFixtureFile(paths...)
	if err != nil {
		if skipIfMissing {
			t.Skip(err.Error())
		}
		t.Fatal(err)
	}

	if err := json.Unmarshal(data, dest); err != nil {
		t.Fatalf("failed to parse fixture: %v", err)
	}
}

func readFirstFixtureFile(paths ...string) ([]byte, error) {
	var lastErr error
	for _, p := range paths {
		data, err := os.ReadFile(p)
		if err == nil {
			return data, nil
		}
		lastErr = err
	}

	return nil, fmt.Errorf("fixture not found (tried %v): %w", paths, lastErr)
}

func promptFromCase(t *testing.T, engine *gogguf.Engine, input, chatUser string) string {
	t.Helper()
	if chatUser == "" {
		return input
	}

	prompt, err := chat.FormatUser(chatUser, chat.Options{
		Metadata: engine.Metadata(),
	})
	if err != nil {
		t.Fatal(err)
	}

	return prompt
}
