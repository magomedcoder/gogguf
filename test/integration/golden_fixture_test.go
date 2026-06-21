//go:build integration

package integration

import (
	"encoding/json"
	"math"
	"os"
	"testing"

	"github.com/magomedcoder/gguf.go"
	"github.com/magomedcoder/gguf.go/pkg/chat"
)

type goldenFile struct {
	Model string       `json:"model"`
	Cases []goldenCase `json:"cases"`
}

type goldenCase struct {
	Name         string        `json:"name"`
	Input        string        `json:"input,omitempty"`
	ChatUser     string        `json:"chat_user,omitempty"`
	Encode       []int         `json:"encode,omitempty"`
	GreedyNext   int           `json:"greedy_next,omitempty"`
	GreedyTokens []int         `json:"greedy_tokens,omitempty"`
	MaxTokens    int           `json:"max_tokens,omitempty"`
	TopLogits    []goldenLogit `json:"top_logits,omitempty"`
}

type goldenLogit struct {
	ID    int     `json:"id"`
	Logit float32 `json:"logit"`
}

func loadGolden(t *testing.T) goldenFile {
	t.Helper()

	paths := []string{
		"../fixtures/qwen3_golden.json",
		"test/fixtures/qwen3_golden.json",
	}
	if p := os.Getenv("GGUF_GOLDEN"); p != "" {
		paths = []string{p}
	}

	var data []byte
	var err error
	for _, p := range paths {
		data, err = os.ReadFile(p)
		if err == nil {
			break
		}
	}
	if data == nil {
		t.Skip("golden fixture не найден")
	}

	var gf goldenFile
	if err := json.Unmarshal(data, &gf); err != nil {
		t.Fatalf("не удалось разобрать fixture: %v", err)
	}
	return gf
}

func TestGoldenFixture(t *testing.T) {
	gf := loadGolden(t)
	engine, err := gguf.Load(modelPath(t), gguf.LoadOptions{})
	if err != nil {
		t.Fatalf("не удалось загрузить модель: %v", err)
	}

	for _, tc := range gf.Cases {
		t.Run(tc.Name, func(t *testing.T) {
			switch {
			case tc.Encode != nil:
				ids, err := engine.Tokenizer().Encode(tc.Input)
				if err != nil {
					t.Fatal(err)
				}

				if len(ids) != len(tc.Encode) {
					t.Fatalf("Encode(%q) = %v, ожидали %v", tc.Input, ids, tc.Encode)
				}

				for i := range ids {
					if ids[i] != tc.Encode[i] {
						t.Fatalf("Encode(%q)[%d] = %d, ожидали %d", tc.Input, i, ids[i], tc.Encode[i])
					}
				}

			case tc.ChatUser != "" && len(tc.GreedyTokens) > 0:
				prompt, err := chat.FormatUser(tc.ChatUser, chat.Options{
					Metadata: engine.Metadata(),
				})
				if err != nil {
					t.Fatal(err)
				}

				ctx, err := engine.NewContext()
				if err != nil {
					t.Fatal(err)
				}

				sess, err := ctx.StartGeneration(prompt)
				if err != nil {
					t.Fatal(err)
				}

				maxTok := tc.MaxTokens
				if maxTok <= 0 {
					maxTok = len(tc.GreedyTokens)
				}

				if err := sess.GenerateSteps(maxTok, gguf.Greedy, nil); err != nil {
					t.Fatal(err)
				}

				got := sess.GeneratedTokens()
				if len(got) != len(tc.GreedyTokens) {
					t.Fatalf("greedy tokens = %v (len %d), ожидали %v (len %d)", got, len(got), tc.GreedyTokens, len(tc.GreedyTokens))
				}

				for i := range got {
					if got[i] != tc.GreedyTokens[i] {
						t.Fatalf("token[%d] = %d, ожидали %d (full: %v)", i, got[i], tc.GreedyTokens[i], got)
					}
				}

			case tc.ChatUser != "":
				prompt, err := chat.FormatUser(tc.ChatUser, chat.Options{
					Metadata: engine.Metadata(),
				})
				if err != nil {
					t.Fatal(err)
				}

				ids, err := engine.Tokenizer().Encode(prompt)
				if err != nil {
					t.Fatal(err)
				}

				engine.Model.ResetCache()
				logits, err := engine.Model.Forward(ids, 0)
				if err != nil {
					t.Fatal(err)
				}

				next := gguf.Greedy(logits)
				if next != tc.GreedyNext {
					t.Fatalf("greedy next = %d, ожидали %d", next, tc.GreedyNext)
				}

			case tc.Input != "" && tc.GreedyNext > 0:
				ids, err := engine.Tokenizer().Encode(tc.Input)
				if err != nil {
					t.Fatal(err)
				}

				engine.Model.ResetCache()
				logits, err := engine.Model.Forward(ids, 0)
				if err != nil {
					t.Fatal(err)
				}

				next := gguf.Greedy(logits)
				if next != tc.GreedyNext {
					t.Fatalf("greedy next = %d, ожидали %d", next, tc.GreedyNext)
				}

				for _, want := range tc.TopLogits {
					got := logits[want.ID]
					if math.Abs(float64(got-want.Logit)) > 1e-3 {
						t.Fatalf("logit[%d] = %v, ожидали ~%v", want.ID, got, want.Logit)
					}
				}
			}
		})
	}
}

func TestFixtureFilePresent(t *testing.T) {
	for _, p := range []string{
		"../fixtures/qwen3_golden.json",
		"test/fixtures/qwen3_golden.json",
	} {
		if _, err := os.Stat(p); err == nil {
			return
		}
	}

	t.Fatal("qwen3_golden.json не найден в репозитории")
}
