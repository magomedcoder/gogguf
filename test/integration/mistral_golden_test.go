//go:build integration

package integration

import (
	"encoding/json"
	"os"
	"testing"

	"github.com/magomedcoder/gogguf"
	"github.com/magomedcoder/gogguf/pkg/chat"
)

func mistralModelPath(t *testing.T) string {
	t.Helper()

	if p := os.Getenv("MISTRAL_GGUF_MODEL"); p != "" {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}

	for _, p := range []string{
		"models/Mistral-7B-Instruct-v0.2-Q4_K_M.gguf",
		"../models/Mistral-7B-Instruct-v0.2-Q4_K_M.gguf",
		"../../models/Mistral-7B-Instruct-v0.2-Q4_K_M.gguf",
	} {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}

	t.Skip("Mistral-7B-Instruct-v0.2-Q4_K_M.gguf не найден")

	return ""
}

func loadMistralGolden(t *testing.T) goldenFile {
	t.Helper()

	paths := []string{
		"../fixtures/mistral72_golden.json",
		"test/fixtures/mistral72_golden.json",
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
		t.Fatal("mistral72_golden.json не найден")
	}

	var gf goldenFile
	if err := json.Unmarshal(data, &gf); err != nil {
		t.Fatalf("не удалось разобрать fixture: %v", err)
	}

	return gf
}

func TestMistral72GoldenFixture(t *testing.T) {
	gf := loadMistralGolden(t)
	engine, err := gogguf.Load(mistralModelPath(t), gogguf.LoadOptions{})
	if err != nil {
		t.Fatalf("не удалось загрузить модель: %v", err)
	}

	ctx, err := engine.NewContext()
	if err != nil {
		t.Fatal(err)
	}

	for _, tc := range gf.Cases {
		t.Run(tc.Name, func(t *testing.T) {
			switch {
			case tc.Encode != nil:
				ids, err := engine.Tokenizer().Encode(tc.Input)
				if err != nil {
					t.Fatal(err)
				}

				if len(ids) != len(tc.Encode) || ids[0] != tc.Encode[0] {
					t.Fatalf("Encode(%q) = %v, ожидали %v", tc.Input, ids, tc.Encode)
				}

			case tc.ChatUser != "" && len(tc.GreedyTokens) > 0:
				prompt, err := chat.FormatUser(tc.ChatUser, chat.Options{
					Metadata: engine.Metadata(),
				})
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

				if err := sess.GenerateSteps(gogguf.GenerateParams{
					MaxTokens: maxTok,
					Sampler:   gogguf.Greedy,
				}); err != nil {
					t.Fatal(err)
				}

				got := sess.GeneratedTokens()
				if len(got) != len(tc.GreedyTokens) {
					t.Fatalf("greedy tokens = %v, ожидали %v", got, tc.GreedyTokens)
				}

				for i := range got {
					if got[i] != tc.GreedyTokens[i] {
						t.Fatalf("token[%d] = %d, ожидали %d", i, got[i], tc.GreedyTokens[i])
					}
				}

			case tc.ChatUser != "":
				prompt, err := chat.FormatUser(tc.ChatUser, chat.Options{
					Metadata: engine.Metadata(),
				})
				if err != nil {
					t.Fatal(err)
				}

				if len(tc.TopLogits) > 0 {
					ids, err := ctx.EncodeForInference(prompt)
					if err != nil {
						t.Fatal(err)
					}

					engine.Model.ResetCache()
					logits, err := engine.Model.Forward(ids, 0)
					if err != nil {
						t.Fatal(err)
					}

					assertGreedyAndTopLogits(t, logits, tc.GreedyNext, tc.TopLogits)
					return
				}

				sess, err := ctx.StartGeneration(prompt)
				if err != nil {
					t.Fatal(err)
				}

				next, err := sess.DecodeStep(gogguf.Greedy)
				if err != nil {
					t.Fatal(err)
				}

				if next != tc.GreedyNext {
					t.Fatalf("greedy next = %d, ожидали %d", next, tc.GreedyNext)
				}

			case tc.Input != "" && tc.GreedyNext > 0:
				if len(tc.TopLogits) > 0 {
					ids, err := ctx.EncodeForInference(tc.Input)
					if err != nil {
						t.Fatal(err)
					}

					engine.Model.ResetCache()
					logits, err := engine.Model.Forward(ids, 0)
					if err != nil {
						t.Fatal(err)
					}

					assertGreedyAndTopLogits(t, logits, tc.GreedyNext, tc.TopLogits)
					return
				}

				sess, err := ctx.StartGeneration(tc.Input)
				if err != nil {
					t.Fatal(err)
				}

				next, err := sess.DecodeStep(gogguf.Greedy)
				if err != nil {
					t.Fatal(err)
				}

				if next != tc.GreedyNext {
					t.Fatalf("greedy next = %d, ожидали %d", next, tc.GreedyNext)
				}
			}
		})
	}
}
