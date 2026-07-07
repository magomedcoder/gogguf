//go:build integration

package integration

import (
	"encoding/json"
	"os"
	"testing"

	"github.com/magomedcoder/gogguf"
	"github.com/magomedcoder/gogguf/pkg/chat"
	"github.com/magomedcoder/gogguf/pkg/model/qwen3"
)

type layerLogitsFile struct {
	Model string            `json:"model"`
	Cases []layerLogitsCase `json:"cases"`
}

type layerLogitsCase struct {
	Name        string `json:"name"`
	Input       string `json:"input,omitempty"`
	ChatUser    string `json:"chat_user,omitempty"`
	LayerGreedy []int  `json:"layer_greedy"`
}

func loadLayerLogitsFixture(t *testing.T) layerLogitsFile {
	t.Helper()

	paths := []string{
		"../fixtures/qwen3_layer_logits.json",
		"test/fixtures/qwen3_layer_logits.json",
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
		t.Skip("layer logits fixture не найден")
	}

	var lf layerLogitsFile
	if err := json.Unmarshal(data, &lf); err != nil {
		t.Fatalf("не удалось разобрать fixture: %v", err)
	}

	return lf
}

func TestLayerLogitsFixture(t *testing.T) {
	lf := loadLayerLogitsFixture(t)
	engine, err := gogguf.Load(modelPath(t), gogguf.LoadOptions{})
	if err != nil {
		t.Fatalf("не удалось загрузить модель: %v", err)
	}

	setter, ok := engine.Model.(interface {
		SetDebugHooks(*qwen3.DebugHooks)
	})
	if !ok {
		t.Fatal("модель не поддерживает debug hooks")
	}

	for _, tc := range lf.Cases {
		t.Run(tc.Name, func(t *testing.T) {
			prompt := tc.Input
			if tc.ChatUser != "" {
				var err error
				prompt, err = chat.FormatUser(tc.ChatUser, chat.Options{
					Metadata: engine.Metadata(),
				})
				if err != nil {
					t.Fatal(err)
				}
			}

			ids, err := engine.Tokenizer().Encode(prompt)
			if err != nil {
				t.Fatal(err)
			}

			got := make([]int, 0, len(tc.LayerGreedy))

			setter.SetDebugHooks(&qwen3.DebugHooks{
				OnLayerLogits: func(layer int, logits []float32) {
					_ = layer
					got = append(got, gogguf.Greedy(logits))
				},
			})

			engine.Model.ResetCache()
			if _, err := engine.Model.Forward(ids, 0); err != nil {
				t.Fatal(err)
			}

			if len(got) != len(tc.LayerGreedy) {
				t.Fatalf("layer count = %d, ожидали %d", len(got), len(tc.LayerGreedy))
			}

			for i := range got {
				if got[i] != tc.LayerGreedy[i] {
					t.Fatalf("layer[%d] greedy = %d, ожидали %d", i, got[i], tc.LayerGreedy[i])
				}
			}
		})
	}
}
