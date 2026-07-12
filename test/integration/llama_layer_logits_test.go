//go:build integration

package integration

import (
	"encoding/json"
	"math"
	"os"
	"testing"

	"github.com/magomedcoder/gogguf"
	"github.com/magomedcoder/gogguf/pkg/chat"
	"github.com/magomedcoder/gogguf/pkg/debug"
	"github.com/magomedcoder/gogguf/pkg/model/llama"
)

func loadLlamaLayerLogitsFixture(t *testing.T) layerLogitsFile {
	t.Helper()

	paths := []string{
		"../fixtures/llama32_layer_logits.json",
		"test/fixtures/llama32_layer_logits.json",
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
		t.Fatal("llama32_layer_logits.json не найден")
	}

	var lf layerLogitsFile
	if err := json.Unmarshal(data, &lf); err != nil {
		t.Fatalf("не удалось разобрать fixture: %v", err)
	}

	return lf
}

func TestLlama32LayerLogitsFixture(t *testing.T) {
	lf := loadLlamaLayerLogitsFixture(t)
	engine, err := gogguf.Load(llamaModelPath(t), gogguf.LoadOptions{})
	if err != nil {
		t.Fatalf("не удалось загрузить модель: %v", err)
	}

	ctx, err := engine.NewContext()
	if err != nil {
		t.Fatal(err)
	}

	setter, ok := engine.Model.(interface {
		SetDebugHooks(*llama.DebugHooks)
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

			ids, err := ctx.EncodeForInference(prompt)
			if err != nil {
				t.Fatal(err)
			}

			got := make([]int, 0, len(tc.LayerGreedy))
			gotTop := make([][]debug.Logit, 0, len(tc.LayerTopLogits))

			setter.SetDebugHooks(&llama.DebugHooks{
				OnLayerLogits: func(layer int, logits []float32) {
					_ = layer
					got = append(got, gogguf.Greedy(logits))
					if len(tc.LayerTopLogits) > 0 {
						gotTop = append(gotTop, debug.TopLogits(logits, len(tc.LayerTopLogits[0])))
					}
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

			if len(tc.LayerTopLogits) == 0 {
				return
			}

			if len(gotTop) != len(tc.LayerTopLogits) {
				t.Fatalf("layer top logits count = %d, ожидали %d", len(gotTop), len(tc.LayerTopLogits))
			}

			const logitTol = 1e-4
			for layer := range gotTop {
				want := tc.LayerTopLogits[layer]
				if len(gotTop[layer]) != len(want) {
					t.Fatalf("layer[%d] top logits len = %d, ожидали %d", layer, len(gotTop[layer]), len(want))
				}

				for j := range want {
					if gotTop[layer][j].ID != want[j].ID {
						t.Fatalf("layer[%d] top[%d] id = %d, ожидали %d", layer, j, gotTop[layer][j].ID, want[j].ID)
					}

					diff := math.Abs(float64(gotTop[layer][j].Logit - want[j].Logit))
					if diff > logitTol {
						t.Fatalf("layer[%d] logit[%d] = %v, ожидали ~%v", layer, want[j].ID, gotTop[layer][j].Logit, want[j].Logit)
					}
				}
			}
		})
	}
}
