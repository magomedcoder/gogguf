//go:build integration

package integration

import (
	"math"
	"testing"

	"github.com/magomedcoder/gogguf"
	"github.com/magomedcoder/gogguf/pkg/debug"
	"github.com/magomedcoder/gogguf/pkg/model"
)

type layerLogitsFile struct {
	Model string            `json:"model"`
	Cases []layerLogitsCase `json:"cases"`
}

type layerLogitsCase struct {
	Name           string          `json:"name"`
	Input          string          `json:"input,omitempty"`
	ChatUser       string          `json:"chat_user,omitempty"`
	LayerGreedy    []int           `json:"layer_greedy"`
	LayerTopLogits [][]debug.Logit `json:"layer_top_logits,omitempty"`
}

func loadLayerLogitsFixture(t *testing.T) layerLogitsFile {
	t.Helper()

	var lf layerLogitsFile
	loadFixtureJSON(t, true, &lf, fixtureJSONPaths("qwen3_layer_logits.json")...)
	return lf
}

func TestLayerLogitsFixture(t *testing.T) {
	lf := loadLayerLogitsFixture(t)
	engine, err := gogguf.LoadMapped(modelPath(t), gogguf.LoadOptions{})
	if err != nil {
		t.Fatalf("failed to load model: %v", err)
	}
	defer engine.Close()

	ctx, err := engine.NewContext()
	if err != nil {
		t.Fatal(err)
	}

	setter, ok := engine.Model.(model.DebugHookSetter)
	if !ok {
		t.Fatal("model does not support debug hooks")
	}

	for _, tc := range lf.Cases {
		t.Run(tc.Name, func(t *testing.T) {
			prompt := promptFromCase(t, engine, tc.Input, tc.ChatUser)

			ids, err := ctx.EncodeForInference(prompt)
			if err != nil {
				t.Fatal(err)
			}

			got := make([]int, 0, len(tc.LayerGreedy))
			gotTop := make([][]debug.Logit, 0, len(tc.LayerTopLogits))

			setter.SetDebugHooks(&debug.Hooks{
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
				t.Fatalf("layer count = %d, expected %d", len(got), len(tc.LayerGreedy))
			}

			for i := range got {
				if got[i] != tc.LayerGreedy[i] {
					t.Fatalf("layer[%d] greedy = %d, expected %d", i, got[i], tc.LayerGreedy[i])
				}
			}

			if len(tc.LayerTopLogits) == 0 {
				return
			}

			if len(gotTop) != len(tc.LayerTopLogits) {
				t.Fatalf("layer top logits count = %d, expected %d", len(gotTop), len(tc.LayerTopLogits))
			}

			const logitTol = 1e-4

			for layer := range gotTop {
				want := tc.LayerTopLogits[layer]
				if len(gotTop[layer]) != len(want) {
					t.Fatalf("layer[%d] top logits len = %d, expected %d", layer, len(gotTop[layer]), len(want))
				}

				for j := range want {
					if gotTop[layer][j].ID != want[j].ID {
						t.Fatalf("layer[%d] top[%d] id = %d, expected %d", layer, j, gotTop[layer][j].ID, want[j].ID)
					}

					diff := math.Abs(float64(gotTop[layer][j].Logit - want[j].Logit))
					if diff > logitTol {
						t.Fatalf("layer[%d] logit[%d] = %v, expected ~%v (diff %v)", layer, want[j].ID, gotTop[layer][j].Logit, want[j].Logit, diff)
					}
				}
			}
		})
	}
}
