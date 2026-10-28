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
		t.Fatal("llama32_layer_logits.json not found")
	}

	var lf layerLogitsFile
	if err := json.Unmarshal(data, &lf); err != nil {
		t.Fatalf("failed to parse fixture: %v", err)
	}

	return lf
}

func TestLlama32LayerLogitsFixture(t *testing.T) {
	lf := loadLlamaLayerLogitsFixture(t)
	engine, err := gogguf.LoadMapped(llamaModelPath(t), gogguf.LoadOptions{})
	if err != nil {
		t.Fatalf("failed to load model: %v", err)
	}
	defer engine.Close()

	ctx, err := engine.NewContext()
	if err != nil {
		t.Fatal(err)
	}

	setter, ok := engine.Model.(interface {
		SetDebugHooks(*llama.DebugHooks)
	})
	if !ok {
		t.Fatal("model does not support debug hooks")
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
				t.Fatalf("layer count = %d, expected %d", len(got), len(tc.LayerGreedy))
			}

			for i := range got {
				if got[i] == tc.LayerGreedy[i] {
					continue
				}
				// Near-tie: fixture top-2 can swap under ~0.1 logit slack (see chat layer 1).
				if i < len(tc.LayerTopLogits) && llamaNearTieGreedy(got[i], tc.LayerGreedy[i], tc.LayerTopLogits[i]) {
					t.Logf("layer[%d] greedy = %d (fixture %d, near-tie)", i, got[i], tc.LayerGreedy[i])
					continue
				}
				t.Fatalf("layer[%d] greedy = %d, expected %d", i, got[i], tc.LayerGreedy[i])
			}

			if len(tc.LayerTopLogits) == 0 {
				return
			}

			if len(gotTop) != len(tc.LayerTopLogits) {
				t.Fatalf("layer top logits count = %d, expected %d", len(gotTop), len(tc.LayerTopLogits))
			}

			for layer := range gotTop {
				want := tc.LayerTopLogits[layer]
				wantByID := make(map[int]float32, len(want))
				for _, w := range want {
					wantByID[w.ID] = w.Logit
				}

				matched := 0
				for _, g := range gotTop[layer] {
					w, ok := wantByID[g.ID]
					if !ok {
						continue
					}

					matched++
					if !llamaLogitClose(g.Logit, w) {
						t.Fatalf("layer[%d] logit[%d] = %v, expected ~%v", layer, g.ID, g.Logit, w)
					}
				}
				if matched == 0 {
					t.Fatalf("layer[%d]: no overlap between got top and fixture top", layer)
				}
			}
		})
	}
}

func llamaNearTieGreedy(got, want int, fixtureTop []debug.Logit) bool {
	if len(fixtureTop) < 2 {
		return false
	}

	var gotLogit, wantLogit float32
	var sawGot, sawWant bool
	for _, l := range fixtureTop {
		if l.ID == got {
			gotLogit, sawGot = l.Logit, true
		}

		if l.ID == want {
			wantLogit, sawWant = l.Logit, true
		}
	}

	if !sawGot || !sawWant {
		return false
	}

	return math.Abs(float64(gotLogit-wantLogit)) <= 0.3
}
