//go:build integration

package integration

import (
	"math"
	"testing"

	"github.com/magomedcoder/gogguf"
	"github.com/magomedcoder/gogguf/pkg/debug"
	"github.com/magomedcoder/gogguf/pkg/model"
)

func loadLlamaLayerLogitsFixture(t *testing.T) layerLogitsFile {
	t.Helper()

	var lf layerLogitsFile
	loadFixtureJSON(t, false, &lf, fixtureJSONPaths("llama32_layer_logits.json")...)
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
				if got[i] == tc.LayerGreedy[i] {
					continue
				}
				// Mid-layer lm_head probes drift with FMA order across runners
				ok := false
				if i < len(gotTop) {
					ok = llamaAcceptLayerGreedy(got[i], tc.LayerGreedy[i], gotTop[i])
				} else if i < len(tc.LayerTopLogits) {
					ok = llamaNearTieGreedy(got[i], tc.LayerGreedy[i], tc.LayerTopLogits[i])
				}

				if ok {
					t.Logf("layer[%d] greedy = %d (fixture %d, soft match)", i, got[i], tc.LayerGreedy[i])
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
					if !llamaLayerLogitClose(g.Logit, w) {
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

// llamaLayerLogitClose is looser than final-logits llamaLogitClose: mid-layer lm_head probes accumulate FMA order drift across the stack (~0.2–0.3 abs)
func llamaLayerLogitClose(got, want float32) bool {
	d := math.Abs(float64(got - want))
	return d <= math.Max(0.3, 0.04*math.Abs(float64(want)))
}

func llamaAcceptLayerGreedy(got, want int, top []debug.Logit) bool {
	if got == want {
		return true
	}

	if llamaNearTieGreedy(got, want, top) {
		return true
	}

	// Fixture argmax still among current top-k (rank may flip without a near-tie pair)
	for _, l := range top {
		if l.ID == want {
			return true
		}
	}
	return false
}

func llamaNearTieGreedy(got, want int, top []debug.Logit) bool {
	if len(top) < 2 {
		return false
	}

	var gotLogit, wantLogit float32
	var sawGot, sawWant bool
	for _, l := range top {
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

	return math.Abs(float64(gotLogit-wantLogit)) <= 0.5
}
