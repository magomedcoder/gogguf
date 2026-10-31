//go:build integration

package integration

import (
	"math"
	"testing"

	"github.com/magomedcoder/gogguf"
	"github.com/magomedcoder/gogguf/pkg/compute"
	"github.com/magomedcoder/gogguf/pkg/debug"
	"github.com/magomedcoder/gogguf/pkg/model"
)

type layersFile struct {
	Model string       `json:"model"`
	Cases []layersCase `json:"cases"`
}

type layersCase struct {
	Name       string    `json:"name"`
	Input      string    `json:"input,omitempty"`
	ChatUser   string    `json:"chat_user,omitempty"`
	EmbedRMS   float32   `json:"embed_rms"`
	LayerRMS   []float32 `json:"layer_rms"`
	GreedyNext int       `json:"greedy_next"`
}

func loadLayersFixture(t *testing.T) layersFile {
	t.Helper()

	var lf layersFile
	loadFixtureJSON(t, true, &lf, fixtureJSONPaths("qwen3_layers.json")...)
	return lf
}

func TestLayersFixture(t *testing.T) {
	lf := loadLayersFixture(t)
	engine, err := gogguf.LoadMapped(modelPath(t), gogguf.LoadOptions{})
	if err != nil {
		t.Fatalf("failed to load model: %v", err)
	}
	defer engine.Close()

	setter, ok := engine.Model.(model.DebugHookSetter)
	if !ok {
		t.Fatal("model does not support debug hooks")
	}

	for _, tc := range lf.Cases {
		t.Run(tc.Name, func(t *testing.T) {
			prompt := promptFromCase(t, engine, tc.Input, tc.ChatUser)

			ids, err := engine.Tokenizer().Encode(prompt)
			if err != nil {
				t.Fatal(err)
			}

			var embedRMS float32
			layerRMS := make([]float32, 0, len(tc.LayerRMS))

			setter.SetDebugHooks(&debug.Hooks{
				OnEmbed: func(x []float32) {
					embedRMS = compute.VectorRMS(x)
				},
				OnLayer: func(layer int, x []float32) {
					layerRMS = append(layerRMS, compute.VectorRMS(x))
				},
			})

			engine.Model.ResetCache()
			logits, err := engine.Model.Forward(ids, 0)
			if err != nil {
				t.Fatal(err)
			}

			if math.Abs(float64(embedRMS-tc.EmbedRMS)) > 1e-2 {
				t.Fatalf("embed_rms = %v, expected ~%v", embedRMS, tc.EmbedRMS)
			}

			if len(layerRMS) != len(tc.LayerRMS) {
				t.Fatalf("layer count = %d, expected %d", len(layerRMS), len(tc.LayerRMS))
			}

			for i := range layerRMS {
				if math.Abs(float64(layerRMS[i]-tc.LayerRMS[i])) > 1e-1 {
					t.Fatalf("layer[%d] rms = %v, expected ~%v", i, layerRMS[i], tc.LayerRMS[i])
				}
			}

			next := gogguf.Greedy(logits)
			if next != tc.GreedyNext {
				t.Fatalf("greedy next = %d, expected %d", next, tc.GreedyNext)
			}
		})
	}
}
