//go:build integration

package integration

import (
	"encoding/json"
	"math"
	"os"
	"testing"

	"github.com/magomedcoder/gogguf"
	"github.com/magomedcoder/gogguf/pkg/chat"
	"github.com/magomedcoder/gogguf/pkg/model/qwen3"
	"github.com/magomedcoder/gogguf/pkg/ops"
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

	paths := []string{
		"../fixtures/qwen3_layers.json",
		"test/fixtures/qwen3_layers.json",
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
		t.Skip("layers fixture не найден")
	}

	var lf layersFile
	if err := json.Unmarshal(data, &lf); err != nil {
		t.Fatalf("не удалось разобрать fixture: %v", err)
	}

	return lf
}

func TestLayersFixture(t *testing.T) {
	lf := loadLayersFixture(t)
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

			var embedRMS float32
			layerRMS := make([]float32, 0, len(tc.LayerRMS))

			setter.SetDebugHooks(&qwen3.DebugHooks{
				OnEmbed: func(x []float32) {
					embedRMS = ops.VectorRMS(x)
				},
				OnLayer: func(layer int, x []float32) {
					layerRMS = append(layerRMS, ops.VectorRMS(x))
				},
			})

			engine.Model.ResetCache()
			logits, err := engine.Model.Forward(ids, 0)
			if err != nil {
				t.Fatal(err)
			}

			if math.Abs(float64(embedRMS-tc.EmbedRMS)) > 1e-2 {
				t.Fatalf("embed_rms = %v, ожидали ~%v", embedRMS, tc.EmbedRMS)
			}

			if len(layerRMS) != len(tc.LayerRMS) {
				t.Fatalf("layer count = %d, ожидали %d", len(layerRMS), len(tc.LayerRMS))
			}

			for i := range layerRMS {
				if math.Abs(float64(layerRMS[i]-tc.LayerRMS[i])) > 1e-1 {
					t.Fatalf("layer[%d] rms = %v, ожидали ~%v", i, layerRMS[i], tc.LayerRMS[i])
				}
			}

			next := gogguf.Greedy(logits)
			if next != tc.GreedyNext {
				t.Fatalf("greedy next = %d, ожидали %d", next, tc.GreedyNext)
			}
		})
	}
}
