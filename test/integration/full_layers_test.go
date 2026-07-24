//go:build integration

package integration

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/magomedcoder/gogguf"
	"github.com/magomedcoder/gogguf/pkg/chat"
	"github.com/magomedcoder/gogguf/pkg/debug"
	"github.com/magomedcoder/gogguf/pkg/model/qwen3"
)

func fullLayersFixture(t *testing.T, name string) string {
	t.Helper()
	paths := []string{
		filepath.Join("test/fixtures", name),
		filepath.Join("../fixtures", name),
		filepath.Join("fixtures", name),
	}

	if p := os.Getenv("GGUF_FULL_LAYERS"); p != "" {
		paths = []string{filepath.Join(p, name)}
	}

	for _, p := range paths {
		if _, err := os.Stat(p + ".bin"); err == nil {
			return p
		}
	}
	t.Skipf("full layers fixture %s.bin не найден", name)

	return ""
}

func collectLayerHidden(t *testing.T, engine *gogguf.Engine, ids []int) debug.LayersDump {
	t.Helper()
	setter, ok := engine.Model.(interface{ SetDebugHooks(*qwen3.DebugHooks) })
	if !ok {
		t.Fatal("модель не поддерживает debug hooks")
	}

	var dump debug.LayersDump
	setter.SetDebugHooks(&qwen3.DebugHooks{
		OnEmbed: func(x []float32) {
			dump.Embed = append([]float32(nil), x...)
		},

		OnLayer: func(_ int, x []float32) {
			dump.Layers = append(dump.Layers, append([]float32(nil), x...))
		},
	})

	engine.Model.ResetCache()
	if _, err := engine.Model.Forward(ids, 0); err != nil {
		t.Fatal(err)
	}

	return dump
}

// TestFullLayersFixture сверяет embed + hidden каждого слоя с CPU fixture (tol 1e-4)
func TestFullLayersFixture(t *testing.T) {
	engine, err := gogguf.Load(modelPath(t), gogguf.LoadOptions{})
	if err != nil {
		t.Fatalf("не удалось загрузить модель: %v", err)
	}

	cases := []struct {
		name    string
		fixture string
		chat    bool
		prompt  string
	}{
		{"raw_hello", "qwen3_raw_hello_layers", false, "Hello"},
		{"chat_hello", "qwen3_chat_hello_layers", true, "Hello"},
	}

	const tol = 1e-4

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			prefix := fullLayersFixture(t, tc.fixture)
			_, want, err := debug.LoadLayersDump(prefix)
			if err != nil {
				t.Fatal(err)
			}

			text := tc.prompt
			if tc.chat {
				text, err = chat.FormatUser(tc.prompt, chat.Options{
					Metadata: engine.Metadata(),
				})
				if err != nil {
					t.Fatal(err)
				}
			}

			ctx, err := engine.NewContext()
			if err != nil {
				t.Fatal(err)
			}

			ids, err := ctx.EncodeForInference(text)
			if err != nil {
				t.Fatal(err)
			}

			got := collectLayerHidden(t, engine, ids)
			embed, layers, err := debug.DiffLayers(got, want, tol)
			if err != nil {
				t.Fatal(err)
			}

			if embed.OverTol > 0 {
				t.Fatalf("embed: max_abs=%.6g@%d over=%d", embed.MaxAbs, embed.MaxAbsIdx, embed.OverTol)
			}

			var worst float64
			var worstI, totalOver int
			for i, st := range layers {
				if st.MaxAbs > worst {
					worst = st.MaxAbs
					worstI = i
				}

				totalOver += st.OverTol
				if st.OverTol > 0 {
					t.Logf("layer[%d]: max_abs=%.6g@%d over=%d", i, st.MaxAbs, st.MaxAbsIdx, st.OverTol)
				}
			}

			if totalOver > 0 {
				t.Fatalf("layers: worst_max_abs=%.6g@layer%d total_over=%d (tol=%g)", worst, worstI, totalOver, tol)
			}
		})
	}
}
