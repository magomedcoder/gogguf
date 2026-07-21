//go:build integration

package integration

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/magomedcoder/gogguf"
	"github.com/magomedcoder/gogguf/pkg/chat"
	"github.com/magomedcoder/gogguf/pkg/debug"
)

func fullLogitsFixture(t *testing.T, name string) string {
	t.Helper()
	paths := []string{
		filepath.Join("test/fixtures", name),
		filepath.Join("../fixtures", name),
		filepath.Join("fixtures", name),
	}

	if p := os.Getenv("GGUF_FULL_LOGITS"); p != "" {
		paths = []string{filepath.Join(p, name)}
	}

	for _, p := range paths {
		if _, err := os.Stat(p + ".bin"); err == nil {
			return p
		}
	}

	t.Skipf("full logits fixture %s.bin не найден", name)

	return ""
}

func TestFullLogitsFixture(t *testing.T) {
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
		{"raw_hello", "qwen3_raw_hello_logits", false, "Hello"},
		{"chat_hello", "qwen3_chat_hello_logits", true, "Hello"},
	}

	const tol = 1e-4

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			prefix := fullLogitsFixture(t, tc.fixture)
			_, want, err := debug.LoadLogitsDump(prefix)
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

			engine.Model.ResetCache()
			got, err := engine.Model.Forward(ids, 0)
			if err != nil {
				t.Fatal(err)
			}

			if len(got) != len(want) {
				t.Fatalf("vocab got=%d want=%d", len(got), len(want))
			}

			st := debug.DiffLogits(got, want, tol)
			if st.OverTol > 0 {
				t.Fatalf("full logits: max_abs=%.6g@%d mean=%.6g over_tol=%d", st.MaxAbs, st.MaxAbsIdx, st.MeanAbs, st.OverTol)
			}

			if gogguf.Greedy(got) != gogguf.Greedy(want) {
				t.Fatalf("greedy got=%d want=%d", gogguf.Greedy(got), gogguf.Greedy(want))
			}
		})
	}
}
