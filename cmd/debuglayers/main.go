package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"github.com/magomedcoder/gguf.go"
	"github.com/magomedcoder/gguf.go/pkg/model/qwen3"
	"github.com/magomedcoder/gguf.go/pkg/ops"
)

type layerReport struct {
	Input      string        `json:"input"`
	Chat       bool          `json:"chat,omitempty"`
	TokenCount int           `json:"token_count"`
	EmbedRMS   float32       `json:"embed_rms"`
	Layers     []layerMetric `json:"layers"`
	GreedyNext int           `json:"greedy_next"`
	TopLogits  []logitMetric `json:"top_logits"`
}

type layerMetric struct {
	Layer int     `json:"layer"`
	RMS   float32 `json:"rms"`
}

type logitMetric struct {
	ID    int     `json:"id"`
	Logit float32 `json:"logit"`
}

func main() {
	modelPath := flag.String("m", "./models/Qwen3-0.6B-Q8_0.gguf", "путь к GGUF")
	prompt := flag.String("p", "Hello", "промпт")
	chat := flag.Bool("chat", false, "chat template")
	topN := flag.Int("top", 5, "число top logits в отчёте")
	flag.Parse()

	engine, err := gguf.Load(*modelPath, gguf.LoadOptions{})
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	text := *prompt
	if *chat {
		text, err = gguf.FormatChatUser(*prompt, gguf.ChatOptions{
			Metadata: engine.Metadata(),
		})
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
	}

	ids, err := engine.Tokenizer().Encode(text)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	report := layerReport{
		Input:      *prompt,
		Chat:       *chat,
		TokenCount: len(ids),
		Layers:     make([]layerMetric, 0, 32),
	}

	setter, ok := engine.Model.(interface {
		SetDebugHooks(*qwen3.DebugHooks)
	})
	if !ok {
		fmt.Fprintln(os.Stderr, "модель не поддерживает debug hooks")
		os.Exit(1)
	}

	setter.SetDebugHooks(&qwen3.DebugHooks{
		OnEmbed: func(x []float32) {
			report.EmbedRMS = ops.VectorRMS(x)
		},
		OnLayer: func(layer int, x []float32) {
			report.Layers = append(report.Layers, layerMetric{
				Layer: layer,
				RMS:   ops.VectorRMS(x),
			})
		},
	})

	engine.Model.ResetCache()
	logits, err := engine.Model.Forward(ids, 0)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	report.GreedyNext = gguf.Greedy(logits)
	report.TopLogits = topLogits(logits, *topN)

	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(report); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func topLogits(logits []float32, n int) []logitMetric {
	items := make([]logitMetric, len(logits))
	for i, v := range logits {
		items[i] = logitMetric{
			ID:    i,
			Logit: v,
		}
	}

	for i := range items {
		for j := i + 1; j < len(items); j++ {
			if items[j].Logit > items[i].Logit {
				items[i], items[j] = items[j], items[i]
			}
		}
	}

	if n > len(items) {
		n = len(items)
	}

	out := make([]logitMetric, n)
	copy(out, items[:n])
	return out
}
