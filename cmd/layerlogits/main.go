package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"github.com/magomedcoder/gogguf"
	"github.com/magomedcoder/gogguf/pkg/debug"
	"github.com/magomedcoder/gogguf/pkg/model/qwen3"
)

type caseOut struct {
	Name           string          `json:"name"`
	Input          string          `json:"input,omitempty"`
	ChatUser       string          `json:"chat_user,omitempty"`
	LayerGreedy    []int           `json:"layer_greedy"`
	LayerTopLogits [][]debug.Logit `json:"layer_top_logits,omitempty"`
}

type fileOut struct {
	Model string    `json:"model"`
	Cases []caseOut `json:"cases"`
}

func main() {
	modelPath := flag.String("m", "./models/Qwen3-0.6B-Q8_0.gguf", "путь к GGUF")
	prompt := flag.String("p", "Hello", "промпт")
	chatMode := flag.Bool("chat", false, "chat template")
	topN := flag.Int("top", 5, "число top logits на слой (0 = только greedy)")
	flag.Parse()

	engine, err := gogguf.Load(*modelPath, gogguf.LoadOptions{})
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	text := *prompt
	name := "raw_hello_layer_greedy"
	if *chatMode {
		name = "chat_hello_layer_greedy"
		text, err = gogguf.FormatChatUser(*prompt, gogguf.ChatOptions{
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

	layerGreedy := make([]int, 0, 32)
	var layerTopLogits [][]debug.Logit

	setter, ok := engine.Model.(interface {
		SetDebugHooks(*qwen3.DebugHooks)
	})
	if !ok {
		fmt.Fprintln(os.Stderr, "модель не поддерживает debug hooks")
		os.Exit(1)
	}

	setter.SetDebugHooks(&qwen3.DebugHooks{
		OnLayerLogits: func(layer int, logits []float32) {
			_ = layer
			layerGreedy = append(layerGreedy, gogguf.Greedy(logits))
			if *topN > 0 {
				layerTopLogits = append(layerTopLogits, debug.TopLogits(logits, *topN))
			}
		},
	})

	engine.Model.ResetCache()
	if _, err := engine.Model.Forward(ids, 0); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	out := fileOut{
		Model: "Qwen3-0.6B-Q8_0",
		Cases: []caseOut{{
			Name:        name,
			LayerGreedy: layerGreedy,
		}},
	}

	if *topN > 0 {
		out.Cases[0].LayerTopLogits = layerTopLogits
	}

	if *chatMode {
		out.Cases[0].ChatUser = *prompt
	} else {
		out.Cases[0].Input = *prompt
	}

	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(out); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
