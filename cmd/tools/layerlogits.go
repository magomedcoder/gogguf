package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"github.com/magomedcoder/gogguf"
	"github.com/magomedcoder/gogguf/pkg/debug"
	"github.com/magomedcoder/gogguf/pkg/model/llama"
	"github.com/magomedcoder/gogguf/pkg/model/qwen3"
)

type layerLogitsCase struct {
	Name           string          `json:"name"`
	Input          string          `json:"input,omitempty"`
	ChatUser       string          `json:"chat_user,omitempty"`
	LayerGreedy    []int           `json:"layer_greedy"`
	LayerTopLogits [][]debug.Logit `json:"layer_top_logits,omitempty"`
}

type layerLogitsFile struct {
	Model string            `json:"model"`
	Cases []layerLogitsCase `json:"cases"`
}

// runLayerLogits печатает greedy next и top logits после каждого слоя
func runLayerLogits(args []string) error {
	fs := flag.NewFlagSet("layerlogits", flag.ContinueOnError)
	modelPath := fs.String("m", "./models/Qwen3-0.6B-Q8_0.gguf", "путь к GGUF")
	prompt := fs.String("p", "Hello", "промпт")
	chatMode := fs.Bool("chat", false, "chat template")
	topN := fs.Int("top", 5, "число top logits на слой (0 = только greedy)")

	if err := fs.Parse(args); err != nil {
		return err
	}

	engine, err := gogguf.Load(*modelPath, gogguf.LoadOptions{})
	if err != nil {
		return err
	}

	ctx, err := engine.NewContext()
	if err != nil {
		return err
	}

	text := *prompt
	name := "raw_hello_layer_greedy"
	if *chatMode {
		name = "chat_hello_layer_greedy"
		text, err = gogguf.FormatChatUser(*prompt, gogguf.ChatOptions{
			Metadata: engine.Metadata(),
		})
		if err != nil {
			return err
		}
	}

	ids, err := ctx.EncodeForInference(text)
	if err != nil {
		return err
	}

	layerGreedy := make([]int, 0, 32)
	var layerTop [][]debug.Logit

	onLayer := func(_ int, logits []float32) {
		layerGreedy = append(layerGreedy, gogguf.Greedy(logits))
		if *topN > 0 {
			layerTop = append(layerTop, debug.TopLogits(logits, *topN))
		}
	}

	if !setLayerLogitsHook(engine, onLayer) {
		return fmt.Errorf("модель не поддерживает debug hooks")
	}

	engine.Model.ResetCache()
	if _, err := engine.Model.Forward(ids, 0); err != nil {
		return err
	}

	out := layerLogitsFile{
		Model: modelBasename(*modelPath),
		Cases: []layerLogitsCase{{
			Name:        name,
			LayerGreedy: layerGreedy,
		}},
	}

	if *topN > 0 {
		out.Cases[0].LayerTopLogits = layerTop
	}

	if *chatMode {
		out.Cases[0].ChatUser = *prompt
	} else {
		out.Cases[0].Input = *prompt
	}

	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	return enc.Encode(out)
}

func setLayerLogitsHook(engine *gogguf.Engine, onLayer func(int, []float32)) bool {
	switch m := engine.Model.(type) {
	case interface {
		SetDebugHooks(*qwen3.DebugHooks)
	}:
		m.SetDebugHooks(&qwen3.DebugHooks{
			OnLayerLogits: onLayer,
		})
		return true
	case interface {
		SetDebugHooks(*llama.DebugHooks)
	}:
		m.SetDebugHooks(&llama.DebugHooks{
			OnLayerLogits: onLayer,
		})
		return true
	default:
		return false
	}
}

func modelBasename(path string) string {
	for i := len(path) - 1; i >= 0; i-- {
		if path[i] == '/' {
			return path[i+1:]
		}
	}

	return path
}
