package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"github.com/magomedcoder/gogguf"
	"github.com/magomedcoder/gogguf/pkg/debug"
	"github.com/magomedcoder/gogguf/pkg/model"
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

// runLayerLogits prints greedy next and top logits after each layer.
func runLayerLogits(args []string) error {
	fs := flag.NewFlagSet("layerlogits", flag.ContinueOnError)
	modelPath := fs.String("m", "./models/Qwen3-4B-Q8_0.gguf", "path to GGUF")
	prompt := fs.String("p", "Hello", "prompt")
	chatMode := fs.Bool("chat", false, "chat template")
	topN := fs.Int("top", 5, "number of top logits per layer (0 = greedy only)")

	if err := fs.Parse(args); err != nil {
		return err
	}

	engine, err := gogguf.LoadMapped(*modelPath, gogguf.LoadOptions{})
	if err != nil {
		return err
	}
	defer engine.Close()

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
		return fmt.Errorf("model does not support debug hooks")
	}

	engine.Model.ResetCache()
	if _, err := engine.Model.Forward(ids, 0); err != nil {
		return err
	}

	out := layerLogitsFile{
		Model: filepath.Base(*modelPath),
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
	setter, ok := engine.Model.(model.DebugHookSetter)
	if !ok {
		return false
	}
	setter.SetDebugHooks(&debug.Hooks{OnLayerLogits: onLayer})
	return true
}
