package main

import (
	"flag"
	"fmt"
	"path/filepath"
	"strings"

	"github.com/magomedcoder/gogguf"
	"github.com/magomedcoder/gogguf/pkg/debug"
	"github.com/magomedcoder/gogguf/pkg/model/llama"
	"github.com/magomedcoder/gogguf/pkg/model/mistral"
	"github.com/magomedcoder/gogguf/pkg/model/qwen3"
)

// runDumpLayers пишет embed + hidden после каждого слоя (.bin/.json)
func runDumpLayers(args []string) error {
	fs := flag.NewFlagSet("dumplayers", flag.ContinueOnError)
	modelPath := fs.String("m", "./models/Qwen3-0.6B-Q8_0.gguf", "путь к GGUF")
	prompt := fs.String("p", "Hello", "промпт")
	chatMode := fs.Bool("chat", false, "chat template")
	out := fs.String("o", "", "префикс выхода")
	ngl := fs.Int("ngl", 0, "слоёв на GPU")

	if err := fs.Parse(args); err != nil {
		return err
	}

	dump, meta, err := collectLayersDump(*modelPath, *prompt, *chatMode, *ngl)
	if err != nil {
		return err
	}

	prefix := *out
	if prefix == "" {
		base := strings.TrimSuffix(filepath.Base(*modelPath), filepath.Ext(*modelPath))
		mode := "raw"
		if *chatMode {
			mode = "chat"
		}

		prefix = filepath.Join(filepath.Dir(*modelPath), base+"_"+mode+"_layers")
	}

	if err := debug.SaveLayersDump(prefix, meta, dump); err != nil {
		return err
	}

	fmt.Printf("записано %s.bin / %s.json (embd=%d layers=%d)\n", prefix, prefix, len(dump.Embed), len(dump.Layers))

	return nil
}

func collectLayersDump(modelPath, prompt string, chat bool, ngl int) (debug.LayersDump, debug.LayersMeta, error) {
	engine, err := gogguf.Load(modelPath, gogguf.LoadOptions{NGL: ngl})
	if err != nil {
		return debug.LayersDump{}, debug.LayersMeta{}, err
	}

	ctx, err := engine.NewContext()
	if err != nil {
		return debug.LayersDump{}, debug.LayersMeta{}, err
	}

	text := prompt
	if chat {
		text, err = gogguf.FormatChatUser(prompt, gogguf.ChatOptions{
			Metadata: engine.Metadata(),
		})
		if err != nil {
			return debug.LayersDump{}, debug.LayersMeta{}, err
		}
	}

	ids, err := ctx.EncodeForInference(text)
	if err != nil {
		return debug.LayersDump{}, debug.LayersMeta{}, err
	}

	var dump debug.LayersDump
	if !setLayerHiddenHooks(engine, &dump) {
		return debug.LayersDump{}, debug.LayersMeta{}, fmt.Errorf("модель не поддерживает debug hooks")
	}

	engine.Model.ResetCache()
	logits, err := engine.Model.Forward(ids, 0)
	if err != nil {
		return debug.LayersDump{}, debug.LayersMeta{}, err
	}

	backend := "cpu"
	if ngl > 0 {
		backend = "cuda"
	}

	meta := debug.LayersMeta{
		Model:   filepath.Base(modelPath),
		Prompt:  prompt,
		Chat:    chat,
		Tokens:  ids,
		Greedy:  gogguf.Greedy(logits),
		Backend: backend,
		NGL:     ngl,
	}

	return dump, meta, nil
}

func setLayerHiddenHooks(engine *gogguf.Engine, dump *debug.LayersDump) bool {
	onEmbed := func(x []float32) {
		dump.Embed = append([]float32(nil), x...)
	}

	onLayer := func(_ int, x []float32) {
		dump.Layers = append(dump.Layers, append([]float32(nil), x...))
	}

	switch m := engine.Model.(type) {
	case interface{ SetDebugHooks(*qwen3.DebugHooks) }:
		m.SetDebugHooks(&qwen3.DebugHooks{
			OnEmbed: onEmbed,
			OnLayer: onLayer,
		})
		return true
	case interface{ SetDebugHooks(*llama.DebugHooks) }:
		m.SetDebugHooks(&llama.DebugHooks{
			OnEmbed: onEmbed,
			OnLayer: onLayer,
		})
		return true
	case interface{ SetDebugHooks(*mistral.DebugHooks) }:
		m.SetDebugHooks(&mistral.DebugHooks{
			OnEmbed: onEmbed,
			OnLayer: onLayer,
		})
		return true
	default:
		return false
	}
}
