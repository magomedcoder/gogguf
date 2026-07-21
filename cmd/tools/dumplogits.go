package main

import (
	"flag"
	"fmt"
	"path/filepath"
	"strings"

	"github.com/magomedcoder/gogguf"
	"github.com/magomedcoder/gogguf/pkg/debug"
)

// runDumpLogits пишет полный vocab logits после prefill (.bin + .json)
func runDumpLogits(args []string) error {
	fs := flag.NewFlagSet("dumplogits", flag.ContinueOnError)
	modelPath := fs.String("m", "./models/Qwen3-0.6B-Q8_0.gguf", "путь к GGUF")
	prompt := fs.String("p", "Hello", "промпт")
	chatMode := fs.Bool("chat", false, "chat template")
	out := fs.String("o", "", "префикс выхода (без расширения); по умолчанию рядом с моделью")
	ngl := fs.Int("ngl", 0, "слоёв на GPU")
	topN := fs.Int("top", 5, "top-N в JSON meta")

	if err := fs.Parse(args); err != nil {
		return err
	}

	engine, err := gogguf.Load(*modelPath, gogguf.LoadOptions{
		NGL: *ngl,
	})
	if err != nil {
		return err
	}

	ctx, err := engine.NewContext()
	if err != nil {
		return err
	}

	text := *prompt
	if *chatMode {
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

	engine.Model.ResetCache()
	logits, err := engine.Model.Forward(ids, 0)
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

		prefix = filepath.Join(filepath.Dir(*modelPath), base+"_"+mode+"_logits")
	}

	backend := "cpu"
	if *ngl > 0 {
		backend = "cuda"
	}

	meta := debug.LogitsMeta{
		Model:   filepath.Base(*modelPath),
		Prompt:  *prompt,
		Chat:    *chatMode,
		Tokens:  ids,
		Greedy:  gogguf.Greedy(logits),
		Top:     debug.TopLogits(logits, *topN),
		Backend: backend,
		NGL:     *ngl,
	}

	if err := debug.SaveLogitsDump(prefix, meta, logits); err != nil {
		return err
	}

	fmt.Printf("записано %s.bin / %s.json (vocab=%d greedy=%d)\n", prefix, prefix, len(logits), meta.Greedy)

	return nil
}
