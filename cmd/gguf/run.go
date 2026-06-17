package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/magomedcoder/gguf.go"
	chattmpl "github.com/magomedcoder/gguf.go/pkg/chat"
)

// runRun выполняет генерацию текста
func runRun(args []string) error {
	fs := flag.NewFlagSet("run", flag.ContinueOnError)
	modelPath := fs.String("m", "", "путь к файлу GGUF")
	prompt := fs.String("p", "", "текст промпта")
	maxTokens := fs.Int("n", 128, "максимум новых токенов")
	temp := fs.Float64("temp", 0, "температура (0 = greedy)")
	topK := fs.Int("top-k", 0, "top-k sampling (0 = выключено)")
	topP := fs.Float64("top-p", 1, "top-p nucleus sampling (1 = выключено)")
	seed := fs.Uint64("seed", 0, "seed PRNG для sampling")
	chat := fs.Bool("chat", false, "обернуть промпт в Qwen chat template")
	thinking := fs.Bool("thinking", false, "Qwen3: включить режим размышления (с --chat)")
	ngl := fs.Int("ngl", 0, "число transformer-слоёв на GPU (CUDA, сборка: -tags cuda)")

	if err := fs.Parse(args); err != nil {
		return err
	}

	if *modelPath == "" {
		return fmt.Errorf("использование: gguf run -m файл.gguf -p \"промпт\" [-n 128] [--temp 0.7] [--top-k 40] [--top-p 0.9] [--seed 42]")
	}
	if *prompt == "" {
		return fmt.Errorf("укажите промпт через -p")
	}

	promptText := *prompt

	engine, err := gguf.Load(*modelPath, gguf.LoadOptions{NGL: *ngl})
	if err != nil {
		return err
	}
	if *ngl > 0 {
		fmt.Fprintf(os.Stderr, "GPU offload: %d слоёв\n", *ngl)
	}

	if *chat {
		var err error
		promptText, err = chattmpl.FormatUser(*prompt, chattmpl.Options{
			Metadata: engine.Metadata(),
			Thinking: thinking,
		})
		if err != nil {
			return err
		}
	}

	ctx, err := engine.NewContext()
	if err != nil {
		return err
	}

	samp := gguf.NewSampler(gguf.SamplerConfig{
		Temp: float32(*temp),
		TopK: *topK,
		TopP: float32(*topP),
		Seed: *seed,
	})

	err = ctx.GenerateStream(promptText, gguf.GenerateParams{
		MaxTokens: *maxTokens,
		Sampler:   samp,
	}, os.Stdout)
	if err != nil {
		return err
	}

	fmt.Fprintln(os.Stdout)
	return nil
}
