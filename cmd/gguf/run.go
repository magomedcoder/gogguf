package main

import (
	"bufio"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

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
	minP := fs.Float64("min-p", 0, "min-p sampling (0 = выключено)")
	repeatPenalty := fs.Float64("repeat-penalty", 1, "штраф за повтор токенов (1 = выключено)")
	repeatLastN := fs.Int("repeat-last-n", 64, "окно истории для repeat-penalty")
	seed := fs.Uint64("seed", 0, "seed PRNG для sampling")
	chat := fs.Bool("chat", false, "обернуть промпт в Qwen chat template")
	thinking := fs.Bool("thinking", false, "Qwen3: включить режим размышления (с --chat)")
	interactive := fs.Bool("i", false, "интерактивный режим (REPL)")
	ngl := fs.Int("ngl", 0, "число transformer-слоёв на GPU (CUDA, сборка: -tags cuda)")

	if err := fs.Parse(args); err != nil {
		return err
	}

	if *modelPath == "" {
		return fmt.Errorf("использование: gguf run -m файл.gguf -p \"промпт\" [-n 128] [-i]")
	}
	if !*interactive && *prompt == "" {
		return fmt.Errorf("укажите промпт через -p или используйте -i")
	}

	engine, err := gguf.Load(*modelPath, gguf.LoadOptions{NGL: *ngl})
	if err != nil {
		return err
	}
	if *ngl > 0 {
		fmt.Fprintf(os.Stderr, "GPU offload: %d слоёв\n", *ngl)
	}

	ctx, err := engine.NewContext()
	if err != nil {
		return err
	}

	samp := gguf.NewSampler(gguf.SamplerConfig{
		Temp: float32(*temp),
		TopK: *topK,
		TopP: float32(*topP),
		MinP: float32(*minP),
		Seed: *seed,
	})

	genParams := gguf.GenerateParams{
		MaxTokens:     *maxTokens,
		Sampler:       samp,
		RepeatPenalty: float32(*repeatPenalty),
		RepeatLastN:   *repeatLastN,
	}

	if *interactive {
		return runInteractive(ctx, engine, *chat, thinking, genParams, os.Stdin, os.Stdout)
	}

	promptText, err := formatPrompt(engine, *prompt, *chat, thinking)
	if err != nil {
		return err
	}

	if err := ctx.GenerateStream(promptText, genParams, os.Stdout); err != nil {
		return err
	}

	fmt.Fprintln(os.Stdout)
	return nil
}

func formatPrompt(engine *gguf.Engine, user string, chat bool, thinking *bool) (string, error) {
	if !chat {
		return user, nil
	}

	return chattmpl.FormatUser(user, chattmpl.Options{
		Metadata: engine.Metadata(),
		Thinking: thinking,
	})
}

func runInteractive(ctx *gguf.Context, engine *gguf.Engine, chat bool, thinking *bool, params gguf.GenerateParams, in io.Reader, out io.Writer) error {
	fmt.Fprintln(os.Stderr, "Интерактивный режим. Пустая строка или Ctrl+D - выход")
	scanner := bufio.NewScanner(in)

	for {
		fmt.Fprint(os.Stderr, "> ")
		if !scanner.Scan() {
			break
		}

		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			break
		}

		prompt, err := formatPrompt(engine, line, chat, thinking)
		if err != nil {
			return err
		}

		if err := ctx.GenerateStream(prompt, params, out); err != nil {
			return err
		}

		fmt.Fprintln(out)
	}

	return scanner.Err()
}
