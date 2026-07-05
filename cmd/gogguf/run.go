package main

import (
	"bufio"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/magomedcoder/gogguf"
	chattmpl "github.com/magomedcoder/gogguf/pkg/chat"
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
		return fmt.Errorf("использование: gogguf run -m файл.gguf -p \"промпт\" [-n 128] [-i]")
	}
	if !*interactive && *prompt == "" {
		return fmt.Errorf("укажите промпт через -p или используйте -i")
	}

	engine, err := gogguf.Load(*modelPath, gogguf.LoadOptions{
		NGL: *ngl,
	})
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

	samp := gogguf.NewSampler(gogguf.SamplerConfig{
		Temp: float32(*temp),
		TopK: *topK,
		TopP: float32(*topP),
		MinP: float32(*minP),
		Seed: *seed,
	})

	genParams := gogguf.GenerateParams{
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

func formatPrompt(engine *gogguf.Engine, user string, chat bool, thinking *bool) (string, error) {
	if !chat {
		return user, nil
	}

	return formatChatHistory(engine.Metadata(), []chattmpl.Message{
		{
			Role:    "user",
			Content: user,
		},
	}, thinking)
}

func formatChatHistory(meta map[string]any, messages []chattmpl.Message, thinking *bool) (string, error) {
	return chattmpl.FormatMessages(messages, chattmpl.Options{
		Metadata: meta,
		Thinking: thinking,
	})
}

func runInteractive(ctx *gogguf.Context, engine *gogguf.Engine, chat bool, thinking *bool, params gogguf.GenerateParams, in io.Reader, out io.Writer) error {
	fmt.Fprintln(os.Stderr, "Интерактивный режим. Пустая строка или Ctrl+D - выход")
	if chat {
		fmt.Fprintln(os.Stderr, "Команды: /clear - сбросить историю диалога")
	}

	conv := ctx.NewConversation()
	var messages []chattmpl.Message
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
		if chat && line == "/clear" {
			messages = nil
			conv.Reset()
			fmt.Fprintln(os.Stderr, "История очищена")
			continue
		}

		var prompt string
		var err error
		if chat {
			messages = append(messages, chattmpl.Message{
				Role:    "user",
				Content: line,
			})
			prompt, err = formatChatHistory(engine.Metadata(), messages, thinking)
		} else {
			prompt, err = formatPrompt(engine, line, false, thinking)
		}
		if err != nil {
			return err
		}

		var reply strings.Builder
		if err := conv.GenerateStream(prompt, params, io.MultiWriter(out, &reply)); err != nil {
			if chat {
				messages = messages[:len(messages)-1]
			}
			return err
		}

		fmt.Fprintln(out)
		if chat {
			messages = append(messages, chattmpl.Message{
				Role:    "assistant",
				Content: reply.String(),
			})
		}
	}

	return scanner.Err()
}
