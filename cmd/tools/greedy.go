package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"github.com/magomedcoder/gogguf"
)

type greedyOutput struct {
	Prompt string `json:"prompt,omitempty"`
	Tokens []int  `json:"tokens"`
}

// runGreedy генерирует N токенов greedy и печатает JSON
func runGreedy(args []string) error {
	fs := flag.NewFlagSet("greedy", flag.ContinueOnError)
	model := fs.String("m", "", "путь к GGUF")
	chatUser := fs.String("chat", "", "user-сообщение (chat template)")
	prompt := fs.String("p", "", "сырой промпт (без chat template)")
	maxTokens := fs.Int("n", 50, "число decode-токенов")

	if err := fs.Parse(args); err != nil {
		return err
	}

	if *model == "" {
		if len(fs.Args()) > 0 {
			*model = fs.Args()[0]
		} else {
			return fmt.Errorf("usage: tools greedy -m model.gguf [--chat USER | -p PROMPT] [-n N]")
		}
	}

	engine, err := gogguf.Load(*model, gogguf.LoadOptions{})
	if err != nil {
		return err
	}

	text := *prompt
	if *chatUser != "" {
		text, err = gogguf.FormatChatUser(*chatUser, gogguf.ChatOptions{
			Metadata: engine.Metadata(),
		})
		if err != nil {
			return err
		}
	}

	if text == "" {
		return fmt.Errorf("укажите --chat или -p")
	}

	ctx, err := engine.NewContext()
	if err != nil {
		return err
	}

	sess, err := ctx.StartGeneration(text)
	if err != nil {
		return err
	}

	if err := sess.GenerateSteps(gogguf.GenerateParams{
		MaxTokens: *maxTokens,
		Sampler:   gogguf.Greedy,
	}); err != nil {
		return err
	}

	out := greedyOutput{
		Prompt: text,
		Tokens: sess.GeneratedTokens(),
	}

	enc := json.NewEncoder(os.Stdout)
	enc.SetEscapeHTML(false)
	return enc.Encode(out)
}
