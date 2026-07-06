package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"github.com/magomedcoder/gogguf"
)

type output struct {
	Prompt string `json:"prompt,omitempty"`
	Tokens []int  `json:"tokens"`
}

func main() {
	model := flag.String("m", "", "путь к GGUF")
	chatUser := flag.String("chat", "", "user-сообщение (chat template)")
	prompt := flag.String("p", "", "сырой промпт (без chat template)")
	maxTokens := flag.Int("n", 50, "число decode-токенов")
	flag.Parse()

	if *model == "" {
		if len(flag.Args()) > 0 {
			*model = flag.Args()[0]
		} else {
			fmt.Fprintln(os.Stderr, "usage: greedy -m model.gguf [--chat USER | -p PROMPT] [-n N]")
			os.Exit(2)
		}
	}

	engine, err := gogguf.Load(*model, gogguf.LoadOptions{})
	if err != nil {
		fatal(err)
	}

	text := *prompt
	if *chatUser != "" {
		text, err = gogguf.FormatChatUser(*chatUser, gogguf.ChatOptions{
			Metadata: engine.Metadata(),
		})
		if err != nil {
			fatal(err)
		}
	}

	if text == "" {
		fmt.Fprintln(os.Stderr, "укажите --chat или -p")
		os.Exit(2)
	}

	ctx, err := engine.NewContext()
	if err != nil {
		fatal(err)
	}

	sess, err := ctx.StartGeneration(text)
	if err != nil {
		fatal(err)
	}

	if err := sess.GenerateSteps(gogguf.GenerateParams{
		MaxTokens: *maxTokens,
		Sampler:   gogguf.Greedy,
	}); err != nil {
		fatal(err)
	}

	out := output{
		Prompt: text,
		Tokens: sess.GeneratedTokens(),
	}

	enc := json.NewEncoder(os.Stdout)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(out); err != nil {
		fatal(err)
	}
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, err)
	os.Exit(1)
}
