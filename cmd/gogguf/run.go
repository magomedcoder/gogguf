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

// runRun runs text generation.
func runRun(args []string) error {
	fs := flag.NewFlagSet("run", flag.ContinueOnError)
	var modelPath, hfRepo string
	fs.StringVar(&modelPath, "m", "", "path to GGUF file")
	fs.StringVar(&hfRepo, "hf", "", "Hugging Face repo[:quant], e.g. Qwen/Qwen3-4B-GGUF:Q8_0")
	fs.StringVar(&hfRepo, "hf-repo", "", "alias for -hf")
	prompt := fs.String("p", "", "prompt text")
	maxTokens := fs.Int("n", 128, "max new tokens")
	temp := fs.Float64("temp", 0, "temperature (0 = greedy)")
	topK := fs.Int("top-k", 0, "top-k sampling (0 = off)")
	topP := fs.Float64("top-p", 1, "top-p nucleus sampling (1 = off)")
	minP := fs.Float64("min-p", 0, "min-p sampling (0 = off)")
	repeatPenalty := fs.Float64("repeat-penalty", 1, "token repeat penalty (1 = off)")
	repeatLastN := fs.Int("repeat-last-n", 64, "history window for repeat-penalty")
	seed := fs.Uint64("seed", 0, "PRNG seed for sampling")
	chat := fs.Bool("chat", false, "wrap prompt in Qwen chat template")
	thinking := fs.Bool("thinking", false, "Qwen3: enable thinking mode (with --chat)")
	interactive := fs.Bool("i", false, "interactive mode (REPL)")
	ngl := fs.Int("ngl", 0, "number of transformer layers on GPU (CUDA, build: -tags cuda)")
	nBatch := fs.Int("b", 1, "prefill chunk size (n_batch); >1 speeds up prefill (Qwen3, works with -ngl)")
	fs.IntVar(nBatch, "n-batch", 1, "alias for -b")
	dev := fs.String("dev", "", "GPU for offload: \"1\" or \"0,1\" (multi-GPU layer split)")
	tensorSplit := fs.String("tensor-split", "", "layer proportions per device with -dev 0,1, e.g. 0.6,0.4")

	if err := fs.Parse(args); err != nil {
		return err
	}

	devices, tsplit, err := gogguf.ParseGPUDevices(*dev, *tensorSplit)
	if err != nil {
		return err
	}

	path, err := resolveModelPath(modelPath, hfRepo)
	if err != nil {
		return fmt.Errorf("%w\nusage: gogguf run -m file.gguf|-hf owner/repo[:quant] -p \"prompt\" [-n 128] [-i]", err)
	}
	if !*interactive && *prompt == "" {
		return fmt.Errorf("specify prompt via -p or use -i")
	}

	engine, err := gogguf.Load(path, gogguf.LoadOptions{
		NGL:         *ngl,
		NBatch:      *nBatch,
		GPUDevices:  devices,
		TensorSplit: tsplit,
	})
	if err != nil {
		return err
	}
	if *ngl > 0 {
		fmt.Fprintf(os.Stderr, "GPU offload: %d layers on %s\n", *ngl, engine.GPUDescription())
	}
	if *nBatch > 1 {
		fmt.Fprintf(os.Stderr, "n_batch: %d\n", *nBatch)
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
	fmt.Fprintln(os.Stderr, "Interactive mode. Empty line or Ctrl+D to exit")
	if chat {
		fmt.Fprintln(os.Stderr, "Commands: /clear - reset conversation history")
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
			fmt.Fprintln(os.Stderr, "History cleared")
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
