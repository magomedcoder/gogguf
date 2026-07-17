package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"syscall"

	"github.com/magomedcoder/gogguf"
	"github.com/magomedcoder/gogguf/server"
)

func runServe(args []string) error {
	fs := flag.NewFlagSet("serve", flag.ContinueOnError)
	var modelPath, hfRepo string
	fs.StringVar(&modelPath, "m", "", "путь к файлу GGUF")
	fs.StringVar(&hfRepo, "hf", "", "Hugging Face repo[:quant], например Qwen/Qwen3-0.6B-GGUF:Q8_0")
	fs.StringVar(&hfRepo, "hf-repo", "", "алиас -hf")
	addr := fs.String("addr", "127.0.0.1:8000", "адрес HTTP-сервера")
	ngl := fs.Int("ngl", 0, "число transformer-слоёв на GPU (CUDA, сборка: -tags cuda)")

	if err := fs.Parse(args); err != nil {
		return err
	}

	path, err := resolveModelPath(modelPath, hfRepo)
	if err != nil {
		return fmt.Errorf("%w\nиспользование: gogguf serve -m файл.gguf|-hf owner/repo[:quant] [--addr 127.0.0.1:8000]", err)
	}

	engine, err := gogguf.Load(path, gogguf.LoadOptions{
		NGL: *ngl,
	})
	if err != nil {
		return err
	}

	srv := server.New(engine, path)

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	fmt.Fprintf(os.Stderr, "gogguf serve: %s (model: %s)\n", *addr, path)

	return srv.Run(ctx, *addr)
}
