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
	modelPath := fs.String("m", "", "путь к файлу GGUF")
	addr := fs.String("addr", "127.0.0.1:8000", "адрес HTTP-сервера")
	ngl := fs.Int("ngl", 0, "число transformer-слоёв на GPU (CUDA, сборка: -tags cuda)")

	if err := fs.Parse(args); err != nil {
		return err
	}

	if *modelPath == "" {
		return fmt.Errorf("использование: gguf serve -m файл.gguf [--addr 127.0.0.1:8000]")
	}

	engine, err := gogguf.Load(*modelPath, gogguf.LoadOptions{
		NGL: *ngl,
	})
	if err != nil {
		return err
	}

	srv := server.New(engine, *modelPath)

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	fmt.Fprintf(os.Stderr, "gguf serve: %s (model: %s)\n", *addr, *modelPath)

	return srv.Run(ctx, *addr)
}
