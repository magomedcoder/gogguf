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
	host := fs.String("host", "127.0.0.1:8000", "адрес HTTP-сервера")
	ngl := fs.Int("ngl", 0, "число transformer-слоёв на GPU (CUDA, сборка: -tags cuda)")
	nBatch := fs.Int("b", 1, "размер chunk prefill (n_batch); >1 ускоряет prefill (Qwen3, работает и с -ngl)")
	fs.IntVar(nBatch, "n-batch", 1, "алиас -b")
	dev := fs.String("dev", "", "GPU для offload: \"1\" или \"0,1\" (multi-GPU split слоёв)")
	tensorSplit := fs.String("tensor-split", "", "пропорции слоёв по устройствам с -dev 0,1, например 0.6,0.4")
	apiKey := fs.String("api-key", "", "API key (Bearer / X-API-Key); пусто = без auth; /v1/health открыт")
	rateLimit := fs.Int("rate-limit", 0, "лимит запросов в минуту на IP (0 = без лимита); /v1/health не учитывается")

	if err := fs.Parse(args); err != nil {
		return err
	}

	devices, tsplit, err := gogguf.ParseGPUDevices(*dev, *tensorSplit)
	if err != nil {
		return err
	}

	path, err := resolveModelPath(modelPath, hfRepo)
	if err != nil {
		return fmt.Errorf("%w\nиспользование: gogguf serve -m файл.gguf|-hf owner/repo[:quant] [--host 127.0.0.1:8000]", err)
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

	srv := server.New(engine, path, server.Options{
		APIKey:             *apiKey,
		RateLimitPerMinute: *rateLimit,
	})

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	fmt.Fprintf(os.Stderr, "gogguf serve: %s (model: %s)", *host, path)
	if *ngl > 0 {
		fmt.Fprintf(os.Stderr, " GPU offload: %d слоёв на %s", *ngl, engine.GPUDescription())
	}

	if *nBatch > 1 {
		fmt.Fprintf(os.Stderr, " n_batch=%d", *nBatch)
	}

	if *apiKey != "" {
		fmt.Fprintf(os.Stderr, " api-key=on")
	}

	if *rateLimit > 0 {
		fmt.Fprintf(os.Stderr, " rate-limit=%d/min", *rateLimit)
	}

	fmt.Fprintln(os.Stderr)

	return srv.Run(ctx, *host)
}
