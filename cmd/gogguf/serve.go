package main

import (
	"context"
	"flag"
	"fmt"
	"github.com/magomedcoder/gogguf"
	"github.com/magomedcoder/gogguf/pkg/server"
	"os"
	"os/signal"
	"syscall"
)

func runServe(args []string) error {
	fs := flag.NewFlagSet("serve", flag.ContinueOnError)
	var modelPath, hfRepo string
	fs.StringVar(&modelPath, "m", "", "path to GGUF file")
	fs.StringVar(&hfRepo, "hf", "", "Hugging Face repo[:quant], e.g. Qwen/Qwen3-4B-GGUF:Q8_0")
	fs.StringVar(&hfRepo, "hf-repo", "", "alias for -hf")
	host := fs.String("host", "127.0.0.1:8000", "HTTP server address")
	ngl := fs.Int("ngl", 0, "number of transformer layers on GPU (CUDA, build: -tags cuda)")
	nBatch := fs.Int("b", 1, "prefill chunk size (n_batch); >1 speeds up prefill (Qwen3, works with -ngl)")
	fs.IntVar(nBatch, "n-batch", 1, "alias for -b")
	dev := fs.String("dev", "", "GPU for offload: \"1\" or \"0,1\" (multi-GPU layer split)")
	tensorSplit := fs.String("tensor-split", "", "layer proportions per device with -dev 0,1, e.g. 0.6,0.4")
	apiKey := fs.String("api-key", "", "API key (Bearer / X-API-Key); empty = no auth; /v1/health is public")
	rateLimit := fs.Int("rate-limit", 0, "requests per minute per IP (0 = unlimited); /v1/health excluded")

	if err := fs.Parse(args); err != nil {
		return err
	}

	devices, tsplit, err := gogguf.ParseGPUDevices(*dev, *tensorSplit)
	if err != nil {
		return err
	}

	path, err := resolveModelPath(modelPath, hfRepo)
	if err != nil {
		return fmt.Errorf("%w\nusage: gogguf serve -m file.gguf|-hf owner/repo[:quant] [--host 127.0.0.1:8000]", err)
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
		fmt.Fprintf(os.Stderr, " GPU offload: %d layers on %s", *ngl, engine.GPUDescription())
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
