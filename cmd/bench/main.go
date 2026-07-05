package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/magomedcoder/gogguf"
	chattmpl "github.com/magomedcoder/gogguf/pkg/chat"
)

const usage = `bench - измерение скорости inference (prefill, decode, TTFT)

Использование:
  bench -m модель.gguf -p "промпт" [-n 128] [-ngl 0] [--runs 3] [--warmup 1]

`

func main() {
	fs := flag.NewFlagSet("bench", flag.ExitOnError)
	modelPath := fs.String("m", "", "путь к файлу GGUF")
	prompt := fs.String("p", "Hello", "текст промпта")
	maxTokens := fs.Int("n", 128, "число decode-токенов для замера")
	ngl := fs.Int("ngl", 0, "число transformer-слоёв на GPU (CUDA, -tags cuda)")
	chat := fs.Bool("chat", false, "обернуть промпт в chat template")
	thinking := fs.Bool("thinking", false, "Qwen3: режим размышления (с --chat)")
	runs := fs.Int("runs", 1, "число прогонов для усреднения")
	warmup := fs.Int("warmup", 1, "число прогревочных прогонов (без вывода)")
	jsonOut := fs.Bool("json", false, "вывод в JSON")

	fs.Usage = func() {
		fmt.Fprint(os.Stderr, usage)
		fs.PrintDefaults()
	}

	if err := fs.Parse(os.Args[1:]); err != nil {
		os.Exit(1)
	}

	if *modelPath == "" {
		fmt.Fprint(os.Stderr, usage)
		os.Exit(1)
	}

	loadStart := time.Now()
	engine, err := gogguf.Load(*modelPath, gogguf.LoadOptions{
		NGL: *ngl,
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	loadMS := ms(time.Since(loadStart))

	promptText := *prompt
	if *chat {
		promptText, err = chattmpl.FormatUser(*prompt, chattmpl.Options{
			Metadata: engine.Metadata(),
			Thinking: thinking,
		})

		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
	}

	ctx, err := engine.NewContext()
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	for i := 0; i < *warmup; i++ {
		if _, err := Run(ctx, promptText, *maxTokens); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
	}

	results := make([]Result, 0, *runs)
	for i := 0; i < *runs; i++ {
		res, err := Run(ctx, promptText, *maxTokens)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		results = append(results, res)
	}

	avg := Average(results)

	if *jsonOut {
		out := map[string]any{
			"model":         *modelPath,
			"ngl":           *ngl,
			"load_ms":       loadMS,
			"runs":          *runs,
			"warmup":        *warmup,
			"prompt_tokens": avg.PromptTokens,
			"decode_tokens": avg.DecodeTokens,
			"prefill_ms":    round2(avg.PrefillMS),
			"ttft_ms":       round2(avg.TTFTMS),
			"decode_ms":     round2(avg.DecodeMS),
			"total_ms":      round2(avg.TotalMS),
			"prefill_tps":   round2(avg.PrefillTPS),
			"decode_tps":    round2(avg.DecodeTPS),
			"total_tps":     round2(avg.TotalTPS),
		}
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		_ = enc.Encode(out)
		return
	}

	fmt.Printf("Модель: %s\n", *modelPath)
	if *ngl > 0 {
		fmt.Printf("GPU offload: %d слоёв\n", *ngl)
	}

	fmt.Printf("Загрузка: %.1f ms\n", loadMS)

	if *runs > 1 {
		fmt.Printf("Прогонов: %d (усреднение)\n", *runs)
	}

	fmt.Printf("Prefill (%d tok): %.1f ms (%.1f tok/s)\n", avg.PromptTokens, avg.PrefillMS, avg.PrefillTPS)
	fmt.Printf("TTFT: %.1f ms\n", avg.TTFTMS)
	fmt.Printf("Decode (%d tok): %.1f ms (%.1f tok/s)\n", avg.DecodeTokens, avg.DecodeMS, avg.DecodeTPS)
	fmt.Printf("Итого: %.1f ms (%.1f tok/s)\n", avg.TotalMS, avg.TotalTPS)
}

func round2(v float64) float64 {
	return float64(int(v*100+0.5)) / 100
}
