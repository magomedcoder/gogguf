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

const benchUsage = `bench - измерение скорости inference (prefill, decode, TTFT)

Использование:
  tools bench -m модель.gguf -p "промпт" [-n 128] [-ngl 0] [--runs 3] [--warmup 1]

`

// benchResult - метрики одного прогона бенчмарка
type benchResult struct {
	PromptTokens int     `json:"prompt_tokens"`
	DecodeTokens int     `json:"decode_tokens"`
	PrefillMS    float64 `json:"prefill_ms"`
	TTFTMS       float64 `json:"ttft_ms"`
	DecodeMS     float64 `json:"decode_ms"`
	TotalMS      float64 `json:"total_ms"`
	PrefillTPS   float64 `json:"prefill_tps"`
	DecodeTPS    float64 `json:"decode_tps"`
	TotalTPS     float64 `json:"total_tps"`
}

// runBench выполняет бенчмарк скорости inference
func runBench(args []string) error {
	fs := flag.NewFlagSet("bench", flag.ContinueOnError)
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
		fmt.Fprint(os.Stderr, benchUsage)
		fs.PrintDefaults()
	}

	if err := fs.Parse(args); err != nil {
		return err
	}

	if *modelPath == "" {
		fmt.Fprint(os.Stderr, benchUsage)
		return fmt.Errorf("укажите модель через -m")
	}

	loadStart := time.Now()
	engine, err := gogguf.Load(*modelPath, gogguf.LoadOptions{
		NGL: *ngl,
	})
	if err != nil {
		return err
	}
	loadMS := durationMS(time.Since(loadStart))

	promptText := *prompt
	if *chat {
		promptText, err = chattmpl.FormatUser(*prompt, chattmpl.Options{
			Metadata: engine.Metadata(),
			Thinking: thinking,
		})
		if err != nil {
			return err
		}
	}

	ctx, err := engine.NewContext()
	if err != nil {
		return err
	}

	for i := 0; i < *warmup; i++ {
		if _, err := runBenchOnce(ctx, promptText, *maxTokens); err != nil {
			return err
		}
	}

	results := make([]benchResult, 0, *runs)
	for i := 0; i < *runs; i++ {
		res, err := runBenchOnce(ctx, promptText, *maxTokens)
		if err != nil {
			return err
		}
		results = append(results, res)
	}

	avg := averageBench(results)

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
		return enc.Encode(out)
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

	return nil
}

// runBenchOnce выполняет один прогон: prefill + greedy decode maxTokens шагов
func runBenchOnce(ctx *gogguf.Context, prompt string, maxTokens int) (benchResult, error) {
	totalStart := time.Now()

	prefillStart := time.Now()
	sess, err := ctx.StartGeneration(prompt)
	if err != nil {
		return benchResult{}, err
	}
	prefillDur := time.Since(prefillStart)

	decodeStart := time.Now()
	decodeTokens := 0
	for range maxTokens {
		id, err := sess.DecodeStep(gogguf.Greedy)
		if err != nil {
			return benchResult{}, err
		}

		if id < 0 {
			break
		}

		decodeTokens++
	}

	decodeDur := time.Since(decodeStart)
	totalDur := time.Since(totalStart)

	promptTokens := sess.PromptTokenCount()
	res := benchResult{
		PromptTokens: promptTokens,
		DecodeTokens: decodeTokens,
		PrefillMS:    durationMS(prefillDur),
		TTFTMS:       durationMS(prefillDur),
		DecodeMS:     durationMS(decodeDur),
		TotalMS:      durationMS(totalDur),
	}

	if promptTokens > 0 && prefillDur > 0 {
		res.PrefillTPS = float64(promptTokens) / prefillDur.Seconds()
	}

	if decodeTokens > 0 && decodeDur > 0 {
		res.DecodeTPS = float64(decodeTokens) / decodeDur.Seconds()
	}

	totalTokens := promptTokens + decodeTokens
	if totalTokens > 0 && totalDur > 0 {
		res.TotalTPS = float64(totalTokens) / totalDur.Seconds()
	}

	return res, nil
}

// averageBench усредняет несколько прогонов (prefill/decode/tps)
func averageBench(results []benchResult) benchResult {
	if len(results) == 0 {
		return benchResult{}
	}

	if len(results) == 1 {
		return results[0]
	}

	var avg benchResult
	for _, r := range results {
		avg.PromptTokens = r.PromptTokens
		avg.DecodeTokens += r.DecodeTokens
		avg.PrefillMS += r.PrefillMS
		avg.TTFTMS += r.TTFTMS
		avg.DecodeMS += r.DecodeMS
		avg.TotalMS += r.TotalMS
		avg.PrefillTPS += r.PrefillTPS
		avg.DecodeTPS += r.DecodeTPS
		avg.TotalTPS += r.TotalTPS
	}

	n := float64(len(results))
	avg.DecodeTokens = int(float64(avg.DecodeTokens) / n)
	avg.PrefillMS /= n
	avg.TTFTMS /= n
	avg.DecodeMS /= n
	avg.TotalMS /= n
	avg.PrefillTPS /= n
	avg.DecodeTPS /= n
	avg.TotalTPS /= n

	return avg
}

func durationMS(d time.Duration) float64 {
	return float64(d) / float64(time.Millisecond)
}

func round2(v float64) float64 {
	return float64(int(v*100+0.5)) / 100
}
