package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/magomedcoder/gogguf"
	chattmpl "github.com/magomedcoder/gogguf/pkg/chat"
	"github.com/magomedcoder/gogguf/pkg/format"
)

const benchUsage = `bench - измерение скорости inference (prefill, decode, TTFT)

Использование:
  tools bench -m модель.gguf -p "промпт" [-n 128] [-ngl 0] [--runs 3] [--warmup 1]
  tools bench -m модель.gguf -p "промпт" -ngl 28 --compare   # CPU vs GPU

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
	ctxLen := fs.Int("c", 0, "макс. длина GPU KV-cache (0 = авто, до 4096)")
	chat := fs.Bool("chat", false, "обернуть промпт в chat template")
	thinking := fs.Bool("thinking", false, "Qwen3: режим размышления (с --chat)")
	runs := fs.Int("runs", 1, "число прогонов для усреднения")
	warmup := fs.Int("warmup", 1, "число прогревочных прогонов (без вывода)")
	jsonOut := fs.Bool("json", false, "вывод в JSON")
	compare := fs.Bool("compare", false, "сравнить CPU (ngl=0) и GPU (-ngl)")

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

	if *compare {
		return runBenchCompare(*modelPath, *prompt, *maxTokens, *ngl, *ctxLen, *chat, *thinking, *runs, *warmup, *jsonOut)
	}

	return runBenchSingle(*modelPath, *prompt, *maxTokens, *ngl, *ctxLen, *chat, *thinking, *runs, *warmup, *jsonOut)
}

func runBenchSingle(modelPath, prompt string, maxTokens, ngl, ctxLen int, chat, thinking bool, runs, warmup int, jsonOut bool) error {
	loadStart := time.Now()
	engine, err := gogguf.Load(modelPath, gogguf.LoadOptions{
		NGL:       ngl,
		GPUMaxSeq: ctxLen,
	})
	if err != nil {
		return err
	}
	loadMS := durationMS(time.Since(loadStart))

	promptText, err := resolveBenchPrompt(engine, prompt, chat, thinking)
	if err != nil {
		return err
	}

	ctx, err := engine.NewContext()
	if err != nil {
		return err
	}

	avg, err := measureBench(ctx, promptText, maxTokens, runs, warmup)
	if err != nil {
		return err
	}

	if jsonOut {
		return writeBenchJSON(map[string]any{
			"model":         modelPath,
			"ngl":           ngl,
			"gpu_max_seq":   ctxLen,
			"load_ms":       loadMS,
			"runs":          runs,
			"warmup":        warmup,
			"prompt_tokens": avg.PromptTokens,
			"decode_tokens": avg.DecodeTokens,
			"prefill_ms":    round2(avg.PrefillMS),
			"ttft_ms":       round2(avg.TTFTMS),
			"decode_ms":     round2(avg.DecodeMS),
			"total_ms":      round2(avg.TotalMS),
			"prefill_tps":   round2(avg.PrefillTPS),
			"decode_tps":    round2(avg.DecodeTPS),
			"total_tps":     round2(avg.TotalTPS),
		})
	}

	printBenchHuman(modelPath, ngl, loadMS, runs, avg)
	return nil
}

func runBenchCompare(modelPath, prompt string, maxTokens, ngl, ctxLen int, chat, thinking bool, runs, warmup int, jsonOut bool) error {
	if ngl <= 0 {
		layers, err := modelLayerCount(modelPath)
		if err != nil {
			return fmt.Errorf("compare: укажите -ngl > 0 или исправьте модель: %w", err)
		}

		ngl = layers
	}

	cpuEngine, err := gogguf.Load(modelPath, gogguf.LoadOptions{
		NGL: 0,
	})
	if err != nil {
		return fmt.Errorf("CPU load: %w", err)
	}

	cpuPrompt, err := resolveBenchPrompt(cpuEngine, prompt, chat, thinking)
	if err != nil {
		return err
	}

	cpuCtx, err := cpuEngine.NewContext()
	if err != nil {
		return err
	}

	cpuAvg, err := measureBench(cpuCtx, cpuPrompt, maxTokens, runs, warmup)
	if err != nil {
		return fmt.Errorf("CPU bench: %w", err)
	}

	gpuEngine, err := gogguf.Load(modelPath, gogguf.LoadOptions{
		NGL:       ngl,
		GPUMaxSeq: ctxLen,
	})
	if err != nil {
		return fmt.Errorf("GPU load (ngl=%d): %w", ngl, err)
	}

	gpuPrompt, err := resolveBenchPrompt(gpuEngine, prompt, chat, thinking)
	if err != nil {
		return err
	}

	gpuCtx, err := gpuEngine.NewContext()
	if err != nil {
		return err
	}

	gpuAvg, err := measureBench(gpuCtx, gpuPrompt, maxTokens, runs, warmup)
	if err != nil {
		return fmt.Errorf("GPU bench: %w", err)
	}

	decodeSpeedup := ratio(gpuAvg.DecodeTPS, cpuAvg.DecodeTPS)
	prefillSpeedup := ratio(gpuAvg.PrefillTPS, cpuAvg.PrefillTPS)
	gpuFaster := gpuAvg.DecodeTPS > cpuAvg.DecodeTPS

	if jsonOut {
		return writeBenchJSON(map[string]any{
			"model":             modelPath,
			"ngl":               ngl,
			"runs":              runs,
			"warmup":            warmup,
			"cpu":               benchResultJSON(cpuAvg),
			"gpu":               benchResultJSON(gpuAvg),
			"decode_speedup":    round2(decodeSpeedup),
			"prefill_speedup":   round2(prefillSpeedup),
			"gpu_decode_faster": gpuFaster,
		})
	}

	fmt.Printf("Модель: %s\n", modelPath)
	fmt.Printf("Сравнение CPU vs GPU (ngl=%d), прогонов=%d, decode=%d tok\n\n", ngl, runs, maxTokens)
	fmt.Printf("%-10s %12s %12s %12s %12s\n", "", "prefill t/s", "decode t/s", "TTFT ms", "total t/s")
	fmt.Printf("%-10s %12.1f %12.1f %12.1f %12.1f\n", "CPU", cpuAvg.PrefillTPS, cpuAvg.DecodeTPS, cpuAvg.TTFTMS, cpuAvg.TotalTPS)
	fmt.Printf("%-10s %12.1f %12.1f %12.1f %12.1f\n", "GPU", gpuAvg.PrefillTPS, gpuAvg.DecodeTPS, gpuAvg.TTFTMS, gpuAvg.TotalTPS)
	fmt.Printf("%-10s %12.2fx %12.2fx\n", "ускорение", prefillSpeedup, decodeSpeedup)
	fmt.Println()

	if gpuFaster {
		fmt.Printf("MVP: GPU decode быстрее CPU (%.1f vs %.1f tok/s)\n", gpuAvg.DecodeTPS, cpuAvg.DecodeTPS)
	} else {
		fmt.Printf("MVP: GPU decode НЕ быстрее CPU (%.1f vs %.1f tok/s, ускорение %.2fx)\n", gpuAvg.DecodeTPS, cpuAvg.DecodeTPS, decodeSpeedup)
	}

	return nil
}

func resolveBenchPrompt(engine *gogguf.Engine, prompt string, chat, thinking bool) (string, error) {
	if !chat {
		return prompt, nil
	}

	return chattmpl.FormatUser(prompt, chattmpl.Options{
		Metadata: engine.Metadata(),
		Thinking: &thinking,
	})
}

func measureBench(ctx *gogguf.Context, promptText string, maxTokens, runs, warmup int) (benchResult, error) {
	for range warmup {
		if _, err := runBenchOnce(ctx, promptText, maxTokens); err != nil {
			return benchResult{}, err
		}
	}

	results := make([]benchResult, 0, runs)
	for range runs {
		res, err := runBenchOnce(ctx, promptText, maxTokens)
		if err != nil {
			return benchResult{}, err
		}

		results = append(results, res)
	}

	return averageBench(results), nil
}

func modelLayerCount(path string) (int, error) {
	r, err := format.OpenFile(path)
	if err != nil {
		return 0, err
	}

	arch, err := r.Metadata.String("general.architecture")
	if err != nil {
		return 0, err
	}

	return r.Metadata.Int(arch + ".block_count")
}

func ratio(num, den float64) float64 {
	if den <= 0 {
		return 0
	}

	return num / den
}

func benchResultJSON(r benchResult) map[string]any {
	return map[string]any{
		"prompt_tokens": r.PromptTokens,
		"decode_tokens": r.DecodeTokens,
		"prefill_ms":    round2(r.PrefillMS),
		"ttft_ms":       round2(r.TTFTMS),
		"decode_ms":     round2(r.DecodeMS),
		"total_ms":      round2(r.TotalMS),
		"prefill_tps":   round2(r.PrefillTPS),
		"decode_tps":    round2(r.DecodeTPS),
		"total_tps":     round2(r.TotalTPS),
	}
}

func writeBenchJSON(out map[string]any) error {
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	return enc.Encode(out)
}

func printBenchHuman(modelPath string, ngl int, loadMS float64, runs int, avg benchResult) {
	fmt.Printf("Модель: %s\n", modelPath)
	if ngl > 0 {
		fmt.Printf("GPU offload: %d слоёв\n", ngl)
	}

	fmt.Printf("Загрузка: %.1f ms\n", loadMS)
	if runs > 1 {
		fmt.Printf("Прогонов: %d (усреднение)\n", runs)
	}

	fmt.Printf("Prefill (%d tok): %.1f ms (%.1f tok/s)\n", avg.PromptTokens, avg.PrefillMS, avg.PrefillTPS)
	fmt.Printf("TTFT: %.1f ms\n", avg.TTFTMS)
	fmt.Printf("Decode (%d tok): %.1f ms (%.1f tok/s)\n", avg.DecodeTokens, avg.DecodeMS, avg.DecodeTPS)
	fmt.Printf("Итого: %.1f ms (%.1f tok/s)\n", avg.TotalMS, avg.TotalTPS)
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
	var firstTokenDur time.Duration
	for range maxTokens {
		id, err := sess.DecodeStep(gogguf.Greedy)
		if err != nil {
			return benchResult{}, err
		}

		if id < 0 {
			break
		}

		decodeTokens++
		if decodeTokens == 1 {
			firstTokenDur = time.Since(prefillStart)
		}
	}

	decodeDur := time.Since(decodeStart)
	totalDur := time.Since(totalStart)

	promptTokens := sess.PromptTokenCount()
	ttft := prefillDur
	if firstTokenDur > 0 {
		ttft = firstTokenDur
	}

	res := benchResult{
		PromptTokens: promptTokens,
		DecodeTokens: decodeTokens,
		PrefillMS:    durationMS(prefillDur),
		TTFTMS:       durationMS(ttft),
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
