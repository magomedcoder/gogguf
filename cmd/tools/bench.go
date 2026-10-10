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

const benchUsage = `bench - measure inference speed (prefill, decode, TTFT)

Usage:
  tools bench -m model.gguf -p "prompt" [-n 128] [-ngl 0] [--runs 3] [--warmup 1]
  tools bench -m model.gguf -p "prompt" -ngl 28 --compare        # CPU vs GPU
  tools bench -m model.gguf -p "prompt" -ngl 28 -dev 1 -json     # GPU 1 + VRAM in JSON

`

// benchResult holds metrics from one benchmark run.
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

// runBench runs an inference speed benchmark.
func runBench(args []string) error {
	fs := flag.NewFlagSet("bench", flag.ContinueOnError)
	modelPath := fs.String("m", "", "path to GGUF file")
	prompt := fs.String("p", "Hello", "prompt text")
	maxTokens := fs.Int("n", 128, "number of decode tokens to measure")
	ngl := fs.Int("ngl", 0, "number of transformer layers on GPU (CUDA, -tags cuda)")
	nBatch := fs.Int("b", 1, "prefill chunk size (n_batch); >1 speeds up prefill (Qwen3, works with -ngl)")
	fs.IntVar(nBatch, "n-batch", 1, "alias for -b")
	ctxLen := fs.Int("c", 0, "max GPU KV-cache length (0 = auto, up to 4096)")
	dev := fs.String("dev", "", "GPU for offload: \"1\" or \"0,1\" (multi-GPU layer split)")
	tensorSplit := fs.String("tensor-split", "", "layer proportions per device with -dev 0,1, e.g. 0.6,0.4")
	chat := fs.Bool("chat", false, "wrap prompt in chat template")
	thinking := fs.Bool("thinking", false, "Qwen3: thinking mode (with --chat)")
	runs := fs.Int("runs", 1, "number of runs to average")
	warmup := fs.Int("warmup", 1, "number of warmup runs (no output)")
	jsonOut := fs.Bool("json", false, "JSON output")
	compare := fs.Bool("compare", false, "compare CPU (ngl=0) and GPU (-ngl)")

	fs.Usage = func() {
		fmt.Fprint(os.Stderr, benchUsage)
		fs.PrintDefaults()
	}

	if err := fs.Parse(args); err != nil {
		return err
	}

	if *modelPath == "" {
		fmt.Fprint(os.Stderr, benchUsage)
		return fmt.Errorf("specify model via -m")
	}

	devices, tsplit, err := gogguf.ParseGPUDevices(*dev, *tensorSplit)
	if err != nil {
		return err
	}

	gpuOpts := benchGPUOptions{devices: devices, tensorSplit: tsplit}

	if *compare {
		return runBenchCompare(*modelPath, *prompt, *maxTokens, *ngl, *ctxLen, *chat, *thinking, *runs, *warmup, *jsonOut, gpuOpts)
	}

	return runBenchSingle(*modelPath, *prompt, *maxTokens, *ngl, *nBatch, *ctxLen, *chat, *thinking, *runs, *warmup, *jsonOut, gpuOpts)
}

// benchGPUOptions selects devices for offload (-dev / -tensor-split).
type benchGPUOptions struct {
	devices     []int
	tensorSplit []float64
}

// vramMB is used/total backend VRAM in MB (0 on CPU runs).
type vramMB struct {
	used  float64
	total float64
}

// readVRAM samples VRAM after a run (cuMemGetInfo).
func readVRAM(engine *gogguf.Engine) vramMB {
	used, total, err := engine.VRAMInfo()
	if err != nil || total == 0 {
		return vramMB{}
	}

	const mb = 1024 * 1024
	return vramMB{
		used:  float64(used) / mb,
		total: float64(total) / mb,
	}
}

func runBenchSingle(modelPath, prompt string, maxTokens, ngl, nBatch, ctxLen int, chat, thinking bool, runs, warmup int, jsonOut bool, gpuOpts benchGPUOptions) error {
	loadStart := time.Now()
	engine, err := gogguf.Load(modelPath, gogguf.LoadOptions{
		NGL:         ngl,
		GPUMaxSeq:   ctxLen,
		NBatch:      nBatch,
		GPUDevices:  gpuOpts.devices,
		TensorSplit: gpuOpts.tensorSplit,
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

	vram := readVRAM(engine)

	if jsonOut {
		out := map[string]any{
			"model":         modelPath,
			"ngl":           ngl,
			"n_batch":       nBatch,
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
		}

		if vram.total > 0 {
			out["gpu"] = engine.GPUDescription()
			out["vram_used_mb"] = round2(vram.used)
			out["vram_total_mb"] = round2(vram.total)
		}

		return writeBenchJSON(out)
	}

	printBenchHuman(modelPath, ngl, loadMS, runs, avg, engine.GPUDescription(), vram)
	return nil
}

func runBenchCompare(modelPath, prompt string, maxTokens, ngl, ctxLen int, chat, thinking bool, runs, warmup int, jsonOut bool, gpuOpts benchGPUOptions) error {
	if ngl <= 0 {
		layers, err := modelLayerCount(modelPath)
		if err != nil {
			return fmt.Errorf("compare: specify -ngl > 0 or fix the model: %w", err)
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
		NGL:         ngl,
		GPUMaxSeq:   ctxLen,
		GPUDevices:  gpuOpts.devices,
		TensorSplit: gpuOpts.tensorSplit,
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
	vram := readVRAM(gpuEngine)

	if jsonOut {
		out := map[string]any{
			"model":             modelPath,
			"ngl":               ngl,
			"runs":              runs,
			"warmup":            warmup,
			"cpu":               benchResultJSON(cpuAvg),
			"gpu":               benchResultJSON(gpuAvg),
			"decode_speedup":    round2(decodeSpeedup),
			"prefill_speedup":   round2(prefillSpeedup),
			"gpu_decode_faster": gpuFaster,
		}

		if vram.total > 0 {
			out["gpu_device"] = gpuEngine.GPUDescription()
			out["vram_used_mb"] = round2(vram.used)
			out["vram_total_mb"] = round2(vram.total)
		}

		return writeBenchJSON(out)
	}

	fmt.Printf("Model: %s\n", modelPath)
	fmt.Printf("CPU vs GPU comparison (ngl=%d), runs=%d, decode=%d tok\n", ngl, runs, maxTokens)
	if vram.total > 0 {
		fmt.Printf("GPU: %s, VRAM %.0f / %.0f MB\n", gpuEngine.GPUDescription(), vram.used, vram.total)
	}

	fmt.Println()
	fmt.Printf("%-10s %12s %12s %12s %12s\n", "", "prefill t/s", "decode t/s", "TTFT ms", "total t/s")
	fmt.Printf("%-10s %12.1f %12.1f %12.1f %12.1f\n", "CPU", cpuAvg.PrefillTPS, cpuAvg.DecodeTPS, cpuAvg.TTFTMS, cpuAvg.TotalTPS)
	fmt.Printf("%-10s %12.1f %12.1f %12.1f %12.1f\n", "GPU", gpuAvg.PrefillTPS, gpuAvg.DecodeTPS, gpuAvg.TTFTMS, gpuAvg.TotalTPS)
	fmt.Printf("%-10s %12.2fx %12.2fx\n", "speedup", prefillSpeedup, decodeSpeedup)
	fmt.Println()

	if gpuFaster {
		fmt.Printf("MVP: GPU decode faster than CPU (%.1f vs %.1f tok/s)\n", gpuAvg.DecodeTPS, cpuAvg.DecodeTPS)
	} else {
		fmt.Printf("MVP: GPU decode NOT faster than CPU (%.1f vs %.1f tok/s, speedup %.2fx)\n", gpuAvg.DecodeTPS, cpuAvg.DecodeTPS, decodeSpeedup)
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

func printBenchHuman(modelPath string, ngl int, loadMS float64, runs int, avg benchResult, gpuName string, vram vramMB) {
	fmt.Printf("Model: %s\n", modelPath)
	if ngl > 0 {
		if gpuName != "" {
			fmt.Printf("GPU offload: %d layers on %s\n", ngl, gpuName)
		} else {
			fmt.Printf("GPU offload: %d layers\n", ngl)
		}
	}

	if vram.total > 0 {
		fmt.Printf("VRAM: %.0f / %.0f MB\n", vram.used, vram.total)
	}

	fmt.Printf("Load: %.1f ms\n", loadMS)
	if runs > 1 {
		fmt.Printf("Runs: %d (averaged)\n", runs)
	}

	fmt.Printf("Prefill (%d tok): %.1f ms (%.1f tok/s)\n", avg.PromptTokens, avg.PrefillMS, avg.PrefillTPS)
	fmt.Printf("TTFT: %.1f ms\n", avg.TTFTMS)
	fmt.Printf("Decode (%d tok): %.1f ms (%.1f tok/s)\n", avg.DecodeTokens, avg.DecodeMS, avg.DecodeTPS)
	fmt.Printf("Total: %.1f ms (%.1f tok/s)\n", avg.TotalMS, avg.TotalTPS)
}

// runBenchOnce runs one iteration: prefill + greedy decode for maxTokens steps.
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

// averageBench averages several runs (prefill/decode/tps).
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
