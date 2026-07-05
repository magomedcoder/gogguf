package main

import (
	"time"

	"github.com/magomedcoder/gogguf"
)

// Result - метрики одного прогона бенчмарка
type Result struct {
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

// Run выполняет один прогон: prefill + greedy decode maxTokens шагов
func Run(ctx *gogguf.Context, prompt string, maxTokens int) (Result, error) {
	totalStart := time.Now()

	prefillStart := time.Now()
	sess, err := ctx.StartGeneration(prompt)
	if err != nil {
		return Result{}, err
	}
	prefillDur := time.Since(prefillStart)

	decodeStart := time.Now()
	decodeTokens := 0
	for range maxTokens {
		id, err := sess.DecodeStep(gogguf.Greedy)
		if err != nil {
			return Result{}, err
		}

		if id < 0 {
			break
		}

		decodeTokens++
	}

	decodeDur := time.Since(decodeStart)
	totalDur := time.Since(totalStart)

	promptTokens := sess.PromptTokenCount()
	res := Result{
		PromptTokens: promptTokens,
		DecodeTokens: decodeTokens,
		PrefillMS:    ms(prefillDur),
		TTFTMS:       ms(prefillDur),
		DecodeMS:     ms(decodeDur),
		TotalMS:      ms(totalDur),
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

// Average усредняет несколько прогонов (prefill/decode/tps).
func Average(results []Result) Result {
	if len(results) == 0 {
		return Result{}
	}
	if len(results) == 1 {
		return results[0]
	}

	var avg Result
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

func ms(d time.Duration) float64 {
	return float64(d) / float64(time.Millisecond)
}
