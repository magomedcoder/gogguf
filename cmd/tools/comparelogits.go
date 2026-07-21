package main

import (
	"flag"
	"fmt"

	"github.com/magomedcoder/gogguf"
	"github.com/magomedcoder/gogguf/pkg/debug"
)

// runCompareLogits сравнивает два dump или CPU vs GPU prefill
func runCompareLogits(args []string) error {
	fs := flag.NewFlagSet("comparelogits", flag.ContinueOnError)
	aPath := fs.String("a", "", "префикс/путь dump A (.bin)")
	bPath := fs.String("b", "", "префикс/путь dump B (.bin)")
	modelPath := fs.String("m", "", "модель: сравнить CPU (ngl=0) vs GPU")
	prompt := fs.String("p", "Hello", "промпт для -m")
	chatMode := fs.Bool("chat", false, "chat template для -m")
	ngl := fs.Int("ngl", -1, "слоёв GPU (по умолчанию все)")
	tol := fs.Float64("tol", 1e-4, "допуск |diff|")
	align := fs.Bool("align", false, "выровнять A по max к B (инвариант softmax)")
	logprob := fs.Bool("logprob", false, "сравнивать log-softmax вместо raw logits")

	if err := fs.Parse(args); err != nil {
		return err
	}

	var a, b []float32
	var err error

	switch {
	case *modelPath != "":
		a, b, err = prefillCPUAndGPU(*modelPath, *prompt, *chatMode, *ngl)
	case *aPath != "" && *bPath != "":
		_, a, err = debug.LoadLogitsDump(*aPath)
		if err != nil {
			return fmt.Errorf("A: %w", err)
		}
		_, b, err = debug.LoadLogitsDump(*bPath)
	default:
		return fmt.Errorf("укажите -a/-b или -m")
	}
	if err != nil {
		return err
	}

	if len(a) != len(b) {
		return fmt.Errorf("разный vocab: A=%d B=%d", len(a), len(b))
	}

	if *logprob {
		debug.LogSoftmaxInPlace(a, a)
		debug.LogSoftmaxInPlace(b, b)
	} else if *align {
		a = debug.AlignByMax(a, b)
	}

	st := debug.DiffLogits(a, b, *tol)
	fmt.Printf("vocab=%d  max_abs=%.6g@%d  mean_abs=%.6g  rmse=%.6g  over_tol(%g)=%d\n", st.N, st.MaxAbs, st.MaxAbsIdx, st.MeanAbs, st.RMSE, st.Tol, st.OverTol)

	if st.OverTol > 0 {
		return fmt.Errorf("FAIL: %d позиций > tol %g (max_abs=%.6g @%d)", st.OverTol, *tol, st.MaxAbs, st.MaxAbsIdx)
	}

	fmt.Println("OK: full vocab в допуске")

	return nil
}

func prefillCPUAndGPU(modelPath, prompt string, chat bool, ngl int) (cpu, gpu []float32, err error) {
	cpu, err = prefillLogits(modelPath, prompt, chat, 0)
	if err != nil {
		return nil, nil, err
	}

	if ngl < 0 {
		ngl = 999
	}

	gpu, err = prefillLogits(modelPath, prompt, chat, ngl)
	return cpu, gpu, err
}

func prefillLogits(modelPath, prompt string, chat bool, ngl int) ([]float32, error) {
	engine, err := gogguf.Load(modelPath, gogguf.LoadOptions{
		NGL: ngl,
	})
	if err != nil {
		return nil, err
	}

	ctx, err := engine.NewContext()
	if err != nil {
		return nil, err
	}

	text := prompt
	if chat {
		text, err = gogguf.FormatChatUser(prompt, gogguf.ChatOptions{
			Metadata: engine.Metadata(),
		})
		if err != nil {
			return nil, err
		}
	}

	ids, err := ctx.EncodeForInference(text)
	if err != nil {
		return nil, err
	}

	engine.Model.ResetCache()
	logits, err := engine.Model.Forward(ids, 0)
	if err != nil {
		return nil, err
	}

	out := make([]float32, len(logits))
	copy(out, logits)
	return out, nil
}
