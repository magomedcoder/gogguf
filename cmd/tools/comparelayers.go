package main

import (
	"flag"
	"fmt"

	"github.com/magomedcoder/gogguf/pkg/debug"
)

// runCompareLayers сравнивает два dump или CPU vs GPU hidden по слоям
func runCompareLayers(args []string) error {
	fs := flag.NewFlagSet("comparelayers", flag.ContinueOnError)
	aPath := fs.String("a", "", "dump A")
	bPath := fs.String("b", "", "dump B")
	modelPath := fs.String("m", "", "модель: CPU vs GPU")
	prompt := fs.String("p", "Hello", "промпт для -m")
	chatMode := fs.Bool("chat", false, "chat template")
	ngl := fs.Int("ngl", -1, "слоёв GPU (-1 = все)")
	tol := fs.Float64("tol", 1e-4, "допуск |diff|")

	if err := fs.Parse(args); err != nil {
		return err
	}

	var a, b debug.LayersDump
	var err error

	switch {
	case *modelPath != "":
		a, _, err = collectLayersDump(*modelPath, *prompt, *chatMode, 0)
		if err != nil {
			return err
		}

		gpuNGL := *ngl
		if gpuNGL < 0 {
			gpuNGL = 999
		}

		b, _, err = collectLayersDump(*modelPath, *prompt, *chatMode, gpuNGL)
	case *aPath != "" && *bPath != "":
		_, a, err = debug.LoadLayersDump(*aPath)
		if err != nil {
			return fmt.Errorf("A: %w", err)
		}

		_, b, err = debug.LoadLayersDump(*bPath)
	default:
		return fmt.Errorf("укажите -a/-b или -m")
	}

	if err != nil {
		return err
	}

	embed, layers, err := debug.DiffLayers(a, b, *tol)
	if err != nil {
		return err
	}

	fmt.Printf("embed: max_abs=%.6g@%d mean=%.6g over=%d\n", embed.MaxAbs, embed.MaxAbsIdx, embed.MeanAbs, embed.OverTol)
	var worst float64
	var worstI, totalOver int
	for i, st := range layers {
		if st.MaxAbs > worst {
			worst = st.MaxAbs
			worstI = i
		}

		totalOver += st.OverTol
		if st.OverTol > 0 {
			fmt.Printf("layer[%d]: max_abs=%.6g@%d over=%d\n", i, st.MaxAbs, st.MaxAbsIdx, st.OverTol)
		}
	}
	fmt.Printf("layers: n=%d worst_max_abs=%.6g@layer%d total_over_tol=%d (tol=%g)\n", len(layers), worst, worstI, totalOver, *tol)

	if embed.OverTol > 0 || totalOver > 0 {
		return fmt.Errorf("FAIL: hidden states вне допуска")
	}

	fmt.Println("OK: layer hidden в допуске")

	return nil
}
