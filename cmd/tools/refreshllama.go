package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"github.com/magomedcoder/gogguf"
	"github.com/magomedcoder/gogguf/pkg/debug"
	"github.com/magomedcoder/gogguf/pkg/model"
)

// runRefreshLlama rewrites test/fixtures/llama32_golden.json and llama32_layer_logits.json from the current CPU forward (use after numeric path changes).
func runRefreshLlama(args []string) error {
	fs := flag.NewFlagSet("refreshllama", flag.ContinueOnError)
	modelPath := fs.String("m", "models/Llama-3.2-1B-Instruct-Q8_0.gguf", "path to GGUF")
	outDir := fs.String("o", "test/fixtures", "fixture directory")
	topN := fs.Int("top", 5, "top logits to record")
	if err := fs.Parse(args); err != nil {
		return err
	}

	engine, err := gogguf.LoadMapped(*modelPath, gogguf.LoadOptions{})
	if err != nil {
		return err
	}
	defer engine.Close()

	ctx, err := engine.NewContext()
	if err != nil {
		return err
	}

	setter, ok := engine.Model.(model.DebugHookSetter)
	if !ok {
		return fmt.Errorf("llama model missing debug hooks")
	}

	type gLogit struct {
		ID    int     `json:"id"`
		Logit float32 `json:"logit"`
	}
	type gCase struct {
		Name         string   `json:"name"`
		Input        string   `json:"input,omitempty"`
		ChatUser     string   `json:"chat_user,omitempty"`
		Encode       []int    `json:"encode,omitempty"`
		GreedyNext   int      `json:"greedy_next,omitempty"`
		TopLogits    []gLogit `json:"top_logits,omitempty"`
		MaxTokens    int      `json:"max_tokens,omitempty"`
		GreedyTokens []int    `json:"greedy_tokens,omitempty"`
	}

	golden := struct {
		Model string  `json:"model"`
		Cases []gCase `json:"cases"`
	}{Model: "Llama-3.2-1B-Instruct-Q8_0"}

	ids, err := engine.Tokenizer().Encode("Hello")
	if err != nil {
		return err
	}

	golden.Cases = append(golden.Cases, gCase{
		Name:   "encode_hello",
		Input:  "Hello",
		Encode: ids,
	})

	forwardTop := func(text string) (int, []gLogit, error) {
		ids, err := ctx.EncodeForInference(text)
		if err != nil {
			return 0, nil, err
		}

		engine.Model.ResetCache()
		logits, err := engine.Model.Forward(ids, 0)
		if err != nil {
			return 0, nil, err
		}

		top := debug.TopLogits(logits, *topN)
		out := make([]gLogit, len(top))
		for i, l := range top {
			out[i] = gLogit{
				ID:    l.ID,
				Logit: l.Logit,
			}
		}

		return gogguf.Greedy(logits), out, nil
	}

	next, top, err := forwardTop("Hello")
	if err != nil {
		return err
	}
	golden.Cases = append(golden.Cases, gCase{
		Name: "raw_hello_greedy", Input: "Hello", GreedyNext: next, TopLogits: top,
	})

	chatPrompt, err := gogguf.FormatChatUser("Hello", gogguf.ChatOptions{Metadata: engine.Metadata()})
	if err != nil {
		return err
	}
	next, top, err = forwardTop(chatPrompt)
	if err != nil {
		return err
	}
	golden.Cases = append(golden.Cases, gCase{
		Name: "chat_hello_greedy", ChatUser: "Hello", GreedyNext: next, TopLogits: top,
	})

	// Keep multi-step greedy tokens by running generation
	sess, err := ctx.StartGeneration(chatPrompt)
	if err != nil {
		return err
	}

	const nTok = 7
	if err := sess.GenerateSteps(gogguf.GenerateParams{MaxTokens: nTok, Sampler: gogguf.Greedy}); err != nil {
		return err
	}

	golden.Cases = append(golden.Cases, gCase{
		Name: "chat_greedy_8", ChatUser: "Hello", MaxTokens: nTok, GreedyTokens: sess.GeneratedTokens(),
	})

	type layerCase struct {
		Name           string          `json:"name"`
		Input          string          `json:"input,omitempty"`
		ChatUser       string          `json:"chat_user,omitempty"`
		LayerGreedy    []int           `json:"layer_greedy"`
		LayerTopLogits [][]debug.Logit `json:"layer_top_logits,omitempty"`
	}
	layersOut := struct {
		Model string      `json:"model"`
		Cases []layerCase `json:"cases"`
	}{
		Model: filepath.Base(*modelPath),
	}

	collectLayers := func(name, input, chatUser, text string) error {
		ids, err := ctx.EncodeForInference(text)
		if err != nil {
			return err
		}
		var greedy []int
		var tops [][]debug.Logit
		setter.SetDebugHooks(&debug.Hooks{
			OnLayerLogits: func(_ int, logits []float32) {
				greedy = append(greedy, gogguf.Greedy(logits))
				tops = append(tops, debug.TopLogits(logits, *topN))
			},
		})
		engine.Model.ResetCache()
		if _, err := engine.Model.Forward(ids, 0); err != nil {
			return err
		}

		c := layerCase{Name: name, LayerGreedy: greedy, LayerTopLogits: tops}
		if chatUser != "" {
			c.ChatUser = chatUser
		} else {
			c.Input = input
		}

		layersOut.Cases = append(layersOut.Cases, c)
		return nil
	}

	if err := collectLayers("raw_hello_layer_greedy", "Hello", "", "Hello"); err != nil {
		return err
	}

	if err := collectLayers("chat_hello_layer_greedy", "", "Hello", chatPrompt); err != nil {
		return err
	}

	writeJSON := func(name string, v any) error {
		path := filepath.Join(*outDir, name)
		f, err := os.Create(path)
		if err != nil {
			return err
		}
		defer f.Close()
		enc := json.NewEncoder(f)
		enc.SetIndent("", "  ")
		if err := enc.Encode(v); err != nil {
			return err
		}

		fmt.Println("wrote", path)
		return nil
	}

	if err := writeJSON("llama32_golden.json", golden); err != nil {
		return err
	}

	return writeJSON("llama32_layer_logits.json", layersOut)
}
