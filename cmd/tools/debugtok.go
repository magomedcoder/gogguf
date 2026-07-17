package main

import (
	"fmt"

	"github.com/magomedcoder/gogguf"
)

// runDebugTok проверяет encode промпта и logits после prefill
func runDebugTok(args []string) error {
	path := "./models/Qwen3-0.6B-Q8_0.gguf"
	if len(args) > 0 {
		path = args[0]
	}

	engine, err := gogguf.Load(path, gogguf.LoadOptions{})
	if err != nil {
		return err
	}

	tok := engine.Tokenizer()
	text := "Hello"
	if len(args) > 1 {
		text = args[1]
	}

	ids, err := tok.Encode(text)
	if err != nil {
		return err
	}

	fmt.Printf("Encode(%q) = %v\n", text, ids)
	for _, id := range ids {
		fmt.Printf("  %d: %q\n", id, tok.Decode([]int{id}))
	}

	if _, err = engine.NewContext(); err != nil {
		return err
	}

	engine.Model.ResetCache()
	logits, err := engine.Model.Forward(ids, 0)
	if err != nil {
		return err
	}

	top := debugTokTopN(logits, 5)
	fmt.Println("top logits after prefill:")

	for _, t := range top {
		fmt.Printf("  %d: %.4f %q\n", t.id, t.score, tok.Decode([]int{t.id}))
	}

	next := gogguf.Greedy(logits)
	fmt.Printf("greedy next = %d %q\n", next, tok.Decode([]int{next}))
	return nil
}

type scoredToken struct {
	id    int
	score float32
}

func debugTokTopN(logits []float32, n int) []scoredToken {
	items := make([]scoredToken, len(logits))
	for i, v := range logits {
		items[i] = scoredToken{i, v}
	}

	for i := range items {
		for j := i + 1; j < len(items); j++ {
			if items[j].score > items[i].score {
				items[i], items[j] = items[j], items[i]
			}
		}
	}

	if n > len(items) {
		n = len(items)
	}

	return items[:n]
}
