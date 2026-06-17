package main

import (
	"fmt"
	"os"

	"github.com/magomedcoder/gguf.go"
)

func main() {
	path := "./models/Qwen3-0.6B-Q8_0.gguf"
	if len(os.Args) > 1 {
		path = os.Args[1]
	}

	engine, err := gguf.Load(path, gguf.LoadOptions{})
	if err != nil {
		panic(err)
	}

	tok := engine.Tokenizer()
	text := "Hello"
	if len(os.Args) > 2 {
		text = os.Args[2]
	}

	ids, err := tok.Encode(text)
	if err != nil {
		panic(err)
	}

	fmt.Printf("Encode(%q) = %v\n", text, ids)
	for _, id := range ids {
		fmt.Printf("  %d: %q\n", id, tok.Decode([]int{id}))
	}

	_, err = engine.NewContext()
	if err != nil {
		panic(err)
	}

	engine.Model.ResetCache()
	logits, err := engine.Model.Forward(ids, 0)
	if err != nil {
		panic(err)
	}

	top := topN(logits, 5)
	fmt.Println("top logits after prefill:")

	for _, t := range top {
		fmt.Printf("  %d: %.4f %q\n", t.id, t.score, tok.Decode([]int{t.id}))
	}

	next := gguf.Greedy(logits)
	fmt.Printf("greedy next = %d %q\n", next, tok.Decode([]int{next}))
}

type scored struct {
	id    int
	score float32
}

func topN(logits []float32, n int) []scored {
	items := make([]scored, len(logits))
	for i, v := range logits {
		items[i] = scored{i, v}
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
