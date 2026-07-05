package chat

import "github.com/magomedcoder/gogguf/pkg/format"

const (
	qwenThinkingOpenID  = 151667
	qwenThinkingCloseID = 151668
)

// ThinkingTags возвращает строки thinking-токенов Qwen3 (ids 151667/151668)
func ThinkingTags(meta format.Metadata) (open, close string) {
	open = defaultThinkingOpen
	close = defaultThinkingClose
	if meta == nil {
		return open, close
	}

	tokens, err := meta.StringArray("tokenizer.ggml.tokens")
	if err != nil || len(tokens) <= qwenThinkingCloseID {
		return open, close
	}

	if s := tokens[qwenThinkingOpenID]; s != "" {
		open = s
	}
	if s := tokens[qwenThinkingCloseID]; s != "" {
		close = s
	}

	return open, close
}

// EmptyThinkingBlock - hard switch Qwen3: пустой thinking-блок отключает размышление
func EmptyThinkingBlock(meta format.Metadata) string {
	open, close := ThinkingTags(meta)
	return open + "\n\n" + close + "\n\n"
}
