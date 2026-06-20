package server

import (
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/magomedcoder/gguf.go/pkg/chat"
	"github.com/magomedcoder/gguf.go/pkg/runtime"
	"github.com/magomedcoder/gguf.go/pkg/sampler"
)

type chatCompletionRequest struct {
	Model       string        `json:"model"`
	Messages    []chatMessage `json:"messages"`
	MaxTokens   int           `json:"max_tokens"`
	Temperature float32       `json:"temperature"`
	TopP        float32       `json:"top_p"`
	Stream      bool          `json:"stream"`
	Thinking    *bool         `json:"thinking"`
}

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatCompletionResponse struct {
	ID      string                 `json:"id"`
	Object  string                 `json:"object"`
	Created int64                  `json:"created"`
	Model   string                 `json:"model"`
	Choices []chatCompletionChoice `json:"choices"`
	Usage   chatCompletionUsage    `json:"usage"`
}

type chatCompletionChoice struct {
	Index        int         `json:"index"`
	Message      chatMessage `json:"message"`
	FinishReason string      `json:"finish_reason"`
}

type chatCompletionUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type chatStreamChunk struct {
	ID      string             `json:"id"`
	Object  string             `json:"object"`
	Created int64              `json:"created"`
	Model   string             `json:"model"`
	Choices []chatStreamChoice `json:"choices"`
}

type chatStreamChoice struct {
	Index int             `json:"index"`
	Delta chatStreamDelta `json:"delta"`
}

type chatStreamDelta struct {
	Role    string `json:"role,omitempty"`
	Content string `json:"content,omitempty"`
}

func (s *Server) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "метод не разрешен", http.StatusMethodNotAllowed)
		return
	}

	var req chatCompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	if len(req.Messages) == 0 {
		http.Error(w, "messages обязателен", http.StatusBadRequest)
		return
	}

	if req.MaxTokens <= 0 {
		req.MaxTokens = 128
	}

	if req.TopP <= 0 {
		req.TopP = 1
	}

	msgs := make([]chat.Message, len(req.Messages))
	for i, m := range req.Messages {
		msgs[i] = chat.Message{
			Role:    m.Role,
			Content: m.Content,
		}
	}

	prompt, err := chat.FormatMessages(msgs, chat.Options{
		Thinking: req.Thinking,
		Metadata: s.engine.Metadata(),
	})
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	modelName, _ := s.engine.Metadata().String("general.name")
	if modelName == "" {
		modelName = "gguf"
	}

	if req.Model != "" {
		modelName = req.Model
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	ctx, err := s.engine.NewContext()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	samp := sampler.New(sampler.Config{
		Temp: req.Temperature,
		TopP: req.TopP,
	})

	if req.Stream {
		s.serveChatStream(w, ctx, prompt, modelName, req.MaxTokens, samp)
		return
	}

	sess, err := ctx.StartGeneration(prompt)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	if err := sess.GenerateSteps(req.MaxTokens, samp, nil); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	text := sess.GeneratedText()
	writeJSON(w, chatCompletionResponse{
		ID:      fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano()),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   modelName,
		Choices: []chatCompletionChoice{{
			Index: 0,
			Message: chatMessage{
				Role:    "assistant",
				Content: text,
			},
			FinishReason: "stop",
		}},
		Usage: chatCompletionUsage{
			PromptTokens:     sess.PromptTokenCount(),
			CompletionTokens: sess.GeneratedCount(),
			TotalTokens:      sess.PromptTokenCount() + sess.GeneratedCount(),
		},
	})
}

func (s *Server) serveChatStream(w http.ResponseWriter, ctx *runtime.Context, prompt, model string, maxTokens int, samp sampler.Func) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming не поддерживается", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	id := fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano())
	created := time.Now().Unix()

	writeChunk := func(delta chatStreamDelta) {
		chunk := chatStreamChunk{
			ID:      id,
			Object:  "chat.completion.chunk",
			Created: created,
			Model:   model,
			Choices: []chatStreamChoice{{
				Index: 0,
				Delta: delta,
			}},
		}
		data, _ := json.Marshal(chunk)
		fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()
	}

	sess, err := ctx.StartGeneration(prompt)
	if err != nil {
		fmt.Fprintf(w, "data: {\"error\":%q}\n\n", err.Error())
		flusher.Flush()
		return
	}

	writeChunk(chatStreamDelta{
		Role: "assistant",
	})

	err = sess.GenerateSteps(maxTokens, samp, func(tokenID int) bool {
		writeChunk(chatStreamDelta{
			Content: sess.DecodeToken(tokenID),
		})
		return true
	})
	if err != nil {
		fmt.Fprintf(w, "data: {\"error\":%q}\n\n", err.Error())
		flusher.Flush()
		return
	}

	fmt.Fprintf(w, "data: [DONE]\n\n")
	flusher.Flush()
}
