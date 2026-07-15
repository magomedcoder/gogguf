package server

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/magomedcoder/gogguf/pkg/chat"
	"github.com/magomedcoder/gogguf/pkg/runtime"
	"github.com/magomedcoder/gogguf/pkg/sampler"
)

type chatCompletionRequest struct {
	Model          string        `json:"model"`
	Messages       []chatMessage `json:"messages"`
	MaxTokens      int           `json:"max_tokens"`
	Temperature    *float64      `json:"temperature,omitempty"`
	TopK           int           `json:"top_k"`
	TopP           *float64      `json:"top_p,omitempty"`
	MinP           float32       `json:"min_p"`
	RepeatPenalty  float32       `json:"repeat_penalty"`
	RepeatLastN    int           `json:"repeat_last_n"`
	Stop           []string      `json:"stop,omitempty"`
	Stream         bool          `json:"stream"`
	Thinking       *bool         `json:"thinking"`
	EnableThinking *bool         `json:"enable_thinking,omitempty"`
}

type chatMessage struct {
	Role       string          `json:"role"`
	Content    json.RawMessage `json:"content"`
	ToolCallID string          `json:"tool_call_id,omitempty"`
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
	Index        int              `json:"index"`
	Message      chatMessagePlain `json:"message"`
	FinishReason string           `json:"finish_reason"`
}

type chatMessagePlain struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatCompletionUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type chatStreamChunk struct {
	ID      string               `json:"id"`
	Object  string               `json:"object"`
	Created int64                `json:"created"`
	Model   string               `json:"model"`
	Choices []chatStreamChoice   `json:"choices"`
	Usage   *chatCompletionUsage `json:"usage,omitempty"`
}

type chatStreamChoice struct {
	Index int             `json:"index"`
	Delta chatStreamDelta `json:"delta"`
}

type chatStreamDelta struct {
	Role    string `json:"role,omitempty"`
	Content string `json:"content,omitempty"`
}

func parseChatMessageContent(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}

	if raw[0] == '"' {
		var s string
		if err := json.Unmarshal(raw, &s); err == nil {
			return s
		}

		return ""
	}

	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal(raw, &parts); err != nil {
		return ""
	}

	var b strings.Builder
	for _, p := range parts {
		if p.Type == "text" && p.Text != "" {
			b.WriteString(p.Text)
		}
	}

	return b.String()
}

func (req *chatCompletionRequest) thinkingEnabled() *bool {
	if req.EnableThinking != nil {
		return req.EnableThinking
	}

	return req.Thinking
}

func (req *chatCompletionRequest) samplerConfig() sampler.Config {
	cfg := sampler.Config{
		TopK: req.TopK,
		MinP: req.MinP,
	}

	if req.TopP != nil {
		cfg.TopP = float32(*req.TopP)
	} else {
		cfg.TopP = 1
	}

	if req.Temperature != nil {
		cfg.Temp = float32(*req.Temperature)
	}

	return cfg
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

	msgs := make([]chat.Message, 0, len(req.Messages))
	for _, m := range req.Messages {
		content := parseChatMessageContent(m.Content)
		if m.Role == "tool" && content == "" && m.ToolCallID != "" {
			content = m.ToolCallID
		}

		msgs = append(msgs, chat.Message{
			Role:    m.Role,
			Content: content,
		})
	}

	prompt, err := chat.FormatMessages(msgs, chat.Options{
		Thinking: req.thinkingEnabled(),
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

	conv, err := s.conversation()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	genParams := runtime.GenerateParams{
		MaxTokens:     req.MaxTokens,
		Sampler:       sampler.New(req.samplerConfig()),
		RepeatPenalty: req.RepeatPenalty,
		RepeatLastN:   req.RepeatLastN,
		Stop:          req.Stop,
	}

	if req.Stream {
		s.serveChatStream(w, conv, prompt, modelName, genParams)
		return
	}

	snap := conv.TokenCount()
	sess, err := conv.StartGeneration(prompt)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	if err := sess.GenerateSteps(genParams); err != nil {
		conv.Rollback(snap)
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	conv.Commit(sess)

	text := trimStopSuffix(sess.GeneratedText(), req.Stop)
	writeJSON(w, chatCompletionResponse{
		ID:      fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano()),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   modelName,
		Choices: []chatCompletionChoice{{
			Index: 0,
			Message: chatMessagePlain{
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

func trimStopSuffix(text string, stops []string) string {
	for _, stop := range stops {
		stop = strings.TrimSpace(stop)
		if stop != "" && strings.HasSuffix(text, stop) {
			return strings.TrimSuffix(text, stop)
		}
	}

	return text
}

func (s *Server) serveChatStream(w http.ResponseWriter, conv *runtime.Conversation, prompt, model string, params runtime.GenerateParams) {
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

	writeChunk := func(delta chatStreamDelta, usage *chatCompletionUsage) {
		chunk := chatStreamChunk{
			ID:      id,
			Object:  "chat.completion.chunk",
			Created: created,
			Model:   model,
			Choices: []chatStreamChoice{{
				Index: 0,
				Delta: delta,
			}},
			Usage: usage,
		}
		data, _ := json.Marshal(chunk)
		fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()
	}

	snap := conv.TokenCount()
	sess, err := conv.StartGeneration(prompt)
	if err != nil {
		fmt.Fprintf(w, "data: {\"error\":%q}\n\n", err.Error())
		flusher.Flush()
		return
	}

	writeChunk(chatStreamDelta{
		Role: "assistant"},
		nil,
	)

	params.OnToken = func(tokenID int) bool {
		writeChunk(chatStreamDelta{
			Content: sess.DecodeToken(tokenID)},
			nil,
		)

		return true
	}

	err = sess.GenerateSteps(params)
	if err != nil {
		conv.Rollback(snap)
		fmt.Fprintf(w, "data: {\"error\":%q}\n\n", err.Error())
		flusher.Flush()
		return
	}

	conv.Commit(sess)

	writeChunk(chatStreamDelta{}, &chatCompletionUsage{
		PromptTokens:     sess.PromptTokenCount(),
		CompletionTokens: sess.GeneratedCount(),
		TotalTokens:      sess.PromptTokenCount() + sess.GeneratedCount(),
	})

	fmt.Fprintf(w, "data: [DONE]\n\n")
	flusher.Flush()
}
