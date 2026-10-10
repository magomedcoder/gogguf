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

// OpenAI Chat Completions request (unsupported extras are accepted as stubs / ignored).
type chatCompletionRequest struct {
	Model               string          `json:"model"`
	Messages            []chatMessage   `json:"messages"`
	MaxTokens           int             `json:"max_tokens"`
	MaxCompletionTokens int             `json:"max_completion_tokens"`
	Temperature         *float64        `json:"temperature,omitempty"`
	TopP                *float64        `json:"top_p,omitempty"`
	N                   int             `json:"n"`
	Stop                json.RawMessage `json:"stop,omitempty"`
	Stream              bool            `json:"stream"`
	StreamOptions       *streamOptions  `json:"stream_options,omitempty"`
	PresencePenalty     *float64        `json:"presence_penalty,omitempty"` // stub
	FrequencyPenalty    *float64        `json:"frequency_penalty,omitempty"`
	Seed                *int64          `json:"seed,omitempty"`       // stub
	User                string          `json:"user,omitempty"`       // stub
	LogitBias           json.RawMessage `json:"logit_bias,omitempty"` // stub
	Logprobs            *bool           `json:"logprobs,omitempty"`   // stub
	TopLogprobs         *int            `json:"top_logprobs,omitempty"`
	ResponseFormat      json.RawMessage `json:"response_format,omitempty"` // stub
	Tools               []chat.Tool     `json:"tools,omitempty"`
	ToolChoice          json.RawMessage `json:"tool_choice,omitempty"`
	ParallelToolCalls   bool            `json:"parallel_tool_calls,omitempty"`
}

type streamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

type chatMessage struct {
	Role       string          `json:"role"`
	Content    json.RawMessage `json:"content"`
	Name       string          `json:"name,omitempty"`
	ToolCallID string          `json:"tool_call_id,omitempty"`
	ToolCalls  []chat.ToolCall `json:"tool_calls,omitempty"`
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
	Logprobs     *struct{}        `json:"logprobs"` // OpenAI: null when unused
}

type chatMessagePlain struct {
	Role      string          `json:"role"`
	Content   string          `json:"content"`
	ToolCalls []chat.ToolCall `json:"tool_calls,omitempty"`
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
	Index        int             `json:"index"`
	Delta        chatStreamDelta `json:"delta"`
	FinishReason *string         `json:"finish_reason"`
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

func parseStop(raw json.RawMessage) []string {
	if len(raw) == 0 || string(raw) == "null" {
		return nil
	}

	if raw[0] == '"' {
		var s string
		if err := json.Unmarshal(raw, &s); err == nil && s != "" {
			return []string{s}
		}

		return nil
	}

	var arr []string
	if err := json.Unmarshal(raw, &arr); err == nil {
		return arr
	}

	return nil
}

func parseToolChoice(raw json.RawMessage) any {
	if len(raw) == 0 {
		return nil
	}

	var asString string
	if err := json.Unmarshal(raw, &asString); err == nil {
		return asString
	}

	var asObj any
	if err := json.Unmarshal(raw, &asObj); err == nil {
		return asObj
	}

	return nil
}

func (req *chatCompletionRequest) maxNewTokens() int {
	if req.MaxCompletionTokens > 0 {
		return req.MaxCompletionTokens
	}

	if req.MaxTokens > 0 {
		return req.MaxTokens
	}

	return 128
}

func (req *chatCompletionRequest) samplerConfig() sampler.Config {
	cfg := sampler.Config{TopP: 1}
	if req.TopP != nil {
		cfg.TopP = float32(*req.TopP)
	}

	if req.Temperature != nil {
		cfg.Temp = float32(*req.Temperature)
	}

	return cfg
}

// frequencyPenaltyToRepeat maps OpenAI frequency_penalty (-2..2) to repeat_penalty (>=1).
func frequencyPenaltyToRepeat(p *float64) float32 {
	if p == nil || *p <= 0 {
		return 1
	}

	v := 1 + float32(*p)
	if v > 2 {
		return 2
	}

	return v
}

func (s *Server) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeAPIError(w, http.StatusMethodNotAllowed, "method not allowed", "invalid_request_error")
		return
	}

	var req chatCompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeAPIError(w, http.StatusBadRequest, err.Error(), "invalid_request_error")
		return
	}

	if len(req.Messages) == 0 {
		writeAPIError(w, http.StatusBadRequest, "messages is required", "invalid_request_error")
		return
	}

	if req.N > 1 {
		writeAPIError(w, http.StatusBadRequest, "n > 1 is not supported (stub: only n=1)", "invalid_request_error")
		return
	}

	// Stubs: presence_penalty, seed, user, logit_bias, logprobs, response_format are accepted and ignored.

	stops := parseStop(req.Stop)
	msgs := make([]chat.Message, 0, len(req.Messages))
	for _, m := range req.Messages {
		content := parseChatMessageContent(m.Content)
		msgs = append(msgs, chat.Message{
			Role:       m.Role,
			Content:    content,
			Name:       m.Name,
			ToolCallID: m.ToolCallID,
			ToolCalls:  m.ToolCalls,
		})
	}

	prompt, err := chat.FormatMessages(msgs, chat.Options{
		Metadata:          s.engine.Metadata(),
		Tools:             req.Tools,
		ToolChoice:        parseToolChoice(req.ToolChoice),
		ParallelToolCalls: req.ParallelToolCalls,
	})
	if err != nil {
		writeAPIError(w, http.StatusInternalServerError, err.Error(), "server_error")
		return
	}

	modelName := s.modelID()
	if req.Model != "" {
		modelName = req.Model
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	conv, err := s.conversation()
	if err != nil {
		writeAPIError(w, http.StatusInternalServerError, err.Error(), "server_error")
		return
	}

	genParams := runtime.GenerateParams{
		MaxTokens:     req.maxNewTokens(),
		Sampler:       sampler.New(req.samplerConfig()),
		RepeatPenalty: frequencyPenaltyToRepeat(req.FrequencyPenalty),
		RepeatLastN:   64,
		Stop:          stops,
	}

	if req.Stream {
		includeUsage := req.StreamOptions != nil && req.StreamOptions.IncludeUsage
		s.serveChatStream(w, conv, prompt, modelName, genParams, includeUsage)
		return
	}

	snap := conv.TokenCount()
	sess, err := conv.StartGeneration(prompt)
	if err != nil {
		writeAPIError(w, http.StatusInternalServerError, err.Error(), "server_error")
		return
	}

	if err := sess.GenerateSteps(genParams); err != nil {
		conv.Rollback(snap)
		writeAPIError(w, http.StatusInternalServerError, err.Error(), "server_error")
		return
	}

	conv.Commit(sess)

	text := trimStopSuffix(sess.GeneratedText(), stops)
	parsed := chat.ParseAssistantOutput(text)
	writeJSON(w, chatCompletionResponse{
		ID:      fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano()),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   modelName,
		Choices: []chatCompletionChoice{{
			Index: 0,
			Message: chatMessagePlain{
				Role:      "assistant",
				Content:   parsed.Content,
				ToolCalls: parsed.ToolCalls,
			},
			FinishReason: parsed.FinishReason(),
			Logprobs:     nil,
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

func (s *Server) serveChatStream(w http.ResponseWriter, conv *runtime.Conversation, prompt, model string, params runtime.GenerateParams, includeUsage bool) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeAPIError(w, http.StatusInternalServerError, "streaming is not supported", "server_error")
		return
	}

	snap := conv.TokenCount()
	sess, err := conv.StartGeneration(prompt)
	if err != nil {
		writeAPIError(w, http.StatusInternalServerError, err.Error(), "server_error")
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	id := fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano())
	created := time.Now().Unix()

	writeChunk := func(delta chatStreamDelta, finish *string, usage *chatCompletionUsage) {
		chunk := chatStreamChunk{
			ID:      id,
			Object:  "chat.completion.chunk",
			Created: created,
			Model:   model,
			Choices: []chatStreamChoice{{
				Index:        0,
				Delta:        delta,
				FinishReason: finish,
			}},
			Usage: usage,
		}
		data, _ := json.Marshal(chunk)
		fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()
	}

	writeChunk(chatStreamDelta{Role: "assistant"}, nil, nil)

	params.OnToken = func(tokenID int) bool {
		writeChunk(chatStreamDelta{Content: sess.DecodeToken(tokenID)}, nil, nil)
		return true
	}

	err = sess.GenerateSteps(params)
	if err != nil {
		conv.Rollback(snap)
		fmt.Fprintf(w, "data: {\"error\":{\"message\":%q,\"type\":\"server_error\"}}\n\n", err.Error())
		flusher.Flush()
		return
	}

	conv.Commit(sess)

	finish := "stop"
	var usage *chatCompletionUsage
	if includeUsage {
		usage = &chatCompletionUsage{
			PromptTokens:     sess.PromptTokenCount(),
			CompletionTokens: sess.GeneratedCount(),
			TotalTokens:      sess.PromptTokenCount() + sess.GeneratedCount(),
		}
	}

	writeChunk(chatStreamDelta{}, &finish, usage)

	fmt.Fprintf(w, "data: [DONE]\n\n")
	flusher.Flush()
}
