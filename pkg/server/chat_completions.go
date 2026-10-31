package server

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/magomedcoder/gogguf/pkg/chat"
	"github.com/magomedcoder/gogguf/pkg/runtime"
)

// OpenAI Chat Completions request (unsupported extras are accepted as stubs / ignored).
type chatCompletionRequest struct {
	Model               string          `json:"model"`
	Messages            []chatMessage   `json:"messages"`
	MaxTokens           int             `json:"max_tokens,omitzero"`
	MaxCompletionTokens int             `json:"max_completion_tokens,omitzero"`
	Temperature         *float64        `json:"temperature,omitempty"`
	TopP                *float64        `json:"top_p,omitempty"`
	N                   int             `json:"n,omitzero"`
	Stop                json.RawMessage `json:"stop,omitempty"`
	Stream              bool            `json:"stream,omitzero"`
	StreamOptions       *streamOptions  `json:"stream_options,omitempty"`
	PresencePenalty     *float64        `json:"presence_penalty,omitempty"`
	FrequencyPenalty    *float64        `json:"frequency_penalty,omitempty"`
	Seed                *int64          `json:"seed,omitempty"`
	User                string          `json:"user,omitempty"` // session key when X-Session-Id is absent
	LogitBias           json.RawMessage `json:"logit_bias,omitempty"`
	Logprobs            *bool           `json:"logprobs,omitempty"`
	TopLogprobs         *int            `json:"top_logprobs,omitempty"`
	ResponseFormat      json.RawMessage `json:"response_format,omitempty"`
	Tools               []chat.Tool     `json:"tools,omitempty"`
	ToolChoice          json.RawMessage `json:"tool_choice,omitempty"`
	ParallelToolCalls   *bool           `json:"parallel_tool_calls,omitempty"`
}

type streamOptions struct {
	IncludeUsage bool `json:"include_usage,omitzero"`
}

type chatMessage struct {
	Role       string          `json:"role"`
	Content    json.RawMessage `json:"content"`
	Name       string          `json:"name,omitempty"`
	ToolCallID string          `json:"tool_call_id,omitempty"`
	ToolCalls  []chat.ToolCall `json:"tool_calls,omitempty"`
}

type chatCompletionResponse struct {
	ID                string                 `json:"id"`
	Object            string                 `json:"object"`
	Created           int64                  `json:"created"`
	Model             string                 `json:"model"`
	SystemFingerprint string                 `json:"system_fingerprint,omitempty"`
	Choices           []chatCompletionChoice `json:"choices"`
	Usage             chatCompletionUsage    `json:"usage"`
}

type chatCompletionChoice struct {
	Index        int                 `json:"index"`
	Message      chatMessagePlain    `json:"message"`
	FinishReason string              `json:"finish_reason"`
	Logprobs     *chatChoiceLogprobs `json:"logprobs"` // OpenAI: null when unused
}

type chatChoiceLogprobs struct {
	Content []chatTokenLogprob `json:"content"`
}

type chatTokenLogprob struct {
	Token       string           `json:"token"`
	Logprob     float64          `json:"logprob"`
	Bytes       []byte           `json:"bytes"`
	TopLogprobs []chatTopLogprob `json:"top_logprobs"`
}

type chatTopLogprob struct {
	Token   string  `json:"token"`
	Logprob float64 `json:"logprob"`
	Bytes   []byte  `json:"bytes"`
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
	ID                string               `json:"id"`
	Object            string               `json:"object"`
	Created           int64                `json:"created"`
	Model             string               `json:"model"`
	SystemFingerprint string               `json:"system_fingerprint,omitempty"`
	Choices           []chatStreamChoice   `json:"choices"`
	Usage             *chatCompletionUsage `json:"usage,omitempty"`
}

type chatStreamChoice struct {
	Index        int             `json:"index"`
	Delta        chatStreamDelta `json:"delta"`
	FinishReason *string         `json:"finish_reason"`
}

type chatStreamDelta struct {
	Role      string               `json:"role,omitempty"`
	Content   string               `json:"content,omitempty"`
	ToolCalls []chatStreamToolCall `json:"tool_calls,omitempty"`
}

type chatStreamToolCall struct {
	Index    int                    `json:"index"`
	ID       string                 `json:"id,omitempty"`
	Type     string                 `json:"type,omitempty"`
	Function chatStreamToolFunction `json:"function"`
}

type chatStreamToolFunction struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

func parseChatMessageContent(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}

	if raw[0] == '"' {
		var s string
		if err := unmarshalWire(raw, &s); err == nil {
			return s
		}

		return ""
	}

	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if err := unmarshalWire(raw, &parts); err != nil {
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
		if err := unmarshalWire(raw, &s); err == nil && s != "" {
			return []string{s}
		}

		return nil
	}

	var arr []string
	if err := unmarshalWire(raw, &arr); err == nil {
		return arr
	}

	return nil
}

func parseToolChoice(raw json.RawMessage) any {
	if len(raw) == 0 {
		return nil
	}

	var asString string
	if err := unmarshalWire(raw, &asString); err == nil {
		return asString
	}

	var asObj any
	if err := unmarshalWire(raw, &asObj); err == nil {
		return asObj
	}

	return nil
}

// openaiFinishReason maps a parsed reply and the generation stop onto OpenAI finish_reason.
func openaiFinishReason(hasToolCalls bool, genStop string) string {
	if hasToolCalls {
		return "tool_calls"
	}

	if genStop == "length" {
		return "length"
	}

	return "stop"
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
	if err := decodeWire(r, &req); err != nil {
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

	wantLogprobs := req.Logprobs != nil && *req.Logprobs
	topLogprobs := 0
	if req.TopLogprobs != nil {
		topLogprobs = *req.TopLogprobs
	}
	if wantLogprobs {
		if topLogprobs < 0 || topLogprobs > 20 {
			writeAPIError(w, http.StatusBadRequest, "top_logprobs must be between 0 and 20", "invalid_request_error")
			return
		}
	} else if req.TopLogprobs != nil && topLogprobs != 0 {
		writeAPIError(w, http.StatusBadRequest, "top_logprobs requires logprobs=true", "invalid_request_error")
		return
	}

	respFmt, err := parseResponseFormat(req.ResponseFormat)
	if err != nil {
		writeAPIError(w, http.StatusBadRequest, err.Error(), "invalid_request_error")
		return
	}

	bias, err := parseLogitBias(req.LogitBias)
	if err != nil {
		writeAPIError(w, http.StatusBadRequest, err.Error(), "invalid_request_error")
		return
	}

	stops := parseStop(req.Stop)
	msgs := make([]chat.Message, 0, len(req.Messages))
	for _, m := range req.Messages {
		if contentHasMultimodal(m.Content) {
			writeAPIError(w, http.StatusBadRequest, "multimodal message content is not supported (text only)", "invalid_request_error")
			return
		}

		content := parseChatMessageContent(m.Content)
		msgs = append(msgs, chat.Message{
			Role:       m.Role,
			Content:    content,
			Name:       m.Name,
			ToolCallID: m.ToolCallID,
			ToolCalls:  m.ToolCalls,
		})
	}

	parallelToolCalls := false
	if req.ParallelToolCalls != nil {
		parallelToolCalls = *req.ParallelToolCalls
	}
	prompt, err := chat.FormatMessages(msgs, chat.Options{
		Metadata:          s.engine.Metadata(),
		Tools:             req.Tools,
		ToolChoice:        parseToolChoice(req.ToolChoice),
		ParallelToolCalls: parallelToolCalls,
	})
	if err != nil {
		writeAPIError(w, http.StatusInternalServerError, err.Error(), "server_error")
		return
	}

	if respFmt == "json_object" {
		prompt += "\n\nRespond with a single valid JSON object or array only, no markdown fences."
	}

	modelName := s.resolveModelID()
	if req.Model != "" {
		modelName = req.Model
	}

	conv, err := s.conversationForSession(sessionIDFromRequest(r, req.User))
	if err != nil {
		writeAPIError(w, http.StatusInternalServerError, err.Error(), "server_error")
		return
	}

	genParams := baseGenerateParams(req.MaxTokens, req.MaxCompletionTokens, req.TopP, req.Temperature, req.Seed, req.FrequencyPenalty, req.PresencePenalty, bias, stops)
	genParams.Logprobs = wantLogprobs
	genParams.TopLogprobs = topLogprobs

	s.inferMu.Lock()
	defer s.inferMu.Unlock()

	if err := s.ensureKV(conv); err != nil {
		writeAPIError(w, http.StatusInternalServerError, err.Error(), "server_error")
		return
	}

	fp := systemFingerprint(modelName)
	if req.Stream {
		genParams.Logprobs = false // stream logprobs not wired yet
		includeUsage := req.StreamOptions != nil && req.StreamOptions.IncludeUsage
		s.serveChatStream(w, conv, prompt, modelName, genParams, includeUsage, len(req.Tools) > 0 || respFmt == "json_object", stops, respFmt, fp)
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
	if respFmt == "json_object" && !parsed.HasToolCalls() {
		if err := ensureJSONObjectOutput(parsed.Content); err != nil {
			writeAPIError(w, http.StatusBadRequest, err.Error(), "invalid_request_error")
			return
		}
	}

	writeJSON(w, chatCompletionResponse{
		ID:                fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano()),
		Object:            "chat.completion",
		Created:           time.Now().Unix(),
		Model:             modelName,
		SystemFingerprint: fp,
		Choices: []chatCompletionChoice{{
			Index: 0,
			Message: chatMessagePlain{
				Role:      "assistant",
				Content:   parsed.Content,
				ToolCalls: parsed.ToolCalls,
			},
			FinishReason: openaiFinishReason(parsed.HasToolCalls(), sess.StopReason()),
			Logprobs:     buildChatLogprobs(sess),
		}},
		Usage: chatCompletionUsage{
			PromptTokens:     sess.PromptTokenCount(),
			CompletionTokens: sess.GeneratedCount(),
			TotalTokens:      sess.PromptTokenCount() + sess.GeneratedCount(),
		},
	})
}

func buildChatLogprobs(sess *runtime.GenerationSession) *chatChoiceLogprobs {
	entries := sess.Logprobs()
	if len(entries) == 0 {
		return nil
	}

	content := make([]chatTokenLogprob, 0, len(entries))
	for _, e := range entries {
		tok := sess.DecodeToken(e.TokenID)
		item := chatTokenLogprob{
			Token:   tok,
			Logprob: e.Logprob,
			Bytes:   []byte(tok),
		}

		if len(e.Top) > 0 {
			item.TopLogprobs = make([]chatTopLogprob, 0, len(e.Top))
			for _, t := range e.Top {
				tt := sess.DecodeToken(t.ID)
				item.TopLogprobs = append(item.TopLogprobs, chatTopLogprob{
					Token:   tt,
					Logprob: t.Logprob,
					Bytes:   []byte(tt),
				})
			}
		} else {
			item.TopLogprobs = []chatTopLogprob{}
		}

		content = append(content, item)
	}

	return &chatChoiceLogprobs{
		Content: content,
	}
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

func (s *Server) serveChatStream(w http.ResponseWriter, conv *runtime.Conversation, prompt, model string, params runtime.GenerateParams, includeUsage, bufferOutput bool, stops []string, respFmt, fingerprint string) {
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
			ID:                id,
			Object:            "chat.completion.chunk",
			Created:           created,
			Model:             model,
			SystemFingerprint: fingerprint,
			Choices: []chatStreamChoice{{
				Index:        0,
				Delta:        delta,
				FinishReason: finish,
			}},
			Usage: usage,
		}
		_ = writeSSEJSON(w, chunk)
		flusher.Flush()
	}

	writeChunk(chatStreamDelta{Role: "assistant"}, nil, nil)

	params.OnToken = func(tokenID int) bool {
		if bufferOutput {
			return true
		}

		writeChunk(chatStreamDelta{Content: sess.DecodeToken(tokenID)}, nil, nil)
		return true
	}

	err = sess.GenerateSteps(params)
	if err != nil {
		conv.Rollback(snap)
		_ = writeSSEJSON(w, apiErrorResponse{Error: apiErrorBody{Message: err.Error(), Type: "server_error"}})
		flusher.Flush()
		return
	}

	conv.Commit(sess)

	text := trimStopSuffix(sess.GeneratedText(), stops)
	parsed := chat.ParseAssistantOutput(text)
	if respFmt == "json_object" && !parsed.HasToolCalls() {
		if err := ensureJSONObjectOutput(parsed.Content); err != nil {
			_ = writeSSEJSON(w, apiErrorResponse{Error: apiErrorBody{Message: err.Error(), Type: "invalid_request_error"}})
			flusher.Flush()
			return
		}
	}

	if parsed.HasToolCalls() {
		if parsed.Content != "" {
			writeChunk(chatStreamDelta{Content: parsed.Content}, nil, nil)
		}
		for i, tc := range parsed.ToolCalls {
			writeChunk(chatStreamDelta{ToolCalls: []chatStreamToolCall{{
				Index: i,
				ID:    tc.ID,
				Type:  tc.Type,
				Function: chatStreamToolFunction{
					Name:      tc.Function.Name,
					Arguments: tc.Function.Arguments,
				},
			}}}, nil, nil)
		}
	} else if bufferOutput && text != "" {
		writeChunk(chatStreamDelta{Content: parsed.Content}, nil, nil)
	}

	finish := openaiFinishReason(parsed.HasToolCalls(), sess.StopReason())
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
