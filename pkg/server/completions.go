package server

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/magomedcoder/gogguf/pkg/runtime"
)

// Legacy OpenAI Completions API (prefer /v1/chat/completions)
type completionsRequest struct {
	Model            string          `json:"model"`
	Prompt           json.RawMessage `json:"prompt"`
	MaxTokens        int             `json:"max_tokens,omitzero"`
	Temperature      *float64        `json:"temperature,omitempty"`
	TopP             *float64        `json:"top_p,omitempty"`
	N                int             `json:"n,omitzero"`
	Stream           bool            `json:"stream,omitzero"`
	Stop             json.RawMessage `json:"stop,omitempty"`
	PresencePenalty  *float64        `json:"presence_penalty,omitempty"`
	FrequencyPenalty *float64        `json:"frequency_penalty,omitempty"`
	User             string          `json:"user,omitempty"`
	Seed             *int64          `json:"seed,omitempty"`
	LogitBias        json.RawMessage `json:"logit_bias,omitempty"`
	Logprobs         *int            `json:"logprobs,omitempty"` // stub
	Echo             bool            `json:"echo,omitzero"`
	Suffix           string          `json:"suffix,omitempty"` // stub
}

type completionsResponse struct {
	ID                string              `json:"id"`
	Object            string              `json:"object"`
	Created           int64               `json:"created"`
	Model             string              `json:"model"`
	SystemFingerprint string              `json:"system_fingerprint,omitempty"`
	Choices           []completionsChoice `json:"choices"`
	Usage             chatCompletionUsage `json:"usage"`
}

type completionsChoice struct {
	Index        int       `json:"index"`
	Text         string    `json:"text"`
	Logprobs     *struct{} `json:"logprobs"`
	FinishReason string    `json:"finish_reason"`
}

type completionsStreamChunk struct {
	ID      string                    `json:"id"`
	Object  string                    `json:"object"`
	Created int64                     `json:"created"`
	Model   string                    `json:"model"`
	Choices []completionsStreamChoice `json:"choices"`
}

type completionsStreamChoice struct {
	Index        int       `json:"index"`
	Text         string    `json:"text"`
	Logprobs     *struct{} `json:"logprobs"`
	FinishReason *string   `json:"finish_reason"`
}

func parseCompletionsPrompt(raw json.RawMessage) (string, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return "", fmt.Errorf("prompt is required")
	}

	if raw[0] == '"' {
		var s string
		if err := unmarshalWire(raw, &s); err != nil {
			return "", err
		}

		return s, nil
	}

	if raw[0] == '[' {
		var strs []string
		if err := unmarshalWire(raw, &strs); err == nil {
			return strings.Join(strs, ""), nil
		}

		return "", fmt.Errorf("prompt token arrays are not supported; use a string or string array")
	}

	return "", fmt.Errorf("invalid prompt")
}

func (s *Server) handleCompletions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeAPIError(w, http.StatusMethodNotAllowed, "method not allowed", "invalid_request_error")
		return
	}

	var req completionsRequest
	if err := decodeWire(r, &req); err != nil {
		writeAPIError(w, http.StatusBadRequest, err.Error(), "invalid_request_error")
		return
	}

	prompt, err := parseCompletionsPrompt(req.Prompt)
	if err != nil {
		writeAPIError(w, http.StatusBadRequest, err.Error(), "invalid_request_error")
		return
	}

	if req.N > 1 {
		writeAPIError(w, http.StatusBadRequest, "n > 1 is not supported (stub: only n=1)", "invalid_request_error")
		return
	}

	// Stubs: logprobs, suffix are accepted and ignored.

	bias, err := parseLogitBias(req.LogitBias)
	if err != nil {
		writeAPIError(w, http.StatusBadRequest, err.Error(), "invalid_request_error")
		return
	}

	stops := parseStop(req.Stop)
	modelName := s.resolveModelID()
	if req.Model != "" {
		modelName = req.Model
	}

	conv, err := s.conversationForSession(sessionIDFromRequest(r, req.User))
	if err != nil {
		writeAPIError(w, http.StatusInternalServerError, err.Error(), "server_error")
		return
	}

	genParams := baseGenerateParams(req.MaxTokens, 0, req.TopP, req.Temperature, req.Seed, req.FrequencyPenalty, req.PresencePenalty, bias, stops)

	s.inferMu.Lock()
	defer s.inferMu.Unlock()

	if err := s.ensureKV(conv); err != nil {
		writeAPIError(w, http.StatusInternalServerError, err.Error(), "server_error")
		return
	}

	if req.Stream {
		s.serveCompletionsStream(w, conv, prompt, modelName, genParams, req.Echo)
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
	if req.Echo {
		text = prompt + text
	}

	writeJSON(w, completionsResponse{
		ID:                fmt.Sprintf("cmpl-%d", time.Now().UnixNano()),
		Object:            "text_completion",
		Created:           time.Now().Unix(),
		Model:             modelName,
		SystemFingerprint: systemFingerprint(modelName),
		Choices: []completionsChoice{{
			Index:        0,
			Text:         text,
			Logprobs:     nil,
			FinishReason: sess.StopReason(),
		}},
		Usage: chatCompletionUsage{
			PromptTokens:     sess.PromptTokenCount(),
			CompletionTokens: sess.GeneratedCount(),
			TotalTokens:      sess.PromptTokenCount() + sess.GeneratedCount(),
		},
	})
}

func (s *Server) serveCompletionsStream(w http.ResponseWriter, conv *runtime.Conversation, prompt, model string, params runtime.GenerateParams, echo bool) {
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

	id := fmt.Sprintf("cmpl-%d", time.Now().UnixNano())
	created := time.Now().Unix()

	writeChunk := func(text string, finish *string) {
		chunk := completionsStreamChunk{
			ID:      id,
			Object:  "text_completion",
			Created: created,
			Model:   model,
			Choices: []completionsStreamChoice{{
				Index:        0,
				Text:         text,
				Logprobs:     nil,
				FinishReason: finish,
			}},
		}
		_ = writeSSEJSON(w, chunk)
		flusher.Flush()
	}

	if echo && prompt != "" {
		writeChunk(prompt, nil)
	}

	params.OnToken = func(tokenID int) bool {
		writeChunk(sess.DecodeToken(tokenID), nil)
		return true
	}

	if err := sess.GenerateSteps(params); err != nil {
		conv.Rollback(snap)
		_ = writeSSEJSON(w, apiErrorResponse{Error: apiErrorBody{Message: err.Error(), Type: "server_error"}})
		flusher.Flush()
		return
	}

	conv.Commit(sess)

	finish := sess.StopReason()
	writeChunk("", &finish)
	fmt.Fprintf(w, "data: [DONE]\n\n")
	flusher.Flush()
}
