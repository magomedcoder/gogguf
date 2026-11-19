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
	Logprobs         *int            `json:"logprobs,omitempty"` // 0..5; nil = off (OpenAI legacy int, not chat bool)
	Echo             bool            `json:"echo,omitzero"`
	Suffix           string          `json:"suffix,omitempty"` // appended to prefill only (not OpenAI insert semantics)
}

type completionsResponse struct {
	streamMeta
	Choices []completionsChoice `json:"choices"`
	Usage   usageTokens         `json:"usage"`
}

type completionsChoice struct {
	Index        int                        `json:"index"`
	Text         string                     `json:"text"`
	Logprobs     *completionsChoiceLogprobs `json:"logprobs"` // null when unused
	FinishReason string                     `json:"finish_reason"`
}

// completionsChoiceLogprobs is the OpenAI legacy parallel-array logprobs object (generated tokens only).
type completionsChoiceLogprobs struct {
	Tokens        []string             `json:"tokens"`
	TokenLogprobs []*float64           `json:"token_logprobs"`
	TopLogprobs   []map[string]float64 `json:"top_logprobs"`
	TextOffset    []int                `json:"text_offset"`
}

type completionsStreamChunk struct {
	streamMeta
	Choices []completionsStreamChoice `json:"choices"`
}

type completionsStreamChoice struct {
	Index        int                        `json:"index"`
	Text         string                     `json:"text"`
	Logprobs     *completionsChoiceLogprobs `json:"logprobs,omitempty"`
	FinishReason *string                    `json:"finish_reason"`
}

// completionsPrompt is either a text string (encode) or raw token IDs (no encode).
type completionsPrompt struct {
	text   string
	tokens []int
	isTok  bool
}

func parseCompletionsPrompt(raw json.RawMessage) (completionsPrompt, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return completionsPrompt{}, fmt.Errorf("prompt is required")
	}

	if raw[0] == '"' {
		var s string
		if err := unmarshalWire(raw, &s); err != nil {
			return completionsPrompt{}, err
		}

		return completionsPrompt{text: s}, nil
	}

	if raw[0] == '[' {
		var strs []string
		strOK := unmarshalWire(raw, &strs) == nil
		var ints []int
		intOK := unmarshalWire(raw, &ints) == nil

		// Prefer non-empty string array, then non-empty token IDs. Empty [] matches both types -> reject.
		if strOK && len(strs) > 0 {
			return completionsPrompt{text: strings.Join(strs, "")}, nil
		}

		if intOK && len(ints) > 0 {
			return completionsPrompt{tokens: ints, isTok: true}, nil
		}

		if strOK || intOK {
			return completionsPrompt{}, fmt.Errorf("prompt array must not be empty")
		}

		var batches [][]int
		if err := unmarshalWire(raw, &batches); err == nil {
			return completionsPrompt{}, fmt.Errorf("batch token prompts are not supported (n=1 only); use a flat []int")
		}

		return completionsPrompt{}, fmt.Errorf("prompt: expected string, []string, or []int")
	}

	return completionsPrompt{}, fmt.Errorf("invalid prompt")
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

	wantLogprobs := req.Logprobs != nil
	topLogprobs := 0
	if wantLogprobs {
		topLogprobs = *req.Logprobs
		if topLogprobs < 0 || topLogprobs > 5 {
			writeAPIError(w, http.StatusBadRequest, "logprobs must be between 0 and 5", "invalid_request_error")
			return
		}
	}

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
	genParams.Logprobs = wantLogprobs
	genParams.TopLogprobs = topLogprobs

	s.inferMu.Lock()
	defer s.inferMu.Unlock()

	if err := s.ensureKV(conv); err != nil {
		writeAPIError(w, http.StatusInternalServerError, err.Error(), "server_error")
		return
	}

	echoText, prefill, err := s.completionsPrefill(conv, prompt, req.Suffix)
	if err != nil {
		writeAPIError(w, http.StatusInternalServerError, err.Error(), "server_error")
		return
	}

	if req.Stream {
		s.serveCompletionsStream(w, conv, echoText, prefill, modelName, genParams, req.Echo)
		return
	}

	snap := conv.TokenCount()
	sess, err := s.startCompletionsGeneration(conv, prefill)
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

	genText := trimStopSuffix(sess.GeneratedText(), stops)
	text := genText
	if req.Echo {
		text = echoText + genText
	}

	var lp *completionsChoiceLogprobs
	if wantLogprobs {
		lp = buildCompletionsLogprobs(sess, text, len(echoText), req.Echo)
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
			Logprobs:     lp,
			FinishReason: sess.StopReason(),
		}},
		Usage: usageTokens{
			PromptTokens:     sess.PromptTokenCount(),
			CompletionTokens: sess.GeneratedCount(),
			TotalTokens:      sess.PromptTokenCount() + sess.GeneratedCount(),
		},
	})
}

// completionsPrefill resolves echo display text and the token/text payload for StartGeneration.
// Suffix is appended only to prefill (not echoed). Token prompts keep IDs; suffix text is encoded and appended.
type completionsPrefill struct {
	text   string
	tokens []int
	isTok  bool
}

func (s *Server) completionsPrefill(conv *runtime.Conversation, prompt completionsPrompt, suffix string) (echoText string, prefill completionsPrefill, err error) {
	if prompt.isTok {
		echoText = conv.DecodeTokens(prompt.tokens)
		prefill = completionsPrefill{tokens: append([]int(nil), prompt.tokens...), isTok: true}
		if suffix != "" {
			suf, encErr := conv.EncodeText(suffix)
			if encErr != nil {
				return "", completionsPrefill{}, encErr
			}

			prefill.tokens = append(prefill.tokens, suf...)
		}

		return echoText, prefill, nil
	}

	echoText = prompt.text
	prefill = completionsPrefill{
		text: prompt.text + suffix,
	}

	return echoText, prefill, nil
}

func (s *Server) startCompletionsGeneration(conv *runtime.Conversation, prefill completionsPrefill) (*runtime.GenerationSession, error) {
	if prefill.isTok {
		return conv.StartGenerationTokens(prefill.tokens)
	}

	return conv.StartGeneration(prefill.text)
}

func buildCompletionsLogprobs(sess *runtime.GenerationSession, fullText string, echoLen int, echo bool) *completionsChoiceLogprobs {
	entries := sess.Logprobs()
	if len(entries) == 0 {
		return nil
	}

	out := &completionsChoiceLogprobs{
		Tokens:        make([]string, 0, len(entries)),
		TokenLogprobs: make([]*float64, 0, len(entries)),
		TopLogprobs:   make([]map[string]float64, 0, len(entries)),
		TextOffset:    make([]int, 0, len(entries)),
	}

	// Offsets are relative to fullText: when echo, generated text starts at echoLen.
	off := 0
	if echo {
		off = echoLen
	}

	for _, e := range entries {
		tok := sess.DecodeToken(e.TokenID)
		lp := e.Logprob
		out.Tokens = append(out.Tokens, tok)
		out.TokenLogprobs = append(out.TokenLogprobs, float64Ptr(lp))
		out.TextOffset = append(out.TextOffset, off)
		off += len(tok)

		top := map[string]float64{}
		for _, t := range e.Top {
			top[sess.DecodeToken(t.ID)] = t.Logprob
		}

		out.TopLogprobs = append(out.TopLogprobs, top)
	}

	_ = fullText // offsets are built from decoded tokens (same as concatenation of generated pieces)

	return out
}

func float64Ptr(v float64) *float64 {
	return &v
}

// lastCompletionsLogprobs is a single-index parallel-array object for one stream token chunk.
func lastCompletionsLogprobs(sess *runtime.GenerationSession, textOffset int) *completionsChoiceLogprobs {
	entries := sess.Logprobs()
	if len(entries) == 0 {
		return nil
	}

	e := entries[len(entries)-1]
	tok := sess.DecodeToken(e.TokenID)
	top := map[string]float64{}
	for _, t := range e.Top {
		top[sess.DecodeToken(t.ID)] = t.Logprob
	}

	return &completionsChoiceLogprobs{
		Tokens:        []string{tok},
		TokenLogprobs: []*float64{float64Ptr(e.Logprob)},
		TopLogprobs:   []map[string]float64{top},
		TextOffset:    []int{textOffset},
	}
}

func (s *Server) serveCompletionsStream(w http.ResponseWriter, conv *runtime.Conversation, echoText string, prefill completionsPrefill, model string, params runtime.GenerateParams, echo bool) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeAPIError(w, http.StatusInternalServerError, "streaming is not supported", "server_error")
		return
	}

	snap := conv.TokenCount()
	sess, err := s.startCompletionsGeneration(conv, prefill)
	if err != nil {
		writeAPIError(w, http.StatusInternalServerError, err.Error(), "server_error")
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	id := fmt.Sprintf("cmpl-%d", time.Now().UnixNano())
	created := time.Now().Unix()

	writeChunk := func(text string, finish *string, logprobs *completionsChoiceLogprobs) {
		chunk := completionsStreamChunk{
			ID:      id,
			Object:  "text_completion",
			Created: created,
			Model:   model,
			Choices: []completionsStreamChoice{{
				Index:        0,
				Text:         text,
				Logprobs:     logprobs,
				FinishReason: finish,
			}},
		}
		_ = writeSSEJSON(w, chunk)
		flusher.Flush()
	}

	textOff := 0
	if echo && echoText != "" {
		writeChunk(echoText, nil, nil)
		textOff = len(echoText)
	}

	params.OnToken = func(tokenID int) bool {
		tok := sess.DecodeToken(tokenID)
		var lp *completionsChoiceLogprobs
		if params.Logprobs {
			lp = lastCompletionsLogprobs(sess, textOff)
		}
		writeChunk(tok, nil, lp)
		textOff += len(tok)
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
	writeChunk("", &finish, nil)
	fmt.Fprintf(w, "data: [DONE]\n\n")
	flusher.Flush()
}
