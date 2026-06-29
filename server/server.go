package server

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
	"time"

	"github.com/magomedcoder/gguf.go/pkg/chat"
	"github.com/magomedcoder/gguf.go/pkg/runtime"
	"github.com/magomedcoder/gguf.go/pkg/sampler"
)

type Server struct {
	engine    *runtime.Engine
	modelPath string
	mu        sync.Mutex
}

func New(engine *runtime.Engine, modelPath string) *Server {
	return &Server{
		engine:    engine,
		modelPath: modelPath,
	}
}

func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/health", s.handleHealth)
	mux.HandleFunc("/models", s.handleModels)
	mux.HandleFunc("/generate", s.handleGenerate)
	mux.HandleFunc("/completions", s.handleChatCompletions)
	return mux
}

func (s *Server) ListenAndServe(addr string) error {
	srv := &http.Server{
		Addr:              addr,
		Handler:           s.Handler(),
		ReadHeaderTimeout: 10 * time.Second,
	}
	return srv.ListenAndServe()
}

// Run блокируется до ctx.Done() или ошибки ListenAndServe
func (s *Server) Run(ctx context.Context, addr string) error {
	srv := &http.Server{
		Addr:              addr,
		Handler:           s.Handler(),
		ReadHeaderTimeout: 10 * time.Second,
	}

	errCh := make(chan error, 1)
	go func() { errCh <- srv.ListenAndServe() }()

	select {
	case err := <-errCh:
		if err != nil && err != http.ErrServerClosed {
			return err
		}
	case <-ctx.Done():
	}

	shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	return srv.Shutdown(shutdownCtx)
}

type healthResponse struct {
	Status string `json:"status"`
}

type modelInfo struct {
	ID            string `json:"id"`
	Path          string `json:"path,omitempty"`
	Architecture  string `json:"architecture,omitempty"`
	Name          string `json:"name,omitempty"`
	ContextLength int    `json:"context_length,omitempty"`
	ChatTemplate  bool   `json:"chat_template"`
}

type modelsResponse struct {
	Models []modelInfo `json:"models"`
}

func (s *Server) handleHealth(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, healthResponse{Status: "ok"})
}

func (s *Server) handleModels(w http.ResponseWriter, _ *http.Request) {
	meta := s.engine.Metadata()
	arch, _ := meta.String("general.architecture")
	name, _ := meta.String("general.name")

	ctxLen := 0
	if arch != "" {
		ctxLen = meta.IntOptional(arch+".context_length", 0)
	}

	writeJSON(w, modelsResponse{
		Models: []modelInfo{{
			ID:            name,
			Path:          s.modelPath,
			Architecture:  arch,
			Name:          name,
			ContextLength: ctxLen,
			ChatTemplate:  chat.HasTemplateMeta(meta),
		}},
	})
}

type completionRequest struct {
	Prompt        string  `json:"prompt"`
	MaxTokens     int     `json:"max_tokens"`
	Temperature   float32 `json:"temperature"`
	TopK          int     `json:"top_k"`
	TopP          float32 `json:"top_p"`
	MinP          float32 `json:"min_p"`
	RepeatPenalty float32 `json:"repeat_penalty"`
	RepeatLastN   int     `json:"repeat_last_n"`
	Seed          uint64  `json:"seed"`
	Chat          bool    `json:"chat"`
	Stream        bool    `json:"stream"`
	System        string  `json:"system"`
	Thinking      *bool   `json:"thinking"`
}

type completionResponse struct {
	Text   string `json:"text"`
	Tokens int    `json:"tokens"`
}

type streamEvent struct {
	Token string `json:"token,omitempty"`
	Done  bool   `json:"done,omitempty"`
	Error string `json:"error,omitempty"`
}

func (s *Server) handleGenerate(w http.ResponseWriter, r *http.Request) {
	var req completionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	if req.Prompt == "" {
		http.Error(w, "prompt обязателен", http.StatusBadRequest)
		return
	}

	if req.MaxTokens <= 0 {
		req.MaxTokens = 128
	}

	if req.TopP <= 0 {
		req.TopP = 1
	}

	prompt := req.Prompt
	if req.Chat {
		var err error
		prompt, err = chat.FormatUser(req.Prompt, chat.Options{
			System:   req.System,
			Thinking: req.Thinking,
			Metadata: s.engine.Metadata(),
		})
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	ctx, err := s.engine.NewContext()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	params := runtime.GenerateParams{
		MaxTokens: req.MaxTokens,
		Sampler: sampler.New(sampler.Config{
			Temp: req.Temperature,
			TopK: req.TopK,
			TopP: req.TopP,
			MinP: req.MinP,
			Seed: req.Seed,
		}),
		RepeatPenalty: req.RepeatPenalty,
		RepeatLastN:   req.RepeatLastN,
	}

	if req.Stream {
		s.serveStream(w, ctx, prompt, params)
		return
	}

	var tokenCount int
	params.OnToken = func(int) bool {
		tokenCount++
		return true
	}

	text, err := ctx.Generate(prompt, params)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	writeJSON(w, completionResponse{
		Text:   text,
		Tokens: tokenCount,
	})
}

func (s *Server) serveStream(w http.ResponseWriter, ctx *runtime.Context, prompt string, params runtime.GenerateParams) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming не поддерживается", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	writeEvent := func(ev streamEvent) {
		data, _ := json.Marshal(ev)
		fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()
	}

	params.OnToken = func(id int) bool {
		writeEvent(streamEvent{
			Token: ctx.DecodeToken(id),
		})
		return true
	}

	if _, err := ctx.Generate(prompt, params); err != nil {
		writeEvent(streamEvent{
			Error: err.Error(),
		})
		return
	}

	writeEvent(streamEvent{Done: true})
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(v)
}
