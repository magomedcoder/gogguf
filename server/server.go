package server

import (
	"context"
	"encoding/json"
	"net/http"
	"sync"
	"time"

	"github.com/magomedcoder/gogguf/pkg/runtime"
)

// Options configures the HTTP server (auth, rate limit).
type Options struct {
	APIKey             string // empty = no auth
	RateLimitPerMinute int    // 0 = unlimited; token bucket per IP
}

type Server struct {
	engine    *runtime.Engine
	modelPath string
	apiKey    string
	limiter   *rateLimiter
	mu        sync.Mutex
	conv      *runtime.Conversation
}

// New creates a server. opts may be omitted.
func New(engine *runtime.Engine, modelPath string, opts ...Options) *Server {
	var o Options
	if len(opts) > 0 {
		o = opts[0]
	}
	return &Server{
		engine:    engine,
		modelPath: modelPath,
		apiKey:    o.APIKey,
		limiter:   newRateLimiter(o.RateLimitPerMinute),
	}
}

func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/health", s.handleHealth)
	mux.HandleFunc("/v1/models", s.handleModels)
	mux.HandleFunc("/v1/reset", s.handleReset)
	mux.HandleFunc("/v1/chat/completions", s.handleChatCompletions)
	mux.HandleFunc("/v1/embeddings", s.handleEmbeddings)

	var h http.Handler = mux
	h = withRateLimit(h, s.limiter)
	h = withAPIKey(h, s.apiKey)
	return h
}

func (s *Server) ListenAndServe(addr string) error {
	srv := &http.Server{
		Addr:              addr,
		Handler:           s.Handler(),
		ReadHeaderTimeout: 10 * time.Second,
	}
	return srv.ListenAndServe()
}

// Run blocks until ctx.Done() or ListenAndServe returns an error.
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

type modelsResponse struct {
	Object string     `json:"object"`
	Data   []modelRef `json:"data"`
}

type modelRef struct {
	ID     string `json:"id"`
	Object string `json:"object"`
}

func (s *Server) conversation() (*runtime.Conversation, error) {
	if s.conv == nil {
		ctx, err := s.engine.NewContext()
		if err != nil {
			return nil, err
		}

		s.conv = ctx.NewConversation()
	}

	return s.conv, nil
}

func (s *Server) handleHealth(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, healthResponse{Status: "ok"})
}

func (s *Server) handleReset(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "метод не разрешен", http.StatusMethodNotAllowed)
		return
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if s.conv != nil {
		s.conv.Reset()
	}

	writeJSON(w, healthResponse{Status: "ok"})
}

func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "метод не разрешен", http.StatusMethodNotAllowed)
		return
	}

	name := ""
	if meta := s.engine.Metadata(); meta != nil {
		name, _ = meta.String("general.name")
	}
	if name == "" {
		name = "gguf"
	}

	writeJSON(w, modelsResponse{
		Object: "list",
		Data: []modelRef{{
			ID:     name,
			Object: "model",
		}},
	})
}

type apiErrorResponse struct {
	Error apiErrorBody `json:"error"`
}

type apiErrorBody struct {
	Message string `json:"message"`
	Type    string `json:"type"`
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(v)
}

func writeAPIError(w http.ResponseWriter, status int, message, typ string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(apiErrorResponse{
		Error: apiErrorBody{
			Message: message,
			Type:    typ,
		},
	})
}
