package server

import (
	"context"
	"encoding/json"
	"net/http"
	"path/filepath"
	"strings"
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
	created   int64
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
		created:   time.Now().Unix(),
	}
}

func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	// OpenAI-compatible surface only
	mux.HandleFunc("GET /v1/models", s.handleModels)
	mux.HandleFunc("GET /v1/models/{model}", s.handleModel)
	mux.HandleFunc("POST /v1/chat/completions", s.handleChatCompletions)
	mux.HandleFunc("POST /v1/completions", s.handleCompletions)
	mux.HandleFunc("POST /v1/embeddings", s.handleEmbeddings)

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

type modelsResponse struct {
	Object string     `json:"object"`
	Data   []modelRef `json:"data"`
}

type modelRef struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

func (s *Server) modelID() string {
	if meta := s.engine.Metadata(); meta != nil {
		if name, err := meta.String("general.name"); err == nil && name != "" {
			return name
		}
	}

	if s.modelPath != "" {
		base := filepath.Base(s.modelPath)
		return strings.TrimSuffix(base, filepath.Ext(base))
	}

	return "gguf"
}

func (s *Server) modelRef() modelRef {
	return modelRef{
		ID:      s.modelID(),
		Object:  "model",
		Created: s.created,
		OwnedBy: "gogguf",
	}
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

func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, modelsResponse{
		Object: "list",
		Data:   []modelRef{s.modelRef()},
	})
}

func (s *Server) handleModel(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("model")
	ref := s.modelRef()
	if id != ref.ID {
		writeAPIError(w, http.StatusNotFound, "model not found", "invalid_request_error")
		return
	}

	writeJSON(w, ref)
}

type apiErrorResponse struct {
	Error apiErrorBody `json:"error"`
}

type apiErrorBody struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    string `json:"code,omitempty"`
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
