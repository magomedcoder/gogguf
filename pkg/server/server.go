package server

import (
	"context"
	"net/http"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/magomedcoder/gogguf/pkg/runtime"
)

// Options configures the HTTP server (auth, rate limit, model id)
type Options struct {
	APIKey             string // empty = no auth
	RateLimitPerMinute int    // 0 = unlimited; token bucket per IP
	ModelID            string // OpenAI model id; empty = metadata name or path basename
}

// SessionIDHeader selects a multi-turn conversation when set (OpenAI `user` is a fallback key).
const SessionIDHeader = "X-Session-Id"

type Server struct {
	engine    *runtime.Engine
	modelPath string
	forcedID  string
	apiKey    string
	limiter   *rateLimiter
	created   int64

	sessionsMu sync.Mutex
	sessions   map[string]*runtime.Conversation

	inferMu sync.Mutex
	kvConv  *runtime.Conversation // conversation whose token history matches model KV; nil if stale/empty
}

// New creates a server. modelPath is optional (used only when Options.ModelID and metadata name are empty).
func New(engine *runtime.Engine, modelPath string, opts ...Options) *Server {
	var o Options
	if len(opts) > 0 {
		o = opts[0]
	}
	return &Server{
		engine:    engine,
		modelPath: modelPath,
		forcedID:  o.ModelID,
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
	h = withRequestID(h)

	return h
}

func (s *Server) ListenAndServe(addr string) error {
	srv := &http.Server{
		Addr:                addr,
		Handler:             s.Handler(),
		ReadHeaderTimeout:   10 * time.Second,
		MaxHeaderValueCount: http.DefaultMaxHeaderValueCount,
	}
	return srv.ListenAndServe()
}

// Run blocks until ctx.Done() or ListenAndServe returns an error.
func (s *Server) Run(ctx context.Context, addr string) error {
	srv := &http.Server{
		Addr:                addr,
		Handler:             s.Handler(),
		ReadHeaderTimeout:   10 * time.Second,
		MaxHeaderValueCount: http.DefaultMaxHeaderValueCount,
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
	ID              string `json:"id"`
	Object          string `json:"object"`
	Created         int64  `json:"created"`
	OwnedBy         string `json:"owned_by"`
	Architecture    string `json:"architecture,omitempty"`
	ContextLength   int    `json:"context_length,omitempty"`
	EmbeddingLength int    `json:"embedding_length,omitempty"`
}

func (s *Server) resolveModelID() string {
	if id := strings.TrimSpace(s.forcedID); id != "" {
		return id
	}

	if meta := s.engine.Metadata(); meta != nil {
		if name, err := meta.String("general.name"); err == nil && name != "" {
			return name
		}
	}

	if s.modelPath != "" {
		base := filepath.Base(s.modelPath)
		return strings.TrimSuffix(base, filepath.Ext(base))
	}

	return "model"
}

func (s *Server) modelRef() modelRef {
	ref := modelRef{
		ID:      s.resolveModelID(),
		Object:  "model",
		Created: s.created,
		OwnedBy: "gogguf",
	}
	if s.engine != nil {
		ref.Architecture = s.engine.Architecture()
		ref.ContextLength = s.engine.ContextLength()
		ref.EmbeddingLength = s.engine.EmbeddingLength()
	}

	return ref
}

func sessionIDFromRequest(r *http.Request, user string) string {
	if id := strings.TrimSpace(r.Header.Get(SessionIDHeader)); id != "" {
		return id
	}

	if u := strings.TrimSpace(user); u != "" {
		return "user:" + u
	}

	return ""
}

func (s *Server) newConversation() (*runtime.Conversation, error) {
	ctx, err := s.engine.NewContext()
	if err != nil {
		return nil, err
	}

	return ctx.NewConversation(), nil
}

// conversationForSession returns a per-request Conversation or one keyed by sessionID.
func (s *Server) conversationForSession(sessionID string) (*runtime.Conversation, error) {
	if sessionID == "" {
		return s.newConversation()
	}

	s.sessionsMu.Lock()
	defer s.sessionsMu.Unlock()

	if conv, ok := s.sessions[sessionID]; ok {
		return conv, nil
	}

	conv, err := s.newConversation()
	if err != nil {
		return nil, err
	}

	if s.sessions == nil {
		s.sessions = make(map[string]*runtime.Conversation)
	}

	s.sessions[sessionID] = conv
	return conv, nil
}

// ensureKV loads model KV-cache to match conv when another session or embeddings used the model.
func (s *Server) ensureKV(conv *runtime.Conversation) error {
	if s.kvConv == conv {
		return nil
	}

	if err := conv.LoadKVCache(); err != nil {
		return err
	}

	s.kvConv = conv
	return nil
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
	data, _ := marshalWire(v)
	_, _ = w.Write(data)
}

func writeAPIError(w http.ResponseWriter, status int, message, typ string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	data, _ := marshalWire(apiErrorResponse{
		Error: apiErrorBody{
			Message: message,
			Type:    typ,
		},
	})
	_, _ = w.Write(data)
}
