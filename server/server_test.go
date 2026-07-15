package server

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/magomedcoder/gogguf/pkg/runtime"
)

func TestHealth(t *testing.T) {
	srv := New(&runtime.Engine{}, "")
	rec := httptest.NewRecorder()
	srv.handleHealth(rec, httptest.NewRequest(http.MethodGet, "/v1/health", nil))

	if rec.Code != http.StatusOK {
		t.Fatalf("статус = %d, ожидали 200", rec.Code)
	}

	var resp healthResponse
	if err := json.NewDecoder(rec.Body).Decode(&resp); err != nil {
		t.Fatalf("не удалось разобрать ответ: %v", err)
	}

	if resp.Status != "ok" {
		t.Fatalf("status = %q, ожидали ok", resp.Status)
	}
}

func TestModelsFormat(t *testing.T) {
	srv := New(&runtime.Engine{}, "/models/test.gguf")
	rec := httptest.NewRecorder()
	srv.handleModels(rec, httptest.NewRequest(http.MethodGet, "/v1/models", nil))

	if rec.Code != http.StatusOK {
		t.Fatalf("статус = %d, ожидали 200", rec.Code)
	}

	var resp modelsResponse
	if err := json.NewDecoder(rec.Body).Decode(&resp); err != nil {
		t.Fatalf("не удалось разобрать ответ: %v", err)
	}

	if len(resp.Data) != 1 || resp.Data[0].ID == "" {
		t.Fatalf("ожидали одну модель с id, получили %+v", resp)
	}
}

func TestHandlerRoutes(t *testing.T) {
	srv := New(&runtime.Engine{}, "")
	h := srv.Handler()

	for _, path := range []string{"/v1/health", "/v1/models"} {
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, path, nil))
		if rec.Code != http.StatusOK {
			t.Fatalf("%s: статус = %d, ожидали 200", path, rec.Code)
		}
	}
}

func TestEmbeddingsNotSupported(t *testing.T) {
	srv := New(&runtime.Engine{}, "")
	rec := httptest.NewRecorder()
	srv.handleEmbeddings(rec, httptest.NewRequest(http.MethodPost, "/v1/embeddings", nil))

	if rec.Code != http.StatusNotImplemented {
		t.Fatalf("статус = %d, ожидали 501", rec.Code)
	}
}

func TestReset(t *testing.T) {
	srv := New(&runtime.Engine{}, "")
	rec := httptest.NewRecorder()
	srv.handleReset(rec, httptest.NewRequest(http.MethodPost, "/v1/reset", nil))

	if rec.Code != http.StatusOK {
		t.Fatalf("статус = %d, ожидали 200", rec.Code)
	}

	var resp healthResponse
	if err := json.NewDecoder(rec.Body).Decode(&resp); err != nil {
		t.Fatalf("не удалось разобрать ответ: %v", err)
	}

	if resp.Status != "ok" {
		t.Fatalf("status = %q, ожидали ok", resp.Status)
	}
}

func TestChatCompletionsBadRequest(t *testing.T) {
	srv := New(&runtime.Engine{}, "")
	body := bytes.NewBufferString(`{"messages":[]}`)
	rec := httptest.NewRecorder()
	srv.handleChatCompletions(rec, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", body))

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("статус = %d, ожидали 400", rec.Code)
	}
}
