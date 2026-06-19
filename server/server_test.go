package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/magomedcoder/gguf.go/pkg/runtime"
)

func TestModelsEmptyEngine(t *testing.T) {
	srv := New(&runtime.Engine{}, "/models/test.gguf")
	rec := httptest.NewRecorder()
	srv.handleModels(rec, httptest.NewRequest(http.MethodGet, "/models", nil))

	if rec.Code != http.StatusOK {
		t.Fatalf("статус = %d, ожидали 200", rec.Code)
	}

	var resp modelsResponse
	if err := json.NewDecoder(rec.Body).Decode(&resp); err != nil {
		t.Fatalf("не удалось разобрать ответ: %v", err)
	}

	if len(resp.Models) != 1 {
		t.Fatalf("моделей = %d, ожидали 1", len(resp.Models))
	}

	if resp.Models[0].Path != "/models/test.gguf" {
		t.Fatalf("path = %q, ожидали /models/test.gguf", resp.Models[0].Path)
	}
}

func TestHandlerRoutes(t *testing.T) {
	srv := New(&runtime.Engine{}, "")
	h := srv.Handler()

	for _, path := range []string{"/models"} {
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, path, nil))
		if rec.Code != http.StatusOK {
			t.Fatalf("%s: статус = %d, ожидали 200", path, rec.Code)
		}
	}
}
