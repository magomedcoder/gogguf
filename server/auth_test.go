package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/magomedcoder/gogguf/pkg/runtime"
)

func TestAPIKeyRequired(t *testing.T) {
	srv := New(&runtime.Engine{}, "", Options{APIKey: "secret"})
	h := srv.Handler()

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/v1/models", nil))
	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("без ключа: статус=%d, ожидали 401", rec.Code)
	}

	var errBody apiErrorResponse
	if err := json.NewDecoder(rec.Body).Decode(&errBody); err != nil {
		t.Fatal(err)
	}

	if errBody.Error.Type != "authentication_error" {
		t.Fatalf("type=%q", errBody.Error.Type)
	}

	// health without API key
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/v1/health", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("health: статус=%d", rec.Code)
	}

	// Bearer
	rec = httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	req.Header.Set("Authorization", "Bearer secret")
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("Bearer: статус=%d", rec.Code)
	}

	// X-API-Key
	rec = httptest.NewRecorder()
	req = httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	req.Header.Set("X-API-Key", "secret")
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("X-API-Key: статус=%d", rec.Code)
	}

	// wrong API key
	rec = httptest.NewRecorder()
	req = httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	req.Header.Set("Authorization", "Bearer wrong")
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("wrong key: статус=%d", rec.Code)
	}
}

func TestRateLimit(t *testing.T) {
	srv := New(&runtime.Engine{}, "", Options{RateLimitPerMinute: 2})
	h := srv.Handler()

	ok := 0
	limited := 0
	for range 5 {
		rec := httptest.NewRecorder()
		req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
		req.RemoteAddr = "10.0.0.1:1234"
		h.ServeHTTP(rec, req)
		switch rec.Code {
		case http.StatusOK:
			ok++
		case http.StatusTooManyRequests:
			limited++
			var errBody apiErrorResponse
			_ = json.NewDecoder(rec.Body).Decode(&errBody)
			if errBody.Error.Type != "rate_limit_error" {
				t.Fatalf("type=%q", errBody.Error.Type)
			}
		default:
			t.Fatalf("неожиданный статус %d", rec.Code)
		}
	}
	if ok < 1 || limited < 1 {
		t.Fatalf("ok=%d limited=%d, ожидали оба >0", ok, limited)
	}

	// health is not rate-limited
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/v1/health", nil)
	req.RemoteAddr = "10.0.0.1:1234"
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("health под лимитом: %d", rec.Code)
	}
}

func TestAPIKeyFromRequest(t *testing.T) {
	req := httptest.NewRequest(http.MethodGet, "/", nil)
	req.Header.Set("Authorization", "Bearer abc")
	if got := apiKeyFromRequest(req); got != "abc" {
		t.Fatalf("Bearer: %q", got)
	}

	req = httptest.NewRequest(http.MethodGet, "/", nil)
	req.Header.Set("X-API-Key", "xyz")
	if got := apiKeyFromRequest(req); got != "xyz" {
		t.Fatalf("X-API-Key: %q", got)
	}
}
