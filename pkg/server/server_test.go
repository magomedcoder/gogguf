package server

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/magomedcoder/gogguf/pkg/runtime"
)

func TestModelsFormat(t *testing.T) {
	srv := New(&runtime.Engine{}, "/models/test.gguf")
	rec := httptest.NewRecorder()
	srv.handleModels(rec, httptest.NewRequest(http.MethodGet, "/v1/models", nil))

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, expected 200", rec.Code)
	}

	var resp modelsResponse
	if err := json.NewDecoder(rec.Body).Decode(&resp); err != nil {
		t.Fatalf("failed to parse response: %v", err)
	}

	if len(resp.Data) != 1 || resp.Data[0].ID == "" {
		t.Fatalf("expected one model with id, got %+v", resp)
	}
}

// TestOpenAIAPI smoke-checks the registered /v1 surface without a loaded model
func TestOpenAIAPI(t *testing.T) {
	h := New(&runtime.Engine{}, "/models/demo.gguf").Handler()

	type step struct {
		method string
		path   string
		body   string
		want   int
	}
	steps := []step{
		{http.MethodGet, "/v1/models", "", http.StatusOK},
		{http.MethodGet, "/v1/models/demo", "", http.StatusOK},
		{http.MethodGet, "/v1/models/missing", "", http.StatusNotFound},
		{http.MethodPost, "/v1/chat/completions", `{"messages":[]}`, http.StatusBadRequest},
		{http.MethodPost, "/v1/completions", `{}`, http.StatusBadRequest},
		{http.MethodPost, "/v1/embeddings", `{"input":null}`, http.StatusBadRequest},
		{http.MethodGet, "/v1/health", "", http.StatusNotFound},
		{http.MethodPost, "/v1/moderations", `{"input":"hi"}`, http.StatusNotFound},
	}

	for _, s := range steps {
		var body *bytes.Buffer
		if s.body != "" {
			body = bytes.NewBufferString(s.body)
		}

		var req *http.Request
		if body != nil {
			req = httptest.NewRequest(s.method, s.path, body)
		} else {
			req = httptest.NewRequest(s.method, s.path, nil)
		}

		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, req)
		if rec.Code != s.want {
			t.Fatalf("%s %s: status=%d, want %d, body=%s", s.method, s.path, rec.Code, s.want, rec.Body.String())
		}
	}
}

func TestEmbeddingsBadRequest(t *testing.T) {
	srv := New(&runtime.Engine{}, "")
	rec := httptest.NewRecorder()
	body := bytes.NewBufferString(`{"input":null}`)
	srv.handleEmbeddings(rec, httptest.NewRequest(http.MethodPost, "/v1/embeddings", body))
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, expected 400", rec.Code)
	}
}

func TestParseEmbeddingsInput(t *testing.T) {
	cases := []struct {
		raw  string
		n    int
		toks bool
	}{
		{`"hi"`, 1, false},
		{`["a","b"]`, 2, false},
		{`[1,2,3]`, 1, true},
		{`[[1,2],[3]]`, 2, true},
	}
	for _, tc := range cases {
		got, err := parseEmbeddingsInput(json.RawMessage(tc.raw))
		if err != nil {
			t.Fatalf("%s: %v", tc.raw, err)
		}

		if len(got) != tc.n {
			t.Fatalf("%s: len=%d want %d", tc.raw, len(got), tc.n)
		}

		if tc.toks && got[0].tokens == nil {
			t.Fatalf("%s: expected tokens", tc.raw)
		}

		if !tc.toks && got[0].text == "" && tc.raw != `["a","b"]` {
			// first of ["a","b"] is "a"
		}

		if !tc.toks && len(got) > 0 && got[0].tokens != nil {
			t.Fatalf("%s: did not expect tokens", tc.raw)
		}
	}
}

func TestChatCompletionsBadRequest(t *testing.T) {
	srv := New(&runtime.Engine{}, "")
	body := bytes.NewBufferString(`{"messages":[]}`)
	rec := httptest.NewRecorder()
	srv.handleChatCompletions(rec, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", body))

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, expected 400", rec.Code)
	}
}

func TestCompletionsBadRequest(t *testing.T) {
	srv := New(&runtime.Engine{}, "/models/demo.gguf")
	h := srv.Handler()
	rec := httptest.NewRecorder()
	body := bytes.NewBufferString(`{"max_tokens":8}`)
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/completions", body))
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, expected 400", rec.Code)
	}
}

func TestParseCompletionsPrompt(t *testing.T) {
	got, err := parseCompletionsPrompt(json.RawMessage(`"hi"`))
	if err != nil || got != "hi" {
		t.Fatalf("string: got=%q err=%v", got, err)
	}

	got, err = parseCompletionsPrompt(json.RawMessage(`["a","b"]`))
	if err != nil || got != "ab" {
		t.Fatalf("array: got=%q err=%v", got, err)
	}

	if _, err := parseCompletionsPrompt(json.RawMessage(`null`)); err == nil {
		t.Fatal("expected error for null prompt")
	}
}

func TestModelsOpenAIShape(t *testing.T) {
	srv := New(&runtime.Engine{}, "/models/Qwen3-4B-Q8_0.gguf")
	h := srv.Handler()

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/v1/models", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d", rec.Code)
	}

	var list modelsResponse
	if err := json.NewDecoder(rec.Body).Decode(&list); err != nil {
		t.Fatal(err)
	}

	if list.Object != "list" || len(list.Data) != 1 {
		t.Fatalf("list=%+v", list)
	}

	m := list.Data[0]
	if m.Object != "model" || m.OwnedBy != "gogguf" || m.ID == "" || m.Created == 0 {
		t.Fatalf("model=%+v", m)
	}

	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/v1/models/"+m.ID, nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("retrieve status=%d", rec.Code)
	}

	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/v1/models/missing", nil))
	if rec.Code != http.StatusNotFound {
		t.Fatalf("missing status=%d", rec.Code)
	}
}

func TestNonOpenAIRoutesRemoved(t *testing.T) {
	srv := New(&runtime.Engine{}, "")
	h := srv.Handler()
	for _, tc := range []struct {
		method, path string
	}{
		{http.MethodGet, "/v1/reset"},
		{http.MethodGet, "/v1/health"},
		{http.MethodPost, "/v1/moderations"},
	} {
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, httptest.NewRequest(tc.method, tc.path, nil))
		if rec.Code != http.StatusNotFound {
			t.Fatalf("%s %s: status=%d, expected 404", tc.method, tc.path, rec.Code)
		}
	}
}
