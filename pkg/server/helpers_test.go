package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/magomedcoder/gogguf/pkg/runtime"
)

func TestParseLogitBias(t *testing.T) {
	got, err := parseLogitBias(json.RawMessage(`{"42":1.5,"7":-2}`))
	if err != nil {
		t.Fatal(err)
	}

	if got[42] != 1.5 || got[7] != -2 {
		t.Fatalf("got %#v", got)
	}
}

func TestParseResponseFormat(t *testing.T) {
	typ, err := parseResponseFormat(json.RawMessage(`{"type":"json_object"}`))
	if err != nil || typ != "json_object" {
		t.Fatalf("got %q %v", typ, err)
	}

	if _, err := parseResponseFormat(json.RawMessage(`{"type":"json_schema"}`)); err == nil {
		t.Fatal("expected json_schema error")
	}
}

func TestEnsureJSONObjectOutput(t *testing.T) {
	if err := ensureJSONObjectOutput(`{"a":1}`); err != nil {
		t.Fatal(err)
	}

	if err := ensureJSONObjectOutput("```json\n[1,2]\n```"); err != nil {
		t.Fatal(err)
	}

	if err := ensureJSONObjectOutput("hello"); err == nil {
		t.Fatal("expected error")
	}
}

func TestContentHasMultimodal(t *testing.T) {
	if contentHasMultimodal(json.RawMessage(`"hi"`)) {
		t.Fatal("string content")
	}

	if !contentHasMultimodal(json.RawMessage(`[{"type":"image_url","image_url":{"url":"x"}}]`)) {
		t.Fatal("expected multimodal")
	}
}

func TestRequestIDAndFingerprintHeaders(t *testing.T) {
	h := New(&runtime.Engine{}, "/models/demo.gguf").Handler()
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	req.Header.Set(requestIDHeader, "client-req")
	h.ServeHTTP(rec, req)
	if got := rec.Header().Get(requestIDHeader); got != "client-req" {
		t.Fatalf("X-Request-Id = %q", got)
	}

	if fp := systemFingerprint("demo"); fp == "" || fp[:9] != "fp_gogguf" {
		t.Fatalf("fingerprint = %q", fp)
	}
}
