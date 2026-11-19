package server

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/magomedcoder/gogguf/pkg/runtime"
)

func TestBuildCompletionsLogprobs(t *testing.T) {
	// exercise parallel-array shape without a loaded model: decode via a fake session is unavailable, so build the object the same way as lastCompletionsLogprobs / marshal round-trip.
	lp0, lp1 := -0.1, -1.2
	obj := &completionsChoiceLogprobs{
		Tokens:        []string{"Hel", "lo"},
		TokenLogprobs: []*float64{&lp0, &lp1},
		TopLogprobs: []map[string]float64{
			{"Hel": -0.1, "Hi": -2},
			{"lo": -1.2},
		},
		TextOffset: []int{5, 8}, // echoLen=5
	}

	raw, err := json.Marshal(obj)
	if err != nil {
		t.Fatal(err)
	}

	var parsed map[string]any
	if err := json.Unmarshal(raw, &parsed); err != nil {
		t.Fatal(err)
	}

	if len(parsed["tokens"].([]any)) != 2 {
		t.Fatalf("tokens: %v", parsed["tokens"])
	}

	if len(parsed["token_logprobs"].([]any)) != 2 {
		t.Fatalf("token_logprobs: %v", parsed["token_logprobs"])
	}

	if len(parsed["text_offset"].([]any)) != 2 {
		t.Fatalf("text_offset: %v", parsed["text_offset"])
	}

	tops := parsed["top_logprobs"].([]any)
	if len(tops) != 2 {
		t.Fatalf("top_logprobs: %v", tops)
	}

	if _, ok := tops[0].(map[string]any)["Hel"]; !ok {
		t.Fatalf("top[0] missing Hel: %v", tops[0])
	}
}

func TestCompletionsLogprobsValidation(t *testing.T) {
	h := New(&runtime.Engine{}, "/models/demo.gguf").Handler()
	cases := []struct {
		name string
		body string
		want int
	}{
		{"logprobs too high", `{"prompt":"hi","logprobs":6}`, 400},
		{"logprobs negative", `{"prompt":"hi","logprobs":-1}`, 400},
		{"token prompt empty", `{"prompt":[]}`, 400},
		{"batch tokens", `{"prompt":[[1,2]]}`, 400},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodPost, "/v1/completions", bytes.NewBufferString(tc.body))
			req.Header.Set("Content-Type", "application/json")
			rec := httptest.NewRecorder()
			h.ServeHTTP(rec, req)
			if rec.Code != tc.want {
				t.Fatalf("status = %d, want %d, body=%s", rec.Code, tc.want, rec.Body.String())
			}
		})
	}
}

func TestCompletionsStreamChunkLogprobsJSON(t *testing.T) {
	lp := -0.5
	chunk := completionsStreamChunk{
		ID:     "cmpl-1",
		Object: "text_completion",
		Model:  "m",
		Choices: []completionsStreamChoice{{
			Index: 0,
			Text:  "x",
			Logprobs: &completionsChoiceLogprobs{
				Tokens:        []string{"x"},
				TokenLogprobs: []*float64{&lp},
				TopLogprobs:   []map[string]float64{{"x": -0.5, "y": -2}},
				TextOffset:    []int{0},
			},
		}},
	}
	raw, err := json.Marshal(chunk)
	if err != nil {
		t.Fatal(err)
	}

	var parsed map[string]any
	if err := json.Unmarshal(raw, &parsed); err != nil {
		t.Fatal(err)
	}

	c0 := parsed["choices"].([]any)[0].(map[string]any)
	lpObj := c0["logprobs"].(map[string]any)
	if lpObj["tokens"].([]any)[0] != "x" {
		t.Fatalf("tokens: %v", lpObj["tokens"])
	}

	if len(lpObj["top_logprobs"].([]any)[0].(map[string]any)) != 2 {
		t.Fatalf("top_logprobs: %v", lpObj["top_logprobs"])
	}
}

func TestFloat64PtrDistinct(t *testing.T) {
	a := float64Ptr(-0.1)
	b := float64Ptr(-1.2)
	if a == b || *a == *b {
		t.Fatalf("pointers must be distinct values: %v %v", *a, *b)
	}
}
