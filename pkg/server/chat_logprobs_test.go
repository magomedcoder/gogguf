package server

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/magomedcoder/gogguf/pkg/runtime"
	"github.com/magomedcoder/gogguf/pkg/runtime/sampler"
)

func TestChatTokenLogprobFrom(t *testing.T) {
	decode := func(id int) string {
		switch id {
		case 1:
			return "Hello"
		case 2:
			return " world"
		default:
			return "?"
		}
	}

	e := runtime.TokenLogprob{
		TokenID: 1,
		Logprob: -0.25,
		Top: []sampler.TopToken{
			{
				ID:      1,
				Logprob: -0.25,
			},
			{
				ID:      2,
				Logprob: -1.5,
			},
		},
	}
	got := chatTokenLogprobFrom(decode, e)
	if got.Token != "Hello" || got.Logprob != -0.25 {
		t.Fatalf("token entry: %+v", got)
	}

	if string(got.Bytes) != "Hello" {
		t.Fatalf("bytes = %q", got.Bytes)
	}

	if len(got.TopLogprobs) != 2 {
		t.Fatalf("top_logprobs len = %d", len(got.TopLogprobs))
	}

	if got.TopLogprobs[1].Token != " world" || got.TopLogprobs[1].Logprob != -1.5 {
		t.Fatalf("top[1] = %+v", got.TopLogprobs[1])
	}

	empty := chatTokenLogprobFrom(decode, runtime.TokenLogprob{TokenID: 1, Logprob: -1})
	if empty.TopLogprobs == nil || len(empty.TopLogprobs) != 0 {
		t.Fatalf("empty top must be [] not null: %#v", empty.TopLogprobs)
	}
}

func TestStreamChunkLogprobsJSON(t *testing.T) {
	// wire shape for a content SSE chunk with logprobs (OpenAI chat.completion.chunk)
	chunk := chatStreamChunk{
		ID:     "chatcmpl-test",
		Object: "chat.completion.chunk",
		Model:  "demo",
		Choices: []chatStreamChoice{{
			Index: 0,
			Delta: chatStreamDelta{
				Content: "Hi",
			},
			Logprobs: &chatChoiceLogprobs{
				Content: []chatTokenLogprob{{
					Token:   "Hi",
					Logprob: -0.1,
					Bytes:   []byte("Hi"),
					TopLogprobs: []chatTopLogprob{
						{
							Token:   "Hi",
							Logprob: -0.1,
							Bytes:   []byte("Hi"),
						},
						{
							Token:   "Hey",
							Logprob: -2.0,
							Bytes:   []byte("Hey"),
						},
					},
				}},
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

	choices, ok := parsed["choices"].([]any)
	if !ok || len(choices) != 1 {
		t.Fatalf("choices: %v", parsed["choices"])
	}

	c0 := choices[0].(map[string]any)
	lp := c0["logprobs"].(map[string]any)
	content := lp["content"].([]any)
	if len(content) != 1 {
		t.Fatalf("logprobs.content len = %d", len(content))
	}

	item := content[0].(map[string]any)
	if item["token"] != "Hi" {
		t.Fatalf("token = %v", item["token"])
	}

	top := item["top_logprobs"].([]any)
	if len(top) != 2 {
		t.Fatalf("top_logprobs len = %d", len(top))
	}

	// role / finish chunks omit logprobs when nil (omitempty)
	roleOnly := chatStreamChoice{
		Index: 0,
		Delta: chatStreamDelta{
			Role: "assistant",
		},
	}
	roleRaw, err := json.Marshal(roleOnly)
	if err != nil {
		t.Fatal(err)
	}

	var roleMap map[string]any
	if err := json.Unmarshal(roleRaw, &roleMap); err != nil {
		t.Fatal(err)
	}

	if _, has := roleMap["logprobs"]; has {
		t.Fatalf("role chunk must omit logprobs: %s", roleRaw)
	}
}

func TestChatLogprobsValidationStillAppliesToStream(t *testing.T) {
	// stream requests still reject invalid top_logprobs (before generation)
	h := New(&runtime.Engine{}, "/models/demo.gguf").Handler()

	cases := []struct {
		name string
		body string
		want int
	}{
		{"top without logprobs", `{"messages":[{"role":"user","content":"hi"}],"stream":true,"top_logprobs":5}`, 400},
		{"top out of range", `{"messages":[{"role":"user","content":"hi"}],"stream":true,"logprobs":true,"top_logprobs":21}`, 400},
		{"empty messages", `{"messages":[],"stream":true,"logprobs":true}`, 400},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(tc.body))
			req.Header.Set("Content-Type", "application/json")
			rec := httptest.NewRecorder()
			h.ServeHTTP(rec, req)
			if rec.Code != tc.want {
				t.Fatalf("status = %d, want %d, body=%s", rec.Code, tc.want, rec.Body.String())
			}
		})
	}
}
