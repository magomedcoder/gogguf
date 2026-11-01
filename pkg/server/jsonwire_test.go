package server

import (
	"encoding/json"
	"testing"
)

func TestMarshalWireMatchesV1(t *testing.T) {
	chunk := chatStreamChunk{
		ID:      "chatcmpl_test",
		Object:  "chat.completion.chunk",
		Created: 1,
		Model:   "demo",
		Choices: []chatStreamChoice{{
			Index: 0,
			Delta: chatStreamDelta{Role: "assistant", Content: "hi"},
		}},
	}

	want, err := json.Marshal(chunk)
	if err != nil {
		t.Fatal(err)
	}

	got, err := marshalWire(chunk)
	if err != nil {
		t.Fatal(err)
	}

	if string(got) != string(want) {
		t.Fatalf("wire mismatch:\nv1=%s\nv2=%s", want, got)
	}

	var round chatStreamChunk
	if err := unmarshalWire(got, &round); err != nil {
		t.Fatal(err)
	}

	if round.ID != chunk.ID || round.Choices[0].Delta.Content != "hi" {
		t.Fatalf("round-trip: %+v", round)
	}
}

func TestUnmarshalWireChatCompletionRequest(t *testing.T) {
	const raw = `{"model":"m","messages":[{"role":"user","content":"hi"}],"stream":false,"n":0,"max_tokens":0}`
	var req chatCompletionRequest
	if err := unmarshalWire([]byte(raw), &req); err != nil {
		t.Fatal(err)
	}

	if req.Model != "m" || req.Stream != false || req.N != 0 || req.MaxTokens != 0 {
		t.Fatalf("got %+v", req)
	}

	if len(req.Messages) != 1 || string(req.Messages[0].Content) != `"hi"` {
		t.Fatalf("messages: %+v", req.Messages)
	}

	got, err := marshalWire(chatCompletionRequest{
		Model: "m",
		Messages: []chatMessage{
			{
				Role:    "user",
				Content: json.RawMessage(`"x"`),
			},
		},
		Stream:    false,
		N:         0,
		MaxTokens: 0,
	})
	if err != nil {
		t.Fatal(err)
	}

	var back chatCompletionRequest
	if err := unmarshalWire(got, &back); err != nil {
		t.Fatal(err)
	}

	if back.Stream != false || back.N != 0 {
		t.Fatalf("round-trip zeros: %+v", back)
	}

	par := true
	got2, err := marshalWire(chatCompletionRequest{ParallelToolCalls: &par})
	if err != nil {
		t.Fatal(err)
	}

	var back2 chatCompletionRequest
	if err := unmarshalWire(got2, &back2); err != nil {
		t.Fatal(err)
	}

	if back2.ParallelToolCalls == nil || !*back2.ParallelToolCalls {
		t.Fatalf("parallel_tool_calls: %+v", back2.ParallelToolCalls)
	}
}

func TestUnmarshalWireCompletionsRequest(t *testing.T) {
	const raw = `{"model":"m","prompt":"p","echo":true,"stream":false,"max_tokens":16}`
	var req completionsRequest
	if err := unmarshalWire([]byte(raw), &req); err != nil {
		t.Fatal(err)
	}

	if !req.Echo || req.Stream || req.MaxTokens != 16 {
		t.Fatalf("got %+v", req)
	}

	got, err := marshalWire(completionsResponse{
		ID:     "cmpl-1",
		Object: "text_completion",
		Model:  "m",
		Choices: []completionsChoice{{
			Index:        0,
			Text:         "hi",
			FinishReason: "stop",
		}},
		Usage: usageTokens{
			PromptTokens:     1,
			CompletionTokens: 1,
			TotalTokens:      2,
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	want, err := json.Marshal(completionsResponse{
		ID:     "cmpl-1",
		Object: "text_completion",
		Model:  "m",
		Choices: []completionsChoice{{
			Index:        0,
			Text:         "hi",
			FinishReason: "stop",
		}},
		Usage: usageTokens{
			PromptTokens:     1,
			CompletionTokens: 1,
			TotalTokens:      2,
		},
	})

	if err != nil {
		t.Fatal(err)
	}

	if string(got) != string(want) {
		t.Fatalf("completions response wire mismatch:\nv1=%s\nv2=%s", want, got)
	}
}

func TestUsageJSONFieldNames(t *testing.T) {
	usage := usageTokens{
		PromptTokens:     3,
		CompletionTokens: 2,
		TotalTokens:      5,
	}

	chatBody, err := json.Marshal(chatCompletionResponse{
		ID:      "chatcmpl-1",
		Object:  "chat.completion",
		Model:   "m",
		Created: 1,
		Choices: []chatCompletionChoice{
			{
				Index:        0,
				FinishReason: "stop",
			},
		},
		Usage: usage,
	})
	if err != nil {
		t.Fatal(err)
	}
	cmplBody, err := json.Marshal(completionsResponse{
		ID:      "cmpl-1",
		Object:  "text_completion",
		Model:   "m",
		Created: 1,
		Choices: []completionsChoice{
			{
				Index:        0,
				Text:         "x",
				FinishReason: "stop",
			},
		},
		Usage: usage,
	})
	if err != nil {
		t.Fatal(err)
	}

	for _, raw := range []json.RawMessage{chatBody, cmplBody} {
		var m map[string]json.RawMessage
		if err := json.Unmarshal(raw, &m); err != nil {
			t.Fatal(err)
		}

		u, ok := m["usage"]
		if !ok {
			t.Fatalf("missing usage in %s", raw)
		}

		var umap map[string]int
		if err := json.Unmarshal(u, &umap); err != nil {
			t.Fatal(err)
		}

		for _, key := range []string{"prompt_tokens", "completion_tokens", "total_tokens"} {
			if _, ok := umap[key]; !ok {
				t.Fatalf("usage missing %q in %s", key, raw)
			}
		}
	}
}

func TestUnmarshalWireEmbeddingsRequest(t *testing.T) {
	const raw = `{"model":"m","input":"hello","encoding_format":"float"}`
	var req embeddingsRequest
	if err := unmarshalWire([]byte(raw), &req); err != nil {
		t.Fatal(err)
	}

	if req.Model != "m" || string(req.Input) != `"hello"` {
		t.Fatalf("got %+v", req)
	}
}
