package server

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

type embeddingsRequest struct {
	Model string          `json:"model"`
	Input json.RawMessage `json:"input"`
}

type embeddingsResponse struct {
	Object string          `json:"object"`
	Data   []embeddingItem `json:"data"`
	Model  string          `json:"model"`
	Usage  embeddingsUsage `json:"usage"`
}

type embeddingItem struct {
	Object    string    `json:"object"`
	Embedding []float32 `json:"embedding"`
	Index     int       `json:"index"`
}

type embeddingsUsage struct {
	PromptTokens int `json:"prompt_tokens"`
	TotalTokens  int `json:"total_tokens"`
}

func (s *Server) handleEmbeddings(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "метод не разрешен", http.StatusMethodNotAllowed)
		return
	}

	body, err := io.ReadAll(io.LimitReader(r.Body, 8<<20))
	if err != nil {
		writeAPIError(w, http.StatusBadRequest, "не удалось прочитать тело", "invalid_request_error")
		return
	}

	var req embeddingsRequest
	if err := json.Unmarshal(body, &req); err != nil {
		writeAPIError(w, http.StatusBadRequest, "некорректный JSON", "invalid_request_error")
		return
	}

	inputs, err := parseEmbeddingsInput(req.Input)
	if err != nil {
		writeAPIError(w, http.StatusBadRequest, err.Error(), "invalid_request_error")
		return
	}
	if len(inputs) == 0 {
		writeAPIError(w, http.StatusBadRequest, "input пустой", "invalid_request_error")
		return
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	// Embed resets KV-cache; reset conversation too so chat stays consistent with cache.
	if s.conv != nil {
		s.conv.Reset()
	}

	modelName := req.Model
	if modelName == "" {
		if meta := s.engine.Metadata(); meta != nil {
			modelName, _ = meta.String("general.name")
		}

		if modelName == "" {
			modelName = "gguf"
		}
	}

	data := make([]embeddingItem, 0, len(inputs))
	totalTokens := 0
	for i, in := range inputs {
		var vec []float32
		var nTok int
		switch {
		case in.tokens != nil:
			vec, err = s.engine.EmbedTokens(in.tokens)
			nTok = len(in.tokens)
		default:
			var tokens []int
			vec, tokens, err = s.engine.EmbedText(in.text)
			nTok = len(tokens)
		}
		if err != nil {
			writeAPIError(w, http.StatusInternalServerError, err.Error(), "server_error")
			return
		}

		totalTokens += nTok
		data = append(data, embeddingItem{
			Object:    "embedding",
			Embedding: vec,
			Index:     i,
		})
	}

	writeJSON(w, embeddingsResponse{
		Object: "list",
		Data:   data,
		Model:  modelName,
		Usage: embeddingsUsage{
			PromptTokens: totalTokens,
			TotalTokens:  totalTokens,
		},
	})
}

type embedInput struct {
	text   string
	tokens []int
}

func parseEmbeddingsInput(raw json.RawMessage) ([]embedInput, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, fmt.Errorf("input обязателен")
	}

	switch raw[0] {
	case '"':
		var s string
		if err := json.Unmarshal(raw, &s); err != nil {
			return nil, fmt.Errorf("input: ожидалась строка")
		}
		return []embedInput{{text: s}}, nil
	case '[':
		// []string | []int | [][]int
		var asStrings []string
		if err := json.Unmarshal(raw, &asStrings); err == nil {
			out := make([]embedInput, len(asStrings))
			for i, s := range asStrings {
				out[i] = embedInput{text: s}
			}
			return out, nil
		}

		var asInts []int
		if err := json.Unmarshal(raw, &asInts); err == nil {
			return []embedInput{{tokens: asInts}}, nil
		}

		var asTokenBatches [][]int
		if err := json.Unmarshal(raw, &asTokenBatches); err == nil {
			out := make([]embedInput, len(asTokenBatches))
			for i, t := range asTokenBatches {
				out[i] = embedInput{
					tokens: t,
				}
			}

			return out, nil
		}
		return nil, fmt.Errorf("input: ожидается string, []string, []int или [][]int")
	default:
		return nil, fmt.Errorf("input: неподдерживаемый тип")
	}
}
