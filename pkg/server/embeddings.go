package server

import (
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"strings"
)

type embeddingsRequest struct {
	Model          string          `json:"model"`
	Input          json.RawMessage `json:"input"`
	EncodingFormat string          `json:"encoding_format,omitempty"` // "float" (default) or "base64"
}

type embeddingsResponse struct {
	Object string          `json:"object"`
	Data   []embeddingItem `json:"data"`
	Model  string          `json:"model"`
	Usage  embeddingsUsage `json:"usage"`
}

type embeddingItem struct {
	Object    string `json:"object"`
	Embedding any    `json:"embedding"` // []float32 or base64 string
	Index     int    `json:"index"`
}

type embeddingsUsage struct {
	PromptTokens int `json:"prompt_tokens"`
	TotalTokens  int `json:"total_tokens"`
}

func (s *Server) handleEmbeddings(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeAPIError(w, http.StatusMethodNotAllowed, "method not allowed", "invalid_request_error")
		return
	}

	var req embeddingsRequest
	if err := decodeWire(r, &req); err != nil {
		writeAPIError(w, http.StatusBadRequest, err.Error(), "invalid_request_error")
		return
	}

	encFmt, err := parseEncodingFormat(req.EncodingFormat)
	if err != nil {
		writeAPIError(w, http.StatusBadRequest, err.Error(), "invalid_request_error")
		return
	}

	inputs, err := parseEmbeddingsInput(req.Input)
	if err != nil {
		writeAPIError(w, http.StatusBadRequest, err.Error(), "invalid_request_error")
		return
	}
	if len(inputs) == 0 {
		writeAPIError(w, http.StatusBadRequest, "input is empty", "invalid_request_error")
		return
	}

	s.inferMu.Lock()
	defer s.inferMu.Unlock()
	// Embed resets model KV; chat Conversations keep token history and reload KV on next infer.
	s.kvConv = nil

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
		item := embeddingItem{
			Object: "embedding",
			Index:  i,
		}

		if encFmt == "base64" {
			item.Embedding = encodeEmbeddingBase64(vec)
		} else {
			item.Embedding = vec
		}

		data = append(data, item)
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

func parseEncodingFormat(s string) (string, error) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "", "float":
		return "float", nil
	case "base64":
		return "base64", nil
	default:
		return "", fmt.Errorf("encoding_format must be \"float\" or \"base64\"")
	}
}

// encodeEmbeddingBase64 encodes float32 little-endian bytes as OpenAI-style base64 embeddings
func encodeEmbeddingBase64(vec []float32) string {
	buf := make([]byte, 4*len(vec))
	for i, v := range vec {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
	}

	return base64.StdEncoding.EncodeToString(buf)
}

type embedInput struct {
	text   string
	tokens []int
}

func parseEmbeddingsInput(raw json.RawMessage) ([]embedInput, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, fmt.Errorf("input is required")
	}

	switch raw[0] {
	case '"':
		var s string
		if err := unmarshalWire(raw, &s); err != nil {
			return nil, fmt.Errorf("input: expected a string")
		}
		return []embedInput{{text: s}}, nil
	case '[':
		// []string | []int | [][]int
		var asStrings []string
		if err := unmarshalWire(raw, &asStrings); err == nil {
			out := make([]embedInput, len(asStrings))
			for i, s := range asStrings {
				out[i] = embedInput{text: s}
			}
			return out, nil
		}

		var asInts []int
		if err := unmarshalWire(raw, &asInts); err == nil {
			return []embedInput{{tokens: asInts}}, nil
		}

		var asTokenBatches [][]int
		if err := unmarshalWire(raw, &asTokenBatches); err == nil {
			out := make([]embedInput, len(asTokenBatches))
			for i, t := range asTokenBatches {
				out[i] = embedInput{
					tokens: t,
				}
			}

			return out, nil
		}
		return nil, fmt.Errorf("input: expected string, []string, []int, or [][]int")
	default:
		return nil, fmt.Errorf("input: unsupported type")
	}
}
