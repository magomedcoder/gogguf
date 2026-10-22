package server

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"
)

const requestIDHeader = "X-Request-Id"

func newRequestID() string {
	var b [8]byte
	if _, err := rand.Read(b[:]); err != nil {
		return fmt.Sprintf("req_%x", time.Now().UnixNano())
	}

	return "req_" + hex.EncodeToString(b[:])
}

func withRequestID(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		id := strings.TrimSpace(r.Header.Get(requestIDHeader))
		if id == "" {
			id = newRequestID()
		}

		w.Header().Set(requestIDHeader, id)

		next.ServeHTTP(w, r)
	})
}

func systemFingerprint(modelID string) string {
	modelID = strings.TrimSpace(modelID)
	if modelID == "" {
		modelID = "model"
	}

	sum := uint32(2166136261)
	for i := 0; i < len(modelID); i++ {
		sum ^= uint32(modelID[i])
		sum *= 16777619
	}

	return fmt.Sprintf("fp_gogguf_%08x", sum)
}

// parseLogitBias decodes OpenAI logit_bias: {"token_id": bias, ...}
func parseLogitBias(raw json.RawMessage) (map[int]float32, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil
	}

	var asMap map[string]float64
	if err := json.Unmarshal(raw, &asMap); err != nil {
		return nil, fmt.Errorf("logit_bias must be an object of token_id -> float: %w", err)
	}

	out := make(map[int]float32, len(asMap))
	for k, v := range asMap {
		id, err := strconv.Atoi(k)
		if err != nil {
			return nil, fmt.Errorf("logit_bias key %q is not a token id", k)
		}

		out[id] = float32(v)
	}

	return out, nil
}

type responseFormat struct {
	Type string `json:"type"`
}

// parseResponseFormat returns "text", "json_object", or error for unsupported types
func parseResponseFormat(raw json.RawMessage) (string, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return "text", nil
	}

	var rf responseFormat
	if err := json.Unmarshal(raw, &rf); err != nil {
		return "", fmt.Errorf("response_format: %w", err)
	}

	switch strings.ToLower(strings.TrimSpace(rf.Type)) {
	case "", "text":
		return "text", nil
	case "json_object":
		return "json_object", nil
	case "json_schema":
		return "", fmt.Errorf("response_format.type=json_schema is not supported; use json_object")
	default:
		return "", fmt.Errorf("response_format.type %q is not supported", rf.Type)
	}
}

func ensureJSONObjectOutput(text string) error {
	s := strings.TrimSpace(text)
	if s == "" {
		return fmt.Errorf("response_format=json_object: empty completion")
	}

	if strings.HasPrefix(s, "```") {
		s = strings.TrimPrefix(s, "```json")
		s = strings.TrimPrefix(s, "```JSON")
		s = strings.TrimPrefix(s, "```")
		if i := strings.LastIndex(s, "```"); i >= 0 {
			s = s[:i]
		}

		s = strings.TrimSpace(s)
	}

	var v any
	if err := json.Unmarshal([]byte(s), &v); err != nil {
		return fmt.Errorf("response_format=json_object: completion is not valid JSON")
	}

	switch v.(type) {
	case map[string]any, []any:
		return nil
	default:
		return fmt.Errorf("response_format=json_object: completion must be a JSON object or array")
	}
}

func presencePenaltyValue(p *float64) float32 {
	if p == nil {
		return 0
	}

	v := float32(*p)
	if v < -2 {
		return -2
	}

	if v > 2 {
		return 2
	}

	return v
}

// contentHasMultimodal reports image/audio/etc parts that we reject with 400
func contentHasMultimodal(raw json.RawMessage) bool {
	if len(raw) == 0 || raw[0] == '"' {
		return false
	}

	var parts []struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(raw, &parts); err != nil {
		return false
	}

	for _, p := range parts {
		t := strings.ToLower(strings.TrimSpace(p.Type))
		if t != "" && t != "text" {
			return true
		}
	}

	return false
}
