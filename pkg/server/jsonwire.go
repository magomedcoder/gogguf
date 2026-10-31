package server

import (
	"fmt"
	"io"
	"net/http"

	jsonv1 "encoding/json"
	jsonv2 "encoding/json/v2"
)

const maxRequestBody = 8 << 20

// marshalWire encodes with json/v2 under v1 semantics (OpenAI wire compatibility)
func marshalWire(v any) ([]byte, error) {
	return jsonv2.Marshal(v, jsonv1.DefaultOptionsV1())
}

// unmarshalWire decodes with json/v2 under v1 semantics.
func unmarshalWire(data []byte, v any) error {
	return jsonv2.Unmarshal(data, v, jsonv1.DefaultOptionsV1())
}

func decodeWire(r *http.Request, v any) error {
	body, err := io.ReadAll(io.LimitReader(r.Body, maxRequestBody))
	if err != nil {
		return err
	}

	if len(body) == 0 {
		return fmt.Errorf("empty request body")
	}

	return unmarshalWire(body, v)
}

// writeSSEJSON emits one SSE data line.
// jsontext.Encoder was evaluated for streaming without per-chunk []byte; marshalWire kept for v1 wire parity.
func writeSSEJSON(w io.Writer, v any) error {
	data, err := marshalWire(v)
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(w, "data: %s\n\n", data)

	return err
}
