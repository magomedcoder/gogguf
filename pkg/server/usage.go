package server

type usageTokens struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type streamMeta struct {
	ID                string `json:"id"`
	Object            string `json:"object"`
	Model             string `json:"model"`
	Created           int64  `json:"created"`
	SystemFingerprint string `json:"system_fingerprint,omitempty"`
}
