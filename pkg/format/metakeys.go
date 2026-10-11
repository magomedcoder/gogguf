package format

// Common GGUF metadata keys. 
// Strings still contain "ggml" where the file format requires it.
const (
	MetaArchitecture = "general.architecture"
	MetaName         = "general.name"

	// Tokenizer wire keys (GGUF / llama.cpp naming).
	MetaTokModel  = "tokenizer.ggml.model"
	MetaTokPre    = "tokenizer.ggml.pre"
	MetaTokTokens = "tokenizer.ggml.tokens"
	MetaTokMerges = "tokenizer.ggml.merges"
	MetaTokBOS    = "tokenizer.ggml.bos_token_id"
	MetaTokEOS    = "tokenizer.ggml.eos_token_id"
)
