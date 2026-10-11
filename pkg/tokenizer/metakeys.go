package tokenizer

// GGUF metadata keys (wire names; "ggml" is part of the on-disk key, not our Go API)
const (
	MetaModel  = "tokenizer.ggml.model" // GGUF key
	MetaPre    = "tokenizer.ggml.pre"   // GGUF key
	MetaTokens = "tokenizer.ggml.tokens"
	MetaMerges = "tokenizer.ggml.merges"
	MetaScores = "tokenizer.ggml.scores"
	MetaBOS    = "tokenizer.ggml.bos_token_id"
	MetaEOS    = "tokenizer.ggml.eos_token_id"
	MetaUNK    = "tokenizer.ggml.unknown_token_id"
	MetaPad    = "tokenizer.ggml.padding_token_id"
	MetaAddBOS = "tokenizer.ggml.add_bos_token"
	MetaAddEOS = "tokenizer.ggml.add_eos_token"
)
