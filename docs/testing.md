# Testing and llama.cpp comparison

[Русская версия](testing-ru.md)

**gguf.go** is validated against llama.cpp using golden tests and an optional script.

## Unit and integration tests

```bash
go test ./...

# with Qwen3-0.6B-Q8_0.gguf
go test -tags=integration ./test/integration/...
```

Set `GGUF_MODEL` to the `.gguf` path if the file is not in `models/`.

### Golden fixture

`test/fixtures/qwen3_golden.json` - reference for:

- prompt encoding;
- greedy tokens after chat prefill;
- 50 greedy decode tokens.

Run:

```bash
go test -tags=integration ./test/integration/ -run Golden
```

## Benchmark

```bash
go test -bench=. ./pkg/ops/...

go run ./cmd/bench -m models/Qwen3-0.6B-Q8_0.gguf --chat -p "Hello" -n 128
```

## llama.cpp comparison

Requires `llama-cli` from llama.cpp in `PATH`, or set `LLAMA_CLI`.

```bash
./test/compare-llama-cpp.sh models/Qwen3-0.6B-Q8_0.gguf
```

The script:

1. Runs golden integration tests (reference from llama.cpp in the fixture).
2. Optionally runs an llama.cpp smoke test if `llama-cli` is available.

If `llama-cli` is not found - exits with code 0 and a skip message.

### Manual logit comparison

```bash
go run ./cmd/debugtok ./models/Qwen3-0.6B-Q8_0.gguf "Hello"
```

Compare top-5 and greedy next with:

```bash
llama-cli -m model.gguf -p "Hello" --temp 0 -n 0 --log-disable 2>/dev/null
```

For Qwen3 chat prompts use `--chat` / chat template in both tools.

## Acceptable drift

| Level         | Criterion                             |
|---------------|---------------------------------------|
| Tokenizer     | token IDs match                       |
| Greedy decode | full sequence match                   |
| Logits        | target: abs diff < 1e-4 (in progress) |
