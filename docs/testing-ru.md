# Тестирование и сверка с llama.cpp

[English version](testing.md)

**gogguf** сверяется с llama.cpp через golden-тесты и опциональный скрипт

## Unit- и integration-тесты

```bash
go test ./...

# с моделью Qwen3-0.6B-Q8_0.gguf
go test -tags=integration ./test/integration/...
```

Переменная `GGUF_MODEL` - путь к `.gguf`, если файл не в `models/`

### Golden fixture

`test/fixtures/qwen3_golden.json` - эталон для:

- encode промпта;
- greedy-токены после chat-prefill;
- 50 greedy decode-токенов.

Запуск:

```bash
go test -tags=integration ./test/integration/ -run Golden
go test -tags=integration ./test/integration/ -run Layers
```

## Benchmark

```bash
go test -bench=. ./pkg/ops/...

go run ./cmd/bench -m models/Qwen3-0.6B-Q8_0.gguf --chat -p "Привет" -n 128
```

## Сверка с llama.cpp

Требуется собранный `llama-cli` из llama.cpp в `PATH` или переменная `LLAMA_CLI`.

```bash
./test/compare-llama-cpp.sh models/Qwen3-0.6B-Q8_0.gguf
```

Скрипт:

1. Запускает greedy decode в llama.cpp (50 токенов, chat-промпт из fixture).
2. Запускает тот же сценарий через `go test -tags=integration`.
3. Сравнивает token id.

Если `llama-cli` не найден - скрипт завершается с кодом 0 и сообщением «пропуск».

### Ручная сверка logits

```bash
go run ./cmd/debugtok ./models/Qwen3-0.6B-Q8_0.gguf "Привет"
```

### Сверка hidden states по слоям

json-отчёт с RMS hidden state на каждом слое:

```bash
go run ./cmd/debuglayers -m ./models/Qwen3-0.6B-Q8_0.gguf -p "Привет"
go test -tags=integration ./test/integration/ -run Layers
```

Эталон: `test/fixtures/qwen3_layers.json`.

Сравните top-5 и greedy next с:

```bash
llama-cli -m model.gguf -p "Привет" --temp 0 -n 0 --log-disable 2>/dev/null
```

Для chat-промпта Qwen3 используйте `--chat` / chat template в обоих инструментах

## Допустимое расхождение

| Уровень       | Критерий                             |
|---------------|--------------------------------------|
| Tokenizer     | token id совпадают                   |
| Greedy decode | полное совпадение последовательности |
| Logits        | цель: abs diff < 1e-4 (в работе)     |
