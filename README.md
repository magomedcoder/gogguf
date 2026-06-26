# gguf.go

**gguf.go** - лёгковесный способ запуска GGUF-моделей на Go без llama.cpp.

> **Ранний этап разработки.**

Проект можно использовать как **библиотеку** (inference из Go-кода) и как **HTTP-сервер** (`gguf serve` или
пакет `server`).

**Внешних Go-зависимостей нет** - только стандартная библиотека.

Опционально: **CUDA** через Driver API (`libcuda.so`, сборка `-tags cuda`, CGO).

## Что уже работает

- парсинг GGUF v2/v3 (`info`, `inspect`), memory-map (`LoadMapped`, zero-copy `RawView`);
- деквантизация и matmul: Q8_0, Q4_0, Q4_K;
- базовые ops: RoPE, RMSNorm, GQA attention, SwiGLU;
- **SIMD** matmul FP32: AVX2 (amd64), NEON (arm64);
- forward pass **Qwen3** + KV-cache;
- tokenizer BPE из метаданных GGUF;
- chat template ChatML/Qwen и Jinja (`--chat`, `--thinking`, `FormatChatUser`);
- генерация текста: `gguf run` (greedy / temperature / top-k / top-p / min-p / repeat penalty);
- HTTP-сервер: `gguf serve` (`/generate`, `/models`, `/completions`, JSON + SSE);
- **CUDA offload** (`-ngl N`): matmul первых N transformer-слоёв на GPU (сборка `-tags cuda`).

## Тестовая модель

На текущем этапе для разработки и тестирования используется Qwen3-0.6B-Q8_0.gguf:

```bash
mkdir -p models

curl -L -o models/Qwen3-0.6B-Q8_0.gguf https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf
```

## Быстрый старт

```bash
go build -o build/gguf ./cmd/gguf

./build/gguf run -m ./models/Qwen3-0.6B-Q8_0.gguf --chat -p "Привет" -n 64
```

## Документация

| Раздел                                     | Описание                               |
|--------------------------------------------|----------------------------------------|
| [docs/build.md](docs/build.md)             | Сборка: CPU, CUDA, Docker              |
| [docs/cli.md](docs/cli.md)                 | CLI: `info`, `inspect`, `run`, `serve` |
| [docs/api.md](docs/api.md)                 | HTTP API сервера                       |
| [docs/library.md](docs/library.md)         | Inference из Go-кода                   |
| [docs/tools.md](docs/tools.md)             | `debugtok`, `vocab`, `bench`           |
| [docs/testing.md](docs/testing.md)         | Golden-тесты, сверка с llama.cpp       |
| [docs/GGUF-FORMAT.md](docs/GGUF-FORMAT.md) | Формат GGUF                            |
