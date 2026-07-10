# GoGGUF

[English version](README.md)

**GoGGUF** - лёгковесный способ запуска GGUF-моделей на Go без llama.cpp.

> **Проект в активной разработке.** Поддержка GPU (CUDA) пока на раннем этапе.

Проект можно использовать как **библиотеку** (inference из Go-кода) и как **HTTP-сервер** (`gguf serve` или
пакет `server`).

**Внешних Go-зависимостей нет** - только стандартная библиотека.

Опционально: **CUDA** через Driver API (`libcuda.so`, сборка `-tags cuda`, CGO).

## Что уже работает

- парсинг GGUF v2/v3 (`info`, `inspect`), memory-map (`LoadMapped`, zero-copy `RawView`);
- деквантизация и matmul: Q8_0, Q4_0, Q4_K;
- базовые ops: RoPE, RMSNorm, GQA attention, SwiGLU;
- **SIMD** matmul FP32: AVX2 (amd64), NEON (arm64); Q8_0 dot: AVX2 (amd64);
- forward pass **Qwen3** и **Llama 3** + KV-cache;
- tokenizer BPE из метаданных GGUF;
- chat template ChatML/Qwen и Jinja (`--chat`, `--thinking`, `FormatChatUser`);
- генерация текста: `gguf run` (greedy / temperature / top-k / top-p / min-p / repeat penalty);
- HTTP-сервер: `gguf serve` (`/generate`, `/models`, `/completions`, JSON + SSE);
- **CUDA offload** (`-ngl N`): matmul первых N transformer-слоёв на GPU (сборка `-tags cuda`).

## Модели

**Работает:** Qwen3, Llama 3
**Скоро:** Llama 2, Mistral, Phi, Gemma

Подробнее: [docs/models-ru.md](docs/models-ru.md).

## Тестовая модель

На текущем этапе для разработки и тестирования используется Qwen3-0.6B-Q8_0.gguf:

```bash
mkdir -p models

curl -L -o models/Qwen3-0.6B-Q8_0.gguf https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf
```

## Быстрый старт

```bash
go build -o build/gogguf ./cmd/gogguf

./build/gogguf run -m ./models/Qwen3-0.6B-Q8_0.gguf --chat -p "Привет" -n 64
```

## Документация

| Раздел                                           | Описание                               |
|--------------------------------------------------|----------------------------------------|
| [docs/build-ru.md](docs/build-ru.md)             | Сборка: CPU, CUDA, Docker              |
| [docs/cli-ru.md](docs/cli-ru.md)                 | CLI: `info`, `inspect`, `run`, `serve` |
| [docs/api-ru.md](docs/api-ru.md)                 | HTTP API сервера                       |
| [docs/library-ru.md](docs/library-ru.md)         | Inference из Go-кода                   |
| [docs/tools-ru.md](docs/tools-ru.md)             | `debugtok`, `vocab`, `bench`           |
| [docs/GGUF-FORMAT-ru.md](docs/GGUF-FORMAT-ru.md) | Формат GGUF                            |
| [docs/models-ru.md](docs/models-ru.md)           | Поддерживаемые и планируемые модели    |
