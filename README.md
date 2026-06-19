# gguf.go - запуск ML-моделей в формате **GGUF** на чистом **Go**.

> **Ранний этап разработки.**

**gguf.go** - лёгковесный способ запуска GGUF-моделей на языке Go без llama.cpp.

Формат **GGUF** используется в экосистеме llama.cpp.

[GGUF-FORMAT.md](GGUF-FORMAT.md) - описание формата GGUF

Проект можно использовать как **библиотеку** (inference из Go-кода) и как **HTTP-сервер** (`gguf serve` или пакет `server`).

**Внешних Go-зависимостей нет** - только стандартная библиотека.

Опционально: **CUDA** через Driver API (`libcuda.so`, сборка `-tags cuda`, CGO).

## Что уже работает

- парсинг GGUF v2/v3 (`info`, `inspect`), memory-map (`LoadMapped`, zero-copy `RawView`);
- деквантизация и matmul: Q8_0, Q4_0, Q4_K;
- базовые ops: RoPE, RMSNorm, GQA attention, SwiGLU;
- forward pass **Qwen3** + KV-cache;
- tokenizer BPE из метаданных GGUF;
- chat template ChatML/Qwen (`--chat`, `--thinking`, `FormatChatUser`);
- генерация текста: `gguf run` (prefill + greedy / temperature / top-k / top-p);
- HTTP-сервер: `gguf serve` (`/generate`, `/models`, JSON + SSE streaming);
- **CUDA offload** (`-ngl N`): matmul первых N transformer-слоёв на GPU (сборка `-tags cuda`).

---

На текущем этапе для разработки и тестирования используется Qwen3-0.6B-Q8_0.gguf

```bash
mkdir -p models

curl -L -o models/Qwen3-0.6B-Q8_0.gguf https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf
```

---

## Сборка

### Локально (Go 1.26)

```bash
go build -o build/gguf ./cmd/gguf
```

Без CGO - кросс-компиляция на любую платформу

### CUDA (NVIDIA GPU, опционально)

Требуется драйвер NVIDIA (`libcuda.so`) и CGO.

CUDA Toolkit не нужен - используется Driver API через `dlopen`.

```bash
CGO_ENABLED=1 go build -tags cuda -o build/gguf ./cmd/gguf
```

Проверка GPU matmul:

```bash
CGO_ENABLED=1 go test -tags=cuda ./pkg/gpu/cuda/...
```

`-ngl N` - matmul первых N transformer-слоёв на GPU (макс. `block_count` из `gguf inspect`; Qwen3-0.6B - 28).

Сейчас на GPU только matmul: Q8_0 без деквантизации в FP32, остальные типы — через FP32. Attention, norm и RoPE — на CPU. Q8_0 kernel требует GPU sm_70+.

Без `-tags cuda` при `-ngl > 0` будет ошибка `gpu: CUDA недоступна`.

### Через Docker

Multi-stage `Dockerfile`: CPU-кросс-компиляция по умолчанию, отдельные target'ы для CUDA.

**CPU (по умолчанию)** - все платформы, `CGO_ENABLED=0`:

```bash
docker build -t gguf-build .
docker run --rm -v "$(pwd)/build:/out" gguf-build
```

**CUDA** - только `linux-amd64/gguf-cuda`:

```bash
docker build --target cuda -t gguf-cuda .
docker run --rm -v "$(pwd)/build:/out" gguf-cuda
```

**CPU + CUDA**:

```bash
docker build --target release -t gguf-release .
docker run --rm -v "$(pwd)/build:/out" gguf-release
```

| Target / режим | Результат                     |
|----------------|-------------------------------|
| *(default)*    | CPU-бинарники всех платформ   |
| `cuda`         | `build/linux-amd64/gguf-cuda` |
| `release`      | CPU + `gguf-cuda`             |

| Платформа     | CPU-бинарник                   | CUDA-бинарник (target `cuda` / `release`) |
|---------------|--------------------------------|-------------------------------------------|
| Linux amd64   | `build/linux-amd64/gguf`       | `build/linux-amd64/gguf-cuda`             |
| Linux arm64   | `build/linux-arm64/gguf`       | -                                         |
| Windows amd64 | `build/windows-amd64/gguf.exe` | -                                         |
| Windows arm64 | `build/windows-arm64/gguf.exe` | -                                         |
| macOS amd64   | `build/darwin-amd64/gguf`      | -                                         |
| macOS arm64   | `build/darwin-arm64/gguf`      | -                                         |


> **Примечание.** Путь к бинарнику зависит от способа сборки:
> - локально CPU: `./build/gguf`
> - локально CUDA: `./build/gguf` (с `-tags cuda`)
> - Docker CPU: `./build/<os>-<arch>/gguf` (на Windows: `gguf.exe`)
> - Docker CUDA: `./build/linux-amd64/gguf-cuda`

---

## CLI

### `gguf info`

Краткая сводка о модели: версия GGUF, архитектура, имя, число тензоров, размер весов, длина контекста.

```bash
./build/gguf info -m ./models/Qwen3-0.6B-Q8_0.gguf
```

| Флаг | Описание             |
|------|----------------------|
| `-m` | путь к файлу `.gguf` |

### `gguf inspect`

Полный дамп метаданных и списка тензоров (имя, тип, размерности, размер в байтах).

```bash
./build/gguf inspect ./models/Qwen3-0.6B-Q8_0.gguf
```

Аргумент - путь к файлу, без флагов.

### `gguf run`

Генерация текста: prefill промпта -> autoregressive decode -> вывод в stdout.

```bash
./build/gguf run -m ./models/Qwen3-0.6B-Q8_0.gguf --chat -p "Привет" -n 64
```

| Флаг         | По умолчанию | Описание                                        |
|--------------|--------------|-------------------------------------------------|
| `-m`         | -            | путь к файлу `.gguf`                            |
| `-p`         | -            | текст промпта                                   |
| `-n`         | `128`        | максимум новых токенов                          |
| `--temp`     | `0`          | температура sampling (`0` = greedy)             |
| `--top-k`    | `0`          | top-k (`0` = выключено)                         |
| `--top-p`    | `1`          | nucleus sampling (`1` = выключено)              |
| `--seed`     | `0`          | seed PRNG                                       |
| `--chat`     | `false`      | обернуть промпт в ChatML/Qwen template          |
| `--thinking` | `false`      | режим размышления Qwen3 (с `--chat`)            |
| `-ngl`       | `0`          | matmul N transformer-слоёв на GPU (CUDA-сборка) |

Для **Qwen3 Instruct** используйте `--chat`, иначе модель ответит некорректно.

Пример с sampling:

```bash
./build/gguf run -m ./models/Qwen3-0.6B-Q8_0.gguf --chat -p "Привет" -n 64 --temp 0.7 --top-k 40 --top-p 0.9 --seed 42
```

С размышлением:

```bash
./build/gguf run -m ./models/Qwen3-0.6B-Q8_0.gguf --chat --thinking -p "Привет" -n 64
```

С GPU offload (28 слоёв, CUDA-сборка):

```bash
./build/gguf run -m ./models/Qwen3-0.6B-Q8_0.gguf --chat -p "Привет" -ngl 28
```

### `gguf serve`

HTTP-сервер для генерации текста по API.

Graceful shutdown по `Ctrl+C` (SIGINT/SIGTERM).

```bash
./build/gguf serve -m ./models/Qwen3-0.6B-Q8_0.gguf --addr 127.0.0.1:8000
```

| Флаг     | По умолчанию     | Описание                                        |
|----------|------------------|-------------------------------------------------|
| `-m`     | -                | путь к файлу `.gguf`                            |
| `--addr` | `127.0.0.1:8000` | адрес HTTP-сервера                              |
| `-ngl`   | `0`              | matmul N transformer-слоёв на GPU (CUDA-сборка) |

#### API

| Метод | Путь        | Описание                        |
|-------|-------------|---------------------------------|
| GET   | `/models`   | метаданные загруженной модели   |
| POST  | `/generate` | генерация текста (JSON или SSE) |

Тело `POST /generate` (`Content-Type: application/json`):

| Поле          | Тип    | По умолчанию | Описание                    |
|---------------|--------|--------------|-----------------------------|
| `prompt`      | string | -            | текст запроса (обязательно) |
| `max_tokens`  | int    | `128`        | максимум новых токенов      |
| `temperature` | float  | `0`          | `0` = greedy                |
| `top_k`       | int    | `0`          | top-k sampling              |
| `top_p`       | float  | `1`          | nucleus sampling            |
| `seed`        | uint   | `0`          | seed PRNG                   |
| `chat`        | bool   | `false`      | ChatML/Qwen template        |
| `stream`      | bool   | `false`      | SSE streaming               |
| `system`      | string | -            | system prompt (с `chat`)    |
| `thinking`    | bool   | `false`      | режим размышления Qwen3     |

Ответ без stream:

```json
{
  "text": "...",
  "tokens": 32
}
```

Streaming (SSE) - события `data: {"token":"..."}` и в конце `data: {"done":true}`.

Примеры:

```bash
curl -s localhost:8000/generate -H 'Content-Type: application/json' -d '{"prompt":"Привет","chat":true, "max_tokens":32}'

curl -N localhost:8000/generate -H 'Content-Type: application/json' -d '{"prompt":"Привет","chat":true,"stream":true,"max_tokens":32}'

curl -s localhost:8000/models
```

---

## Утилиты для отладки

### `debugtok`

Проверяет encode промпта и logits после prefill: top-5 токенов и greedy-следующий.

```bash
go run ./cmd/debugtok ./models/Qwen3-0.6B-Q8_0.gguf "Hello"
```

### `vocab`

Показывает конфиг Qwen3 (`head_dim`, число heads) и ID special tokens в словаре.

```bash
go run ./cmd/vocab ./models/Qwen3-0.6B-Q8_0.gguf
```

---

## Использование как библиотеки

### Inference

```go
import "github.com/magomedcoder/gguf.go"

engine, err := gguf.Load("./models/Qwen3-0.6B-Q8_0.gguf", gguf.LoadOptions{
  NGL: 0, // matmul N слоёв на GPU; <= block_count, нужна CUDA-сборка
})
ctx, err := engine.NewContext()

prompt, err := gguf.FormatChatUser("Привет", gguf.ChatOptions{
  Metadata: engine.Metadata(),
})
text, err := ctx.Generate(prompt, gguf.GenerateParams{
  MaxTokens: 128,
  Sampler:   gguf.Greedy,
})
```

### Sampling с temperature / top-k / top-p

```go
sampler := gguf.NewSampler(gguf.SamplerConfig{
  Temp: 0.7,
  TopK: 40,
  TopP: 0.9,
  Seed: 42,
})
text, err := ctx.Generate(prompt, gguf.GenerateParams{
  MaxTokens: 64,
  Sampler:   sampler,
})
```

### Загрузка через mmap (zero-copy веса)

```go
engine, err := gguf.LoadMapped("./models/Qwen3-0.6B-Q8_0.gguf", gguf.LoadOptions{NGL: 0})
```

### GPU offload из кода

```go
engine, err := gguf.Load("./models/Qwen3-0.6B-Q8_0.gguf", gguf.LoadOptions{
  NGL: 28, // нужна сборка -tags cuda
})
```

### Парсинг GGUF без inference

```go
import "github.com/magomedcoder/gguf.go"

r, err := gguf.OpenFile("./models/Qwen3-0.6B-Q8_0.gguf")

arch, _ := r.Metadata.String("general.architecture")
```
