# CLI

Примеры - Unix-shell. На Windows бинарник `build\gogguf.exe`, те же флаги в PowerShell (`.\build\gogguf.exe info -m .\models\....gguf`). macOS совпадает с Unix-примерами. GPU / ОС: [platforms.md](platforms.md).

## `gogguf info`

Краткая сводка о модели: версия GGUF, архитектура, имя, число тензоров, размер весов, длина контекста.

```bash
./build/gogguf info -m ./models/Qwen3-4B-Q8_0.gguf
```

| Флаг | Описание             |
|------|----------------------|
| `-m` | путь к файлу `.gguf` |

## `gogguf inspect`

Полный дамп метаданных и списка тензоров (имя, тип, размерности, размер в байтах).

```bash
./build/gogguf inspect ./models/Qwen3-4B-Q8_0.gguf
```

Позиционный аргумент - путь к файлу, без флагов.

## `gogguf run`

Генерация текста: prefill промпта -> autoregressive decode -> stdout.

```bash
./build/gogguf run -m ./models/Qwen3-4B-Q8_0.gguf --chat -p "Привет" -n 64
```

Скачать с Hugging Face (кэш `~/.cache/huggingface/hub`):

```bash
./build/gogguf run -hf Qwen/Qwen3-4B-GGUF:Q8_0 --chat -p "Привет" -n 64
```

| Флаг                | По умолчанию | Описание                                                                                   |
|---------------------|--------------|--------------------------------------------------------------------------------------------|
| `-m`                | -            | путь к `.gguf` (взаимоисключающе с `-hf`)                                                  |
| `-hf` / `--hf-repo` | -            | Hugging Face `owner/repo[:quant]` (скачать + кэш)                                          |
| `-p`                | -            | текст промпта                                                                              |
| `-n`                | `128`        | максимум новых токенов                                                                     |
| `--temp`            | `0`          | температура sampling (`0` = greedy)                                                        |
| `--top-k`           | `0`          | top-k (`0` = выкл.)                                                                        |
| `--top-p`           | `1`          | nucleus (`1` = выкл.)                                                                      |
| `--min-p`           | `0`          | min-p (`0` = выкл.)                                                                        |
| `--repeat-penalty`  | `1`          | штраф за повтор (`1` = выкл.)                                                              |
| `--repeat-last-n`   | `64`         | окно истории для repeat penalty                                                            |
| `--seed`            | `0`          | seed PRNG                                                                                  |
| `--chat`            | `false`      | обернуть промпт в ChatML/Qwen template                                                     |
| `--thinking`        | `false`      | режим размышления Qwen3 (с `--chat`)                                                       |
| `-i`                | `false`      | интерактивный REPL (stdin)                                                                 |
| `-ngl`              | `0`          | слои на GPU (`0` = CPU; нужна сборка `-tags cuda` + `N > 0`): [platforms.md](platforms.md) |
| `-b` / `--n-batch`  | `32`         | chunk prefill; dense batch/GEMM; MoE/MLA/SWA serial (`1`)                                  |
| `-dev`              | пусто        | устройство(а) GPU - см. [platforms.md](platforms.md)                                       |
| `-tensor-split`     | пусто        | multi-GPU split - см. [platforms.md](platforms.md)                                         |

Без `[:quant]` приоритет: `Q4_K_M`, затем `Q8_0`, иначе первый model `.gguf`.

Для **Qwen3 Instruct** нужен `--chat`, иначе ответ будет некорректным.

Пример с sampling:

```bash
./build/gogguf run -m ./models/Qwen3-4B-Q8_0.gguf --chat -p "Привет" -n 64 --temp 0.7 --top-k 40 --top-p 0.9 --min-p 0.05 --repeat-penalty 1.1 --seed 42
```

С размышлением:

```bash
./build/gogguf run -m ./models/Qwen3-4B-Q8_0.gguf --chat --thinking -p "Привет" -n 64
```

GPU (`-ngl`, `-dev`, Docker): [platforms.md](platforms.md).

Интерактивный режим (с `--chat` история сохраняется между репликами; `/clear` - сброс):

```bash
./build/gogguf run -m ./models/Qwen3-4B-Q8_0.gguf --chat -i
```

## `gogguf serve`

HTTP-сервер с OpenAI-совместимым API (см. [api.md](api.md)).

Graceful shutdown по `Ctrl+C` (SIGINT/SIGTERM).

```bash
# CPU (по умолчанию: -ngl 0)
./build/gogguf serve -m ./models/Qwen3-4B-Q8_0.gguf --host 127.0.0.1:8000

# GPU: сборка CUDA + -ngl > 0 (одна сборка offload не включает)
CGO_ENABLED=1 go build -tags cuda -o build/gogguf ./cmd/gogguf
./build/gogguf serve -m ./models/Qwen3-4B-Q8_0.gguf --host 127.0.0.1:8000 -ngl 999
```

Или с Hugging Face:

```bash
./build/gogguf serve -hf Qwen/Qwen3-4B-GGUF:Q8_0 --host 127.0.0.1:8000 -ngl 999
```

| Флаг                     | По умолчанию     | Описание                                                                             |
|--------------------------|------------------|--------------------------------------------------------------------------------------|
| `-m`                     | -                | путь к `.gguf` (взаимоисключающе с `-hf`)                                            |
| `-hf` / `--hf-repo`      | -                | Hugging Face `owner/repo[:quant]`                                                    |
| `--host`                 | `127.0.0.1:8000` | адрес HTTP-сервера                                                                   |
| `-ngl`                   | `0`              | слои на GPU (`0` = CPU); нужна `-tags cuda` + `N > 0` - [platforms.md](platforms.md) |
| `-dev` / `-tensor-split` | пусто            | multi-GPU - см. [platforms.md](platforms.md)                                         |
| `-b` / `--n-batch`       | `32`             | chunk prefill (dense batch/GEMM; MoE/MLA/SWA serial)                                 |
| `--api-key`              | пусто            | auth Bearer / X-API-Key                                                              |
| `--rate-limit`           | `0`              | запросов в минуту на IP (`0` = без лимита)                                           |

Эндпоинты: [HTTP API](api.md).

## Переменные окружения (Hugging Face)

Только для `-hf` / `pkg/hf`. Inference и residency **не** управляются через env - только `LoadOptions` ([library.md](library.md)).

| Переменная              | Описание                                                 |
|-------------------------|----------------------------------------------------------|
| `HF_TOKEN`              | Auth для gated/private репозиториев Hub.                 |
| `MODEL_ENDPOINT`        | Базовый URL Hub (по умолчанию `https://huggingface.co`). |
| `LLAMA_CACHE`           | Каталог кэша моделей (высший приоритет).                 |
| `HF_HUB_CACHE`          | Каталог кэша Hub.                                        |
| `HUGGINGFACE_HUB_CACHE` | Алиас кэша Hub.                                          |
| `HF_HOME`               | База; кэш = `$HF_HOME/hub`.                              |
| `XDG_CACHE_HOME`        | Кэш = `$XDG_CACHE_HOME/huggingface/hub`.                 |
| `LOCALAPPDATA`          | Windows fallback: `%LOCALAPPDATA%\huggingface\hub`.      |

Порядок выбора кэша: `LLAMA_CACHE` -> `HF_HUB_CACHE` -> `HUGGINGFACE_HUB_CACHE` -> `HF_HOME/hub` -> `XDG_CACHE_HOME/huggingface/hub` -> (Windows) `LOCALAPPDATA` -> `~/.cache/huggingface/hub`.

`CUDA_VISIBLE_DEVICES` - стандартный фильтр драйвера CUDA (не специфика GoGGUF); влияет на GPU при `-ngl > 0` - см. [platforms.md](platforms.md).
