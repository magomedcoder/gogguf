# CLI

## `gguf info`

Краткая сводка о модели: версия GGUF, архитектура, имя, число тензоров, размер весов, длина контекста.

```bash
./build/gogguf info -m ./models/Qwen3-4B-Q8_0.gguf
```

| Флаг | Описание             |
|------|----------------------|
| `-m` | путь к файлу `.gguf` |

## `gguf inspect`

Полный дамп метаданных и списка тензоров (имя, тип, размерности, размер в байтах).

```bash
./build/gogguf inspect ./models/Qwen3-4B-Q8_0.gguf
```

Аргумент - путь к файлу, без флагов.

## `gguf run`

Генерация текста: prefill промпта -> autoregressive decode -> stdout.

```bash
./build/gogguf run -m ./models/Qwen3-4B-Q8_0.gguf --chat -p "Привет" -n 64
```

Скачать с Hugging Face (кэш `~/.cache/huggingface/hub`):

```bash
./build/gogguf run -hf Qwen/Qwen3-4B-GGUF:Q8_0 --chat -p "Привет" -n 64
```

| Флаг                | По умолчанию | Описание                                                                                                                                                                                                                 |
|---------------------|--------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `-m`                | -            | путь к `.gguf` (взаимоисключающе с `-hf`)                                                                                                                                                                                |
| `-hf` / `--hf-repo` | -            | Hugging Face `owner/repo[:quant]` (скачать + кэш)                                                                                                                                                                        |
| `-p`                | -            | текст промпта                                                                                                                                                                                                            |
| `-n`                | `128`        | максимум новых токенов                                                                                                                                                                                                   |
| `--temp`            | `0`          | температура sampling (`0` = greedy)                                                                                                                                                                                      |
| `--top-k`           | `0`          | top-k (`0` = выкл.)                                                                                                                                                                                                      |
| `--top-p`           | `1`          | nucleus (`1` = выкл.)                                                                                                                                                                                                    |
| `--min-p`           | `0`          | min-p (`0` = выкл.)                                                                                                                                                                                                      |
| `--repeat-penalty`  | `1`          | штраф за повтор (`1` = выкл.)                                                                                                                                                                                            |
| `--repeat-last-n`   | `64`         | окно истории для repeat penalty                                                                                                                                                                                          |
| `--seed`            | `0`          | seed PRNG                                                                                                                                                                                                                |
| `--chat`            | `false`      | обернуть промпт в ChatML/Qwen template                                                                                                                                                                                   |
| `--thinking`        | `false`      | режим размышления Qwen3 (только с `--chat`)                                                                                                                                                                              |
| `-i`                | `false`      | интерактивный REPL (stdin)                                                                                                                                                                                               |
| `-ngl`              | `0`          | offload N слоёв на GPU (CUDA-сборка); на macOS без Metal kernels не работает. Fused residency: `GGUF_HIDDEN_RESIDENCY`, `GGUF_QKV_RESIDENCY`, `GGUF_ATTN_FFN_RESIDENCY` (значение `0` - выкл., см. [build.md](build.md)) |
| `-b` / `--n-batch`  | `1`          | размер chunk prefill; **`>1` только Qwen3 dense** (MoE принудительно `1`; остальные arch - serial). Не true batched GPU GEMM - loop по токенам/чанками                                                                   |
| `-dev`              | пусто        | GPU ordinal(s): `1` или `0,1` (split слоёв)                                                                                                                                                                              |
| `-tensor-split`     | пусто        | пропорции слоёв по `-dev`, напр. `0.6,0.4`                                                                                                                                                                               |

Без `[:quant]` приоритет: `Q4_K_M`, затем `Q8_0`, иначе первый model `.gguf`. Gated/private: `HF_TOKEN`. Зеркало Hub: `MODEL_ENDPOINT`.

Для **Qwen3 Instruct** нужен `--chat`, иначе ответ будет некорректным.

Пример с sampling:

```bash
./build/gogguf run -m ./models/Qwen3-4B-Q8_0.gguf --chat -p "Привет" -n 64 --temp 0.7 --top-k 40 --top-p 0.9 --min-p 0.05 --repeat-penalty 1.1 --seed 42
```

С размышлением:

```bash
./build/gogguf run -m ./models/Qwen3-4B-Q8_0.gguf --chat --thinking -p "Привет" -n 64
```

С GPU offload (все 36 слоёв Qwen3-4B, CUDA-сборка):

```bash
./build/gogguf run -m ./models/Qwen3-4B-Q8_0.gguf --chat -p "Привет" -ngl 36
```

Интерактивный режим (с `--chat` история сохраняется между репликами; `/clear` - сброс):

```bash
./build/gogguf run -m ./models/Qwen3-4B-Q8_0.gguf --chat -i
```

## `gguf serve`

HTTP-сервер с OpenAI-совместимым API (см. [api.md](api.md)).

Graceful shutdown по `Ctrl+C` (SIGINT/SIGTERM).

```bash
./build/gogguf serve -m ./models/Qwen3-4B-Q8_0.gguf --host 127.0.0.1:8000
```

Или с Hugging Face:

```bash
./build/gogguf serve -hf Qwen/Qwen3-4B-GGUF:Q8_0 --host 127.0.0.1:8000
```

| Флаг                | По умолчанию     | Описание                                                     |
|---------------------|------------------|--------------------------------------------------------------|
| `-m`                | -                | путь к `.gguf` (взаимоисключающе с `-hf`)                    |
| `-hf` / `--hf-repo` | -                | Hugging Face `owner/repo[:quant]`                            |
| `--host`            | `127.0.0.1:8000` | адрес HTTP-сервера                                           |
| `-ngl`              | `0`              | GPU-слои (CUDA); env residency - см. [build.md](build.md)    |
| `-b` / `--n-batch`  | `1`              | chunk prefill (Qwen3 dense; MoE/другие arch остаются serial) |
| `-dev`              | пусто            | GPU ordinal(s): `1` или `0,1`                                |
| `-tensor-split`     | пусто            | пропорции слоёв по `-dev`                                    |
| `--api-key`         | пусто            | auth Bearer / X-API-Key                                      |
| `--rate-limit`      | `0`              | запросов в минуту на IP (`0` = без лимита)                   |

Эндпоинты и сессии (`X-Session-Id`): [HTTP API](api.md).
