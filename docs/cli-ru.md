# CLI

[English version](cli.md)

## `gguf info`

Краткая сводка о модели: версия GGUF, архитектура, имя, число тензоров, размер весов, длина контекста.

```bash
./build/gogguf info -m ./models/Qwen3-0.6B-Q8_0.gguf
```

| Флаг | Описание             |
|------|----------------------|
| `-m` | путь к файлу `.gguf` |

## `gguf inspect`

Полный дамп метаданных и списка тензоров (имя, тип, размерности, размер в байтах).

```bash
./build/gogguf inspect ./models/Qwen3-0.6B-Q8_0.gguf
```

Аргумент - путь к файлу, без флагов.

## `gguf run`

Генерация текста: prefill промпта -> autoregressive decode -> вывод в stdout.

```bash
./build/gogguf run -m ./models/Qwen3-0.6B-Q8_0.gguf --chat -p "Привет" -n 64
```

| Флаг               | По умолчанию | Описание                                        |
|--------------------|--------------|-------------------------------------------------|
| `-m`               | -            | путь к файлу `.gguf`                            |
| `-p`               | -            | текст промпта                                   |
| `-n`               | `128`        | максимум новых токенов                          |
| `--temp`           | `0`          | температура sampling (`0` = greedy)             |
| `--top-k`          | `0`          | top-k (`0` = выключено)                         |
| `--top-p`          | `1`          | nucleus sampling (`1` = выключено)              |
| `--min-p`          | `0`          | min-p sampling (`0` = выключено)                |
| `--repeat-penalty` | `1`          | штраф за повтор токенов (`1` = выключено)       |
| `--repeat-last-n`  | `64`         | окно истории для repeat penalty                 |
| `--seed`           | `0`          | seed PRNG                                       |
| `--chat`           | `false`      | обернуть промпт в ChatML/Qwen template          |
| `--thinking`       | `false`      | режим размышления Qwen3 (с `--chat`)            |
| `-i`               | `false`      | интерактивный режим (REPL)                      |
| `-ngl`             | `0`          | matmul N transformer-слоёв на GPU (CUDA-сборка) |

Для **Qwen3 Instruct** используйте `--chat`, иначе модель ответит некорректно.

Пример с sampling:

```bash
./build/gogguf run -m ./models/Qwen3-0.6B-Q8_0.gguf --chat -p "Привет" -n 64 --temp 0.7 --top-k 40 --top-p 0.9 --min-p 0.05 --repeat-penalty 1.1 --seed 42
```

С размышлением:

```bash
./build/gogguf run -m ./models/Qwen3-0.6B-Q8_0.gguf --chat --thinking -p "Привет" -n 64
```

С GPU offload (28 слоёв, CUDA-сборка):

```bash
./build/gogguf run -m ./models/Qwen3-0.6B-Q8_0.gguf --chat -p "Привет" -ngl 28
```

Интерактивный режим (с `--chat` история диалога сохраняется между репликами; `/clear` - сброс):

```bash
./build/gogguf run -m ./models/Qwen3-0.6B-Q8_0.gguf --chat -i
```

## `gguf serve`

HTTP-сервер для генерации текста по API.

Graceful shutdown по `Ctrl+C` (SIGINT/SIGTERM).

```bash
./build/gogguf serve -m ./models/Qwen3-0.6B-Q8_0.gguf --addr 127.0.0.1:8000
```

| Флаг     | По умолчанию     | Описание                                        |
|----------|------------------|-------------------------------------------------|
| `-m`     | -                | путь к файлу `.gguf`                            |
| `--addr` | `127.0.0.1:8000` | адрес HTTP-сервера                              |
| `-ngl`   | `0`              | matmul N transformer-слоёв на GPU (CUDA-сборка) |

Подробнее об эндпоинтах: [HTTP API](api-ru.md).
