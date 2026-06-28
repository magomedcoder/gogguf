# HTTP API

[Русская версия](api-ru.md)

Start the server with `gguf serve` (see [CLI](cli.md)).

## Endpoints

| Method | Path           | Description                   |
|--------|----------------|-------------------------------|
| GET    | `/models`      | loaded model metadata         |
| POST   | `/generate`    | text generation (JSON or SSE) |
| POST   | `/completions` | chat API (messages + stream)  |

## `POST /generate`

`Content-Type: application/json`

| Field            | Type   | Default | Description                    |
|------------------|--------|---------|--------------------------------|
| `prompt`         | string | -       | request text (required)        |
| `max_tokens`     | int    | `128`   | max new tokens                 |
| `temperature`    | float  | `0`     | `0` = greedy                   |
| `top_k`          | int    | `0`     | top-k sampling                 |
| `top_p`          | float  | `1`     | nucleus sampling               |
| `min_p`          | float  | `0`     | min-p sampling                 |
| `repeat_penalty` | float  | `1`     | repetition penalty (`1` = off) |
| `repeat_last_n`  | int    | `0`     | history window (`0` = 64)      |
| `seed`           | uint   | `0`     | PRNG seed                      |
| `chat`           | bool   | `false` | ChatML/Qwen template           |
| `stream`         | bool   | `false` | SSE streaming                  |
| `system`         | string | -       | system prompt (with `chat`)    |
| `thinking`       | bool   | `false` | Qwen3 thinking mode            |

Non-streaming response:

```json
{
  "text": "...",
  "tokens": 32
}
```

Streaming (SSE) - events `data: {"token":"..."}` and finally `data: {"done":true}`.

Examples:

```bash
curl -s 127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Hello","chat":true,"max_tokens":32}'

curl -N 127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Hello","chat":true,"stream":true,"max_tokens":32}'

curl -s 127.0.0.1:8000/models
```

## `POST /completions`

Chat API with a `messages` array.

```bash
curl -s 127.0.0.1:8000/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":32}'
```

Example response:

```json
{
  "object": "chat.completion",
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "..."
      }
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 5,
    "total_tokens": 15
  }
}
```

Supports `temperature`, `top_p`, `min_p`, `repeat_penalty`, `repeat_last_n`, `stream`, `thinking`.
