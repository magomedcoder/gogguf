# HTTP API

[Русская версия](api-ru.md)

Start the server with `gguf serve` (see [CLI](cli.md)).

## Endpoints

| Method | Path                   | Description                           |
|--------|------------------------|---------------------------------------|
| GET    | `/v1/health`           | server health check                   |
| GET    | `/v1/models`           | list loaded models                    |
| POST   | `/v1/reset`            | reset server-side KV-cache (new chat) |
| POST   | `/v1/chat/completions` | chat API (messages + stream)          |
| POST   | `/v1/embeddings`       | embeddings (not supported yet, 501)   |

## `GET /v1/models`

```json
{
  "object": "list",
  "data": [
    {
      "id": "Qwen3-0.6B",
      "object": "model"
    }
  ]
}
```

## `POST /v1/reset`

Clears the server-side KV-cache for multi-turn chat.

```bash
curl -s -X POST 127.0.0.1:8000/v1/reset
```

Response: `{"status":"ok"}`

## `POST /v1/chat/completions`

`Content-Type: application/json`

| Field                          | Type          | Default   | Description                           |
|--------------------------------|---------------|-----------|---------------------------------------|
| `messages`                     | array         | -         | `{role, content}` (required)          |
| `model`                        | string        | GGUF name | model id                              |
| `max_tokens`                   | int           | `128`     | max new tokens                        |
| `temperature`                  | float         | `0`       | `0` = greedy                          |
| `top_k`                        | int           | `0`       | top-k sampling                        |
| `top_p`                        | float         | `1`       | nucleus sampling                      |
| `min_p`                        | float         | `0`       | min-p sampling                        |
| `repeat_penalty`               | float         | `1`       | repetition penalty (`1` = off)        |
| `repeat_last_n`                | int           | `64`      | history window for repeat penalty     |
| `stop`                         | string[]      | -         | stop sequences                        |
| `stream`                       | bool          | `false`   | SSE streaming (`data: [DONE]` at end) |
| `thinking` / `enable_thinking` | bool          | `false`   | Qwen3 thinking mode                   |
| `tools`                        | array         | -         | OpenAI-style tool definitions         |
| `tool_choice`                  | string/object | `auto`    | `auto` / `none` / `required` / named  |
| `parallel_tool_calls`          | bool          | `false`   | allow multiple tool calls in one turn |

`content` may be a string or an array of `{type:"text", text:"..."}` parts.

Messages may include `tool_calls` (assistant) and `tool_call_id` / `name` (tool role).

Non-streaming response:

```json
{
  "object": "chat.completion",
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "...",
        "tool_calls": [
          {
            "id": "call_0",
            "type": "function",
            "function": {
              "name": "get_weather",
              "arguments": "{\"location\":\"Moscow\"}"
            }
          }
        ]
      },
      "finish_reason": "tool_calls"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 5,
    "total_tokens": 15
  }
}
```

If the model does not call a tool, `finish_reason` is `"stop"` and `tool_calls` is omitted.
Streaming uses SSE chunks: `data: {"choices":[{"delta":{"content":"..."}}]}` and `data: [DONE]`.

Examples:

```bash
curl -s 127.0.0.1:8000/v1/models

curl -s 127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":32}'

curl -N 127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":32,"stream":true}'
```

## `POST /v1/embeddings`

Returns HTTP `501` - generative GGUF models do not provide embeddings yet.
