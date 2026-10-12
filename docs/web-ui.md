# Web UI

Optional web chat for GoGGUF - UI for the HTTP API (`gogguf serve`).

Source: [`web-ui/`](../web-ui/) (Vue 3 + TypeScript + Vite + Tailwind CSS).

## Running

In one terminal - server with the model:

```bash
# CPU
./build/gogguf serve -m ./models/Qwen3-4B-Q8_0.gguf --host 127.0.0.1:8000

# GPU (CUDA build + -ngl > 0)
./build/gogguf serve -m ./models/Qwen3-4B-Q8_0.gguf --host 127.0.0.1:8000 -ngl 999
```

In another - the UI:

```bash
cd web-ui

yarn install # or npm install

yarn dev # or npm run dev

# `http://localhost:5173`
```

In development mode, API requests are proxied: `/api/*` -> `http://127.0.0.1:8000/*` (`web-ui/vite.config.ts`).

## Features

- chat with message history
- real-time response streaming
- model name and metadata
- settings: `max_tokens`, `temperature`, `frequency_penalty` (OpenAI-compatible)
- stop generation, reset chat

More on endpoints: [api.md](api.md). GPU flags: [platforms.md](platforms.md).
