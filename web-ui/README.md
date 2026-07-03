# web-ui for gguf.go

[Русская версия](README-ru.md)

Web chat for [gguf.go](../) - interface to the HTTP API `gguf serve`.

Vue 3 + TypeScript + Vite + Tailwind CSS.

## Running

In one terminal - server with the model:

```bash
./build/gguf serve -m ./models/Qwen3-0.6B-Q8_0.gguf --addr 127.0.0.1:8000
```

In another - the UI:

```bash
cd web-ui

yarn install # or npm install

yarn dev  # or npm run dev

# `http://localhost:5173`
```

In development mode, API requests are proxied: `/api/*` -> `http://127.0.0.1:8000/*` (`vite.config.ts`).

## Features

- chat with message history
- real-time response streaming
- model name and metadata
- settings: `max_tokens`, `temperature`, `thinking`
- stop generation, reset chat

More on endpoints: [docs/api.md](../docs/api.md).
