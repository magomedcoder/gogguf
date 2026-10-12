# Web UI

Опциональный веб-чат для GoGGUF - интерфейс к HTTP API (`gogguf serve`).

Исходники: [`web-ui/`](../../web-ui/) (Vue 3 + TypeScript + Vite + Tailwind CSS).

## Запуск

В одном терминале - сервер с моделью:

```bash
# CPU
./build/gogguf serve -m ./models/Qwen3-4B-Q8_0.gguf --host 127.0.0.1:8000

# GPU (сборка CUDA + -ngl > 0)
./build/gogguf serve -m ./models/Qwen3-4B-Q8_0.gguf --host 127.0.0.1:8000 -ngl 999
```

В другом - UI:

```bash
cd web-ui

yarn install # или npm install

yarn dev # или npm run dev

# `http://localhost:5173`
```

В режиме разработки запросы к API проксируются: `/api/*` -> `http://127.0.0.1:8000/*` (`web-ui/vite.config.ts`).

## Возможности

- диалог с историей сообщений
- стриминг ответа в реальном времени
- название и метаданные модели
- настройки: `max_tokens`, `temperature`, `frequency_penalty` (совместимо с OpenAI)
- остановка генерации, сброс чата

Эндпоинты: [api.md](api.md). GPU-флаги: [platforms.md](platforms.md).
