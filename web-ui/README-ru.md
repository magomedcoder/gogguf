# web-ui для GoGGUF

[English version](README.md)

Веб-чат для [GoGGUF](../) - интерфейс к HTTP API `gguf serve`.

Vue 3 + TypeScript + Vite + Tailwind CSS.

## Запуск

В одном терминале - сервер с моделью:

```bash
./build/gogguf serve -m ./models/Qwen3-0.6B-Q8_0.gguf --addr 127.0.0.1:8000
```

В другом - UI:

```bash
cd web-ui

yarn install # или npm install

yarn dev  # или npm run dev

# `http://localhost:5173`
```

В режиме разработки запросы к API проксируются: `/api/*` -> `http://127.0.0.1:8000/*` (`vite.config.ts`).

## Возможности

- диалог с историей сообщений
- стриминг ответа в реальном времени
- название и метаданные модели
- настройки: `max_tokens`, `temperature`, `thinking`, `repeat_penalty`, `min_p`
- остановка генерации, сброс чата

Подробнее об эндпоинтах: [docs/api-ru.md](../docs/api-ru.md).
