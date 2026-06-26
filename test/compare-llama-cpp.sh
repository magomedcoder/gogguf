#!/usr/bin/env bash
# Сверка gguf.go с эталоном llama.cpp (golden fixture)
# Использование ./test/compare-llama-cpp.sh [models/модель.gguf]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODEL="${1:-models/Qwen3-0.6B-Q8_0.gguf}"

if [[ ! -f "$MODEL" ]]; then
	echo "модель не найдена: $MODEL" >&2
	echo "скачайте: curl -L -o models/Qwen3-0.6B-Q8_0.gguf https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf" >&2
	exit 1
fi

export GGUF_MODEL="$(realpath "$MODEL")"

echo "gguf.go: golden-тесты (эталон llama.cpp в test/fixtures/qwen3_golden.json)"

go test -tags=integration ./test/integration/ -run Golden -count=1 -v

echo ""
echo "OK: greedy-токены совпадают с fixture"

LLAMA_CLI="${LLAMA_CLI:-llama-cli}"

if command -v "$LLAMA_CLI" &>/dev/null; then
	echo ""
	echo "llama.cpp smoke test (текст, не token id)"
	# Chat-промпт как в golden case chat_count_greedy_50
	PROMPT='Count from 1 to 20, one number per line'
	"$LLAMA_CLI" -m "$MODEL" \
		--temp 0 -n 8 \
		-p "$PROMPT" \
		--no-display-prompt 2>/dev/null || true
	echo ""
	echo "Сравните вывод с go test -tags=integration ./test/integration/ -run Golden -v"
else
	echo ""
	echo "llama-cli не найден - опциональный smoke test пропущен"
fi
