# Платформы и видеокарты

Где можно запускать GoGGUF и как включить GPU.

|                  | Сейчас                                                                                                                                                   | Планируется                                                  |
|------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------|
| **CPU**          | Linux, macOS, Windows                                                                                                                                    | -                                                            |
| **GPU (`-ngl`)** | **Linux + NVIDIA**; Windows: загрузка драйвера (`nvcuda.dll`, нужны CGO и GPU); macOS Metal B1 decode + B2 residency / fused Q8_0 decode (`-tags metal`) | Metal GEMM prefill (B3), остальные fused quant, smoke на Mac |
| **Другие GPU**   | -                                                                                                                                                        | AMD / Intel                                                  |

Тестировалось на **GTX 1050 Ti (4 GB)**, **RTX 5060 Ti (16 GB)**, **Quadro RTX 6000 (24 GB)**.

## Linux + NVIDIA (GPU)

Нужен драйвер NVIDIA (`libcuda.so`). CUDA Toolkit **не** нужен.

```bash
CGO_ENABLED=1 go build -tags cuda -o build/gogguf ./cmd/gogguf

./build/gogguf run -m model.gguf --chat -ngl 999 -p "Hi"
./build/gogguf serve -m model.gguf --host 127.0.0.1:8000 -ngl 999
```

**Одной сборки `-tags cuda` недостаточно для GPU.** Она только включает CUDA в бинарник. Offload начинается при **`-ngl N`** с `N > 0` (например `-ngl 999` = сколько слоёв влезет). По умолчанию `-ngl 0` (CPU). При успешном GPU-load в stderr будет `GPU offload: ... layers on CUDA:...`.

Без `-tags cuda` при `-ngl > 0` будет ошибка `gpu: CUDA unavailable`.

| Флаг / env               | Смысл                                                     |
|--------------------------|-----------------------------------------------------------|
| `-ngl N`                 | первые N слоёв на GPU (`0` = CPU; `999` = сколько влезет) |
| `-dev 1` или `-dev 0,1`  | номер GPU **после** `CUDA_VISIBLE_DEVICES`                |
| `-tensor-split 0.6,0.4`  | доля слоёв по устройствам из `-dev`                       |
| `CUDA_VISIBLE_DEVICES=1` | какие GPU видит процесс                                   |

Residency: по умолчанию вкл.; выкл. только через `LoadOptions` в library ([library.md](library.md)) - без env `GGUF_*_RESIDENCY`.

Если модель не влезает в VRAM - меньший квант, меньше `-ngl` или меньшая модель. Между загрузками в коде: `Engine.Close()`.

```bash
# вторая карта / split
./build/gogguf run -m model.gguf -ngl 28 -dev 1 -p "Hi"
./build/gogguf serve -m model.gguf -ngl 28 -dev 0,1 -tensor-split 0.75,0.25
CUDA_VISIBLE_DEVICES=1 ./build/gogguf run -m model.gguf -ngl 28 -p "Hi"
```

Multi-GPU (`-dev 0,1`): слои делятся последовательно (не tensor-parallel). На границах - **async P2P** в double-buffer resid (`cuMemcpyPeerAsync`), иначе host bounce. Опционально: `QueueNextResid` HtoD на уходящем GPU пока идёт peer copy. Отключить P2P: `GOGGUF_CUDA_P2P=0`.

`tools bench` с `-ngl` пишет устройство и VRAM (`VRAM: used / total MB`, или `vram_*_mb` при `--json`).

### Docker (GPU)

Только Linux `amd64`, нужен NVIDIA Container Toolkit:

```bash
docker build --target runtime-cuda -t gogguf-cuda .
docker run --rm --gpus all -p 8000:8000 -v "$(pwd)/models:/models:ro" gogguf-cuda serve -m /models/Qwen3-4B-Q8_0.gguf --host 0.0.0.0:8000 -ngl 36
```

Docker только CPU: [build.md](build.md).

### Что считается на GPU (кратко)

Лучший случай: **Qwen3 dense**, `-ngl` = все слои, fused-кванты (Q8_0 / Q4_0/1 / Q5_0/1 / Q2_K...Q8_K) - почти весь transformer + logits на GPU.

Слабее / частично: MoE / DeepSeek2 MLA - serial-per-token внутри `-b` чанков; F32 batch GEMM ещё loop-B (quant GEMM есть).  
Всегда CPU: tokenizer, sampling, HTTP.  
Default CLI `-b` = **32** для dense prefill.  
Подробнее: [build.md](build.md) (раздел CUDA) и [models.md](models.md).

## CPU (все платформы)

CPU без CGO. Примеры кросс-сборки:

```bash
# Windows
CGO_ENABLED=0 GOOS=windows GOARCH=amd64 go build -o build/gogguf.exe ./cmd/gogguf
CGO_ENABLED=0 GOOS=windows GOARCH=arm64 go build -o build/gogguf-arm64.exe ./cmd/gogguf

# macOS
CGO_ENABLED=0 GOOS=darwin GOARCH=arm64 go build -o build/gogguf ./cmd/gogguf
CGO_ENABLED=0 GOOS=darwin GOARCH=amd64 go build -o build/gogguf-amd64 ./cmd/gogguf
```

| Цель            | Статус        |
|-----------------|---------------|
| `linux/amd64`   | ok            |
| `linux/arm64`   | ok            |
| `darwin/amd64`  | ok            |
| `darwin/arm64`  | ok (NEON asm) |
| `windows/amd64` | ok            |
| `windows/arm64` | ok (NEON asm) |

HF cache: `~/.cache/huggingface/hub` (Unix) или `%LOCALAPPDATA%\huggingface\hub` (Windows). Hub env: [cli.md](cli.md#переменные-окружения-hugging-face).

## macOS

- **CPU:** да (`amd64` / `arm64`)  
- **GPU (`-ngl`):** **Metal B1 decode + B2 residency** при сборке `-tags metal` (нужны cgo и Xcode command line tools). MSL kernels (`matmul_vec_f32`, `matmul_vec_q8_0`, `matmul_vec_q4_0`, `matmul_vec_q4_k`, `rmsnorm`, `rope_heads` NeoX, `swiglu`, `add`, `attn_qk_softmax_v`) и GPU KV в unified memory; веса один раз загружаются в `MTLBuffer` по имени. Лучший случай: **Qwen3 dense Q8_0**, `-ngl` = все слои: общий путь `gpuresid` держит residual в `MTLBuffer` между слоями (`HiddenResident()`; один `HiddenUpload` на токен), выполняет QKV+QK-norm+RoPE+KV append+attention и WO+residual+FFN+residual **одним command buffer на слой**, а `LogitsFromDevice` копирует на host только logits. `Backend.Ready()` (runtime открыт, kernels собраны) разрешает offload по `-ngl`; `HiddenResident()` дополнительно требует в runtime rmsnorm+add+swiglu+attention+Q8_0 matvec. Fused путь - только Q8_0 + NeoX: другие quant, Llama-RoPE, GeGLU/Gemma и batch prefill идут через B1 по операциям (один синхронный command buffer на операцию, всё ещё на GPU) или на CPU. Matvec Q4_0 / Q4_K доступен по операциям и для `lm_head`. Без `-tags metal` `-ngl > 0` отклоняется (CPU: `NGL=0`).  
- **mmap:** да (`mmap`)

```bash
go build -o build/gogguf ./cmd/gogguf
./build/gogguf run -m model.gguf --chat -p "Hi"
./build/gogguf serve -m model.gguf --host 127.0.0.1:8000
# Metal decode (Apple Silicon / Intel Mac):
go build -tags metal -o build/gogguf ./cmd/gogguf
./build/gogguf run -m Qwen3-1.7B-Q8_0.gguf -ngl 999 --chat -p "Hi"
go test -tags metal ./pkg/device/metal/   # parity-тесты GPU (skip без Metal)
```

Ограничения Metal: B1 - один синхронный command buffer на операцию; B2 (Qwen3 dense Q8_0) сводит слой к двум command buffer с device-resident residual. Бенчмарки и smoke на Mac пока не прогнаны; Q4_0 следует nibble-раскладке `pkg/quant`.

## Windows

- **CPU:** да (`amd64` / `arm64`)  
- **GPU (`-ngl`):** загрузчик драйвера подключён (`nvcuda.dll` через `LoadLibrary`, без `-ldl`). Нужны C-компилятор (MinGW или MSVC через CGO) и драйвер NVIDIA:

```powershell
$env:CGO_ENABLED = "1"
go build -tags cuda -o build/gogguf.exe ./cmd/gogguf
.\build\gogguf.exe run -m model.gguf --chat -ngl 999 -p "Hi"
```

CUDA Toolkit не нужен. Без GPU бинарник всё равно собирается; `-ngl > 0` падает в рантайме, если нет `nvcuda.dll` или устройства.  
- **mmap:** да (`CreateFileMapping` / `MapViewOfFile`)

PowerShell (CPU):

```powershell
go build -o build/gogguf.exe ./cmd/gogguf
.\build\gogguf.exe run -m model.gguf --chat -p "Hi"
.\build\gogguf.exe serve -m model.gguf --host 127.0.0.1:8000
```

Ctrl+C останавливает `serve`.

## Планируется (GPU)

- Metal B3 (GEMM prefill), fused слои Q4_0/Q4_K/Q5_K/Q6_K, Llama-RoPE / Gemma на Metal, smoke и bench на Mac; smoke CUDA на реальной Windows-машине  
- True tensor-parallel / более глубокий pipeline-parallel по последовательностям  
- Batched MoE / оставшиеся gaps residency  

Флаги CLI: [cli.md](cli.md). Библиотека: [library.md](library.md).
