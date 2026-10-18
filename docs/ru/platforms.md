# Платформы и видеокарты

Где можно запускать GoGGUF и как включить GPU.

|                  | Сейчас                    | Планируется                     |
|------------------|---------------------------|---------------------------------|
| **CPU**          | Linux, macOS, Windows     | -                               |
| **GPU (`-ngl`)** | **только Linux + NVIDIA** | Metal на macOS, CUDA на Windows |
| **Другие GPU**   | -                         | AMD / Intel                     |

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

Multi-GPU (`-dev 0,1`): слои делятся; на границах устройств hidden идёт через **P2P** (`cuMemcpyPeer`), иначе host bounce. Слои по-прежнему последовательно. Отключить P2P: `GOGGUF_CUDA_P2P=0`.

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

Слабее / частично: MoE / DeepSeek2 MLA остаются `n_batch=1`; F32 batch GEMM ещё loop-B (quant GEMM есть).  
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
- **GPU (`-ngl`):** scaffold. Сборка `-tags metal` подключает `-ngl` к `OpenMetal`; пока kernels не готовы, NGL>0 даёт понятную ошибку (CPU: `NGL=0`).  
- **mmap:** да (`mmap`)

```bash
go build -o build/gogguf ./cmd/gogguf
./build/gogguf run -m model.gguf --chat -p "Hi"
./build/gogguf serve -m model.gguf --host 127.0.0.1:8000
```

В планах: Metal kernels для `-ngl` на Apple Silicon / Intel Mac.

## Windows

- **CPU:** да (`amd64` / `arm64`)  
- **GPU (`-ngl`):** нет - загрузчик CUDA рассчитан на Linux `libcuda.so`; путь Windows-драйвера не подключён.  
- **mmap:** да (`CreateFileMapping` / `MapViewOfFile`)

PowerShell:

```powershell
go build -o build/gogguf.exe ./cmd/gogguf
.\build\gogguf.exe run -m model.gguf --chat -p "Hi"
.\build\gogguf.exe serve -m model.gguf --host 127.0.0.1:8000
```

Ctrl+C останавливает `serve`. В планах: CUDA на Windows.

## Планируется (GPU)

- Metal kernels (wiring `-tags metal` есть), CUDA (Windows)  
- Параллельный multi-GPU pipeline (P2P уже предпочтителен)  
- Batched MoE / оставшиеся gaps residency  

Флаги CLI: [cli.md](cli.md). Библиотека: [library.md](library.md).
