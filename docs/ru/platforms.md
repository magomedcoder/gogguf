# Платформы и видеокарты

Где можно запускать GoGGUF и как включить GPU.

|                  | Сейчас                    | Планируется                     |
|------------------|---------------------------|---------------------------------|
| **CPU**          | Linux, macOS, Windows     | -                               |
| **GPU (`-ngl`)** | **только Linux + NVIDIA** | Metal на macOS, CUDA на Windows |
| **Другие GPU**   | -                         | AMD / Intel                     |

Тестировалось на **GTX 1050 Ti (4 GB)**, **RTX 5060 Ti*, **Quadro RTX 6000 (24 GB)***.

## Linux + NVIDIA (GPU)

Нужен драйвер NVIDIA (`libcuda.so`). CUDA Toolkit **не** нужен.

```bash
CGO_ENABLED=1 go build -tags cuda -o build/gogguf ./cmd/gogguf

./build/gogguf run -m model.gguf --chat -ngl 999 -p "Привет"
./build/gogguf serve -m model.gguf --host 127.0.0.1:8000 -ngl 999
```

Без `-tags cuda` при `-ngl > 0` будет ошибка `gpu: CUDA unavailable`.

| Флаг / env                                                                       | Смысл                                          |
|----------------------------------------------------------------------------------|------------------------------------------------|
| `-ngl N`                                                                         | первые N слоёв на GPU (`999` = сколько влезет) |
| `-dev 1` или `-dev 0,1`                                                          | номер GPU **после** `CUDA_VISIBLE_DEVICES`     |
| `-tensor-split 0.6,0.4`                                                          | доля слоёв по устройствам из `-dev`            |
| `CUDA_VISIBLE_DEVICES=1`                                                         | какие GPU видит процесс                        |
| `GGUF_HIDDEN_RESIDENCY=0` / `GGUF_QKV_RESIDENCY=0` / `GGUF_ATTN_FFN_RESIDENCY=0` | выкл. fused residency                          |

Если модель не влезает в VRAM - меньший квант, меньше `-ngl` или меньшая модель. Между загрузками в коде: `Engine.Close()`.

```bash
# вторая карта / split
./build/gogguf run -m model.gguf -ngl 28 -dev 1 -p "Привет"
./build/gogguf serve -m model.gguf -ngl 28 -dev 0,1 -tensor-split 0.75,0.25
CUDA_VISIBLE_DEVICES=1 ./build/gogguf run -m model.gguf -ngl 28 -p "Привет"
```

Multi-GPU (`-dev 0,1`) - **частично**: слои делятся, hidden между GPU через host, параллельного compute между картами пока нет.

`tools bench` с `-ngl` пишет устройство и VRAM (`VRAM: used / total MB`, или `vram_*_mb` при `--json`).

### Docker (GPU)

Только Linux `amd64`, нужен NVIDIA Container Toolkit:

```bash
docker build --target runtime-cuda -t gogguf-cuda .
docker run --rm --gpus all -p 8000:8000 -v "$(pwd)/models:/models:ro" gogguf-cuda serve -m /models/Qwen3-4B-Q8_0.gguf --host 0.0.0.0:8000 -ngl 36
```

Docker только CPU: [build.md](build.md).

### Что считается на GPU (кратко)

Лучший случай: **Qwen3 dense**, `-ngl` = все слои, веса Q8_0 / Q4_0 / Q4_K / Q5_K / Q6_K - почти весь transformer + logits на GPU.

Слабее / частично: другие arch, MoE, DeepSeek2 MLA, prefill `-b > 1` (ещё не true batched GEMM).  
Всегда CPU: tokenizer, sampling, HTTP.  
Подробнее: [build.md](build.md) (раздел CUDA) и [models.md](models.md).

## macOS

- **CPU:** да  
- **GPU (`-ngl`):** нет - Metal только заглушка (`-tags metal`). `-ngl` отклоняется.

В планах: Metal kernels для `-ngl` на Apple Silicon / Intel Mac.

## Windows

- **CPU:** да  
- **GPU (`-ngl`):** нет - загрузчик CUDA рассчитан на Linux `libcuda.so`.  
- **mmap:** большие GGUF читаются обычным файлом (настоящий `CreateFileMapping` - в планах).

В планах: CUDA на Windows.

## Планируется (GPU)

- Metal (macOS), CUDA (Windows)  
- Более быстрый путь (true batched GEMM, flash-attn, residency на всех arch)  
- Более сильный multi-GPU  

Флаги CLI: [cli.md](cli.md). Библиотека: [library.md](library.md).
