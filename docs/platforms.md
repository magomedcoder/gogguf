# Platforms and GPUs

Where you can run GoGGUF, and how to use a GPU.

|                  | Works now               | Planned                   |
|------------------|-------------------------|---------------------------|
| **CPU**          | Linux, macOS, Windows   | -                         |
| **GPU (`-ngl`)** | **Linux + NVIDIA** only | macOS Metal, Windows CUDA |
| **Other GPUs**   | -                       | AMD / Intel               |

Tested on **GTX 1050 Ti (4 GB)**, **RTX 5060 Ti (16 GB)**, **Quadro RTX 6000 (24 GB)**.

## Linux + NVIDIA (GPU)

Needs an NVIDIA driver (`libcuda.so`). CUDA Toolkit is **not** required.

```bash
CGO_ENABLED=1 go build -tags cuda -o build/gogguf ./cmd/gogguf

./build/gogguf run -m model.gguf --chat -ngl 999 -p "Hi"
./build/gogguf serve -m model.gguf --host 127.0.0.1:8000 -ngl 999
```

Without `-tags cuda`, `-ngl > 0` fails with `gpu: CUDA unavailable`.

| Flag / env                       | Meaning                                         |
|----------------------------------|-------------------------------------------------|
| `-ngl N`                         | offload first N layers (`999` = all that fit)   |
| `-dev 1` or `-dev 0,1`           | GPU ordinal(s) **after** `CUDA_VISIBLE_DEVICES` |
| `-tensor-split 0.6,0.4`          | layer share across devices from `-dev`          |
| `CUDA_VISIBLE_DEVICES=1`         | hide / reorder GPUs for the process             |

Residency paths: on by default; toggle only via `LoadOptions` in the library ([library.md](library.md)) - no `GGUF_*_RESIDENCY` env.

If the model does not fit in VRAM: smaller quant, lower `-ngl`, or a smaller model. Free GPU memory between loads with `Engine.Close()` (library).

```bash
# second GPU / split
./build/gogguf run -m model.gguf -ngl 28 -dev 1 -p "Hi"
./build/gogguf serve -m model.gguf -ngl 28 -dev 0,1 -tensor-split 0.75,0.25
CUDA_VISIBLE_DEVICES=1 ./build/gogguf run -m model.gguf -ngl 28 -p "Hi"
```

Multi-GPU (`-dev 0,1`) is **partial**: layers are split, hidden crosses devices via host, no parallel compute across GPUs yet.

`tools bench` with `-ngl` prints the device and VRAM (`VRAM: used / total MB`, or `vram_*_mb` with `--json`).

### Docker (GPU)

Linux `amd64` only, NVIDIA Container Toolkit:

```bash
docker build --target runtime-cuda -t gogguf-cuda .
docker run --rm --gpus all -p 8000:8000 -v "$(pwd)/models:/models:ro" gogguf-cuda serve -m /models/Qwen3-4B-Q8_0.gguf --host 0.0.0.0:8000 -ngl 36
```

CPU-only Docker: [build.md](build.md).

### What runs on GPU (short)

Best case: **Qwen3 dense**, `-ngl` = all layers, weights Q8_0 / Q4_0 / Q4_K / Q5_K / Q6_K - most of the transformer + logits on GPU.

Partially / weaker: other arches, MoE, DeepSeek2 MLA, `-b > 1` prefill (not true batched GEMM yet).  
Always CPU: tokenizer, sampling, HTTP.  
Details and residency notes: [build.md](build.md) (CUDA section) and [models.md](models.md).

## CPU (all platforms)

CPU works without CGO. Cross-build examples:

```bash
# Windows
CGO_ENABLED=0 GOOS=windows GOARCH=amd64 go build -o build/gogguf.exe ./cmd/gogguf
CGO_ENABLED=0 GOOS=windows GOARCH=arm64 go build -o build/gogguf-arm64.exe ./cmd/gogguf

# macOS
CGO_ENABLED=0 GOOS=darwin GOARCH=arm64 go build -o build/gogguf ./cmd/gogguf
CGO_ENABLED=0 GOOS=darwin GOARCH=amd64 go build -o build/gogguf-amd64 ./cmd/gogguf
```

| Target          | Status        |
|-----------------|---------------|
| `linux/amd64`   | ok            |
| `linux/arm64`   | ok            |
| `darwin/amd64`  | ok            |
| `darwin/arm64`  | ok (NEON asm) |
| `windows/amd64` | ok            |
| `windows/arm64` | ok (NEON asm) |

HF cache default: `~/.cache/huggingface/hub` (Unix) or `%LOCALAPPDATA%\huggingface\hub` (Windows). Hub env: [cli.md](cli.md#environment-hugging-face).

## macOS

- **CPU:** yes (`amd64` / `arm64`)  
- **GPU (`-ngl`):** no - Metal is a stub (`-tags metal`); compute not implemented. `-ngl` is rejected.  
- **mmap:** yes (`mmap`)

```bash
go build -o build/gogguf ./cmd/gogguf
./build/gogguf run -m model.gguf --chat -p "Hi"
./build/gogguf serve -m model.gguf --host 127.0.0.1:8000
```

Planned: Metal kernels so `-ngl` works on Apple Silicon / Intel Mac.

## Windows

- **CPU:** yes (`amd64` / `arm64`)  
- **GPU (`-ngl`):** no - CUDA loader uses Linux `libcuda.so`; Windows driver path not wired.  
- **mmap:** yes (`CreateFileMapping` / `MapViewOfFile`)

PowerShell:

```powershell
go build -o build/gogguf.exe ./cmd/gogguf
.\build\gogguf.exe run -m model.gguf --chat -p "Hi"
.\build\gogguf.exe serve -m model.gguf --host 127.0.0.1:8000
```

Ctrl+C stops `serve`. Planned: Windows CUDA.

## Planned (GPU roadmap)

- macOS Metal, Windows CUDA  
- Faster path (true batched GEMM, flash-attn, full residency on all arches)  
- Stronger multi-GPU  

CLI flags: [cli.md](cli.md). Library: [library.md](library.md).
