# Platforms and GPUs

Where you can run GoGGUF, and how to use a GPU.

|                  | Works now                                                                                                                                                                                   | Planned                                                              |
|------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------|
| **CPU**          | Linux, macOS, Windows                                                                                                                                                                       | -                                                                    |
| **GPU (`-ngl`)** | **Linux + NVIDIA**; Windows driver load (`nvcuda.dll`, needs CGO + GPU); macOS Metal B1 decode + B2 residency / fused Q8_0 decode + B3 Q8_0 prefill GEMM / batch attn + FFN (`-tags metal`) | Tiled `simdgroup_matrix` GEMM, other fused quants, Mac smoke + bench |
| **Other GPUs**   | -                                                                                                                                                                                           | AMD / Intel                                                          |

Tested on **GTX 1050 Ti (4 GB)**, **RTX 5060 Ti (16 GB)**, **Quadro RTX 6000 (24 GB)**.

## Linux + NVIDIA (GPU)

Needs an NVIDIA driver (`libcuda.so`). CUDA Toolkit is **not** required.

```bash
CGO_ENABLED=1 go build -tags cuda -o build/gogguf ./cmd/gogguf

./build/gogguf run -m model.gguf --chat -ngl 999 -p "Hi"
./build/gogguf serve -m model.gguf --host 127.0.0.1:8000 -ngl 999
```

**`-tags cuda` alone does not put the model on the GPU.** It only builds CUDA support into the binary. Offload starts when you pass **`-ngl N`** with `N > 0` (e.g. `-ngl 999` = as many layers as fit). Default is `-ngl 0` (CPU). On a successful GPU load, stderr shows `GPU offload: ... layers on CUDA:...`.

Without `-tags cuda`, `-ngl > 0` fails with `gpu: CUDA unavailable`.

| Flag / env               | Meaning                                                  |
|--------------------------|----------------------------------------------------------|
| `-ngl N`                 | offload first N layers (`0` = CPU; `999` = all that fit) |
| `-dev 1` or `-dev 0,1`   | GPU ordinal(s) **after** `CUDA_VISIBLE_DEVICES`          |
| `-tensor-split 0.6,0.4`  | layer share across devices from `-dev`                   |
| `CUDA_VISIBLE_DEVICES=1` | hide / reorder GPUs for the process                      |

Residency paths: on by default; toggle only via `LoadOptions` in the library ([library.md](library.md)) - no `GGUF_*_RESIDENCY` env.

If the model does not fit in VRAM: smaller quant, lower `-ngl`, or a smaller model. Free GPU memory between loads with `Engine.Close()` (library).

```bash
# second GPU / split
./build/gogguf run -m model.gguf -ngl 28 -dev 1 -p "Hi"
./build/gogguf serve -m model.gguf -ngl 28 -dev 0,1 -tensor-split 0.75,0.25
CUDA_VISIBLE_DEVICES=1 ./build/gogguf run -m model.gguf -ngl 28 -p "Hi"
```

Multi-GPU (`-dev 0,1`): layers are split (sequential layer ranges, not tensor-parallel). At device boundaries hidden moves with **async P2P** into a double-buffered resid when the topology allows (`cuMemcpyPeerAsync`), otherwise host bounce. Optional micro-batch overlap: `QueueNextResid` HtoD on the departing GPU while the peer copy finishes. Disable P2P with `GOGGUF_CUDA_P2P=0`.

`tools bench` with `-ngl` prints the device and VRAM (`VRAM: used / total MB`, or `vram_*_mb` with `--json`).

### Docker (GPU)

Linux `amd64` only, NVIDIA Container Toolkit:

```bash
docker build --target runtime-cuda -t gogguf-cuda .
docker run --rm --gpus all -p 8000:8000 -v "$(pwd)/models:/models:ro" gogguf-cuda serve -m /models/Qwen3-4B-Q8_0.gguf --host 0.0.0.0:8000 -ngl 36
```

CPU-only Docker: [build.md](build.md).

### What runs on GPU (short)

Best case: **Qwen3 dense**, `-ngl` = all layers, fused quants (Q8_0 / Q8_1 / Q4_0/1 / Q5_0/1 / Q2_K...Q8_K) - most of the transformer + logits on GPU.

Partially / weaker: MoE / DeepSeek2 MLA use serial-per-token inside `-b` chunks; F32 batch GEMM is still loop-B (quant GEMM is real).  
Always CPU: tokenizer, sampling, HTTP.  
Default CLI `-b` is **32** (`PrefillBatchSize` reports the effective chunk size).  
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
- **GPU (`-ngl`):** **Metal B1 decode path + B2 residency** with `-tags metal` (needs cgo + Xcode command line tools). MSL kernels (`matmul_vec_f32`, `matmul_vec_q8_0`, `matmul_vec_q4_0`, `matmul_vec_q4_k`, `rmsnorm`, `rope_heads` NeoX, `swiglu`, `add`, `attn_qk_softmax_v`) plus a unified-memory GPU KV cache; weights are uploaded to `MTLBuffer`s once by name. Best case: **Qwen3 dense Q8_0**, `-ngl` = all layers: the shared `gpuresid` path keeps the residual in a `MTLBuffer` between layers (`HiddenResident()`; one `HiddenUpload` per token), runs QKV+QK-norm+RoPE+KV append+attention and WO+residual+FFN+residual each as **one command buffer per layer**, and `LogitsFromDevice` copies only the logits back. `Backend.Ready()` (runtime opened, kernels built in) is what lets `-ngl` offload; `HiddenResident()` additionally needs rmsnorm+add+swiglu+attention+Q8_0 matvec in the runtime. The fused decode path is Q8_0 + NeoX only: other quant types, Llama-style RoPE and GeGLU/Gemma fall back to the B1 per-op path (one synchronous command buffer per op, still on the GPU) or the CPU. Q4_0 / Q4_K matvec is available per op and for the `lm_head`. **B3 prefill (`-b` ≥ 2, Q8_0):** the Metal backend implements `MatMulMatBackend`, `FFNSwiGLUMatBackend` and `KVBatchAttentionBackend` like CUDA, so Qwen3 dense Q8_0 prefill runs through the same `pkg/model/qwen3` batch path. `MatMulMatQ8_0CachedInto` is **one `matmul_mat_q8_0` launch for W·X[batch]** (CUDA-parity algorithm: one thread per (batch, row) output with fused Q8_0 dequant + dot; `batch == 1` uses the decode matvec kernel; the weight is shared with decode through the same by-name device cache); consecutive GEMMs on the same host slice (Q->K->V, gate->up) skip re-staging the input. `FFNSwiGLUMatQuantCached` (Q8_0 only) runs gate/up GEMM -> SwiGLU -> down GEMM with activations on the device in **one command buffer**; `AttentionScoresKVBatch` encodes the n causal single-query attention launches (query *i* sees `pastLen+i+1` positions) into **one command buffer**. F32 and other quants (Q4_0 / Q4_K / ...) keep working: the other `MatMulMat*` entries loop the matvec kernels inside the method (no GEMM kernel, so no prefill speed-up for them), the batch FFN returns `ErrUnavailable` and `qwen3` falls back to per-matmul batches. `SoftcapInPlace` stays on the host. The GEMM is the plain thread-per-output kernel; a tiled `simdgroup_matrix` kernel is a follow-up once it can be tuned on a Mac. Without `-tags metal`, `-ngl > 0` is rejected (CPU: `NGL=0`).  
- **mmap:** yes (`mmap`)

```bash
go build -o build/gogguf ./cmd/gogguf
./build/gogguf run -m model.gguf --chat -p "Hi"
./build/gogguf serve -m model.gguf --host 127.0.0.1:8000
# Metal decode (Apple Silicon / Intel Mac):
go build -tags metal -o build/gogguf ./cmd/gogguf
./build/gogguf run -m Qwen3-1.7B-Q8_0.gguf -ngl 999 --chat -p "Hi"
go test -tags metal ./pkg/device/metal/   # GPU parity tests (skip without Metal)
```

Metal limits: B1 per-op dispatch is one synchronous command buffer per op; B2 (Qwen3 dense Q8_0) fuses a layer into two command buffers with a device-resident residual; B3 prefill is one GEMM launch per matmul, one command buffer per batch FFN / batch attention. Not yet benchmarked or smoke-tested on a Mac (see the Metal protocol in [tools.md](tools.md)); Q4_0 follows the `pkg/quant` nibble layout.

## Windows

- **CPU:** yes (`amd64` / `arm64`)  
- **GPU (`-ngl`):** driver load is wired (`nvcuda.dll` via `LoadLibrary`, no `-ldl`). Build with a C compiler (MinGW or MSVC via CGO) and an NVIDIA driver:

```powershell
$env:CGO_ENABLED = "1"
go build -tags cuda -o build/gogguf.exe ./cmd/gogguf
.\build\gogguf.exe run -m model.gguf --chat -ngl 999 -p "Hi"
```

CUDA Toolkit is not required. Without a GPU the binary still builds; `-ngl > 0` fails at runtime if `nvcuda.dll` or a device is missing.  
- **mmap:** yes (`CreateFileMapping` / `MapViewOfFile`)

PowerShell (CPU):

```powershell
go build -o build/gogguf.exe ./cmd/gogguf
.\build\gogguf.exe run -m model.gguf --chat -p "Hi"
.\build\gogguf.exe serve -m model.gguf --host 127.0.0.1:8000
```

Ctrl+C stops `serve`.

## Planned (GPU roadmap)

- Metal: tiled `simdgroup_matrix` GEMM, GEMM for quants other than Q8_0, fused Q4_0/Q4_K/Q5_K/Q6_K layers, Llama RoPE / Gemma on Metal, Mac smoke + bench; Windows CUDA needs a real-GPU smoke  
- True tensor-parallel / deeper pipeline-parallel across sequences  
- Batched MoE / remaining arch residency gaps  

CLI flags: [cli.md](cli.md). Library: [library.md](library.md).
