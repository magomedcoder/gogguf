//go:build metal

package main

// nglUsage is the -ngl flag help for -tags metal builds: on macOS -ngl runs the Metal B1 decode path (Q8_0 / F32 dense matvec, RMSNorm, RoPE, SwiGLU, attention, GPU KV; other quants fall back to CPU per op), see docs/platforms.md
const nglUsage = "GPU layers (CUDA, -tags cuda; macOS Metal, -tags metal: B1 decode path, best on Qwen3 dense Q8_0; see docs/platforms.md)"
