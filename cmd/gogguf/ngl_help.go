//go:build !metal

package main

// nglUsage is the -ngl flag help (default build: CUDA only, macOS offload rejected)
const nglUsage = "GPU layers (CUDA, -tags cuda); on macOS unsupported until Metal kernels (build -tags metal wires -ngl once kernels exist; see docs/platforms.md)"
