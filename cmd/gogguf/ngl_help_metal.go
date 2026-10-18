//go:build metal

package main

// nglUsage is the -ngl flag help for -tags metal builds: macOS routes -ngl to the Metal backend, which still rejects offload while kernels are scaffold (see docs/platforms.md)
const nglUsage = "GPU layers (CUDA, -tags cuda; macOS Metal, -tags metal: scaffold, -ngl>0 rejected until Metal kernels land; see docs/platforms.md)"
