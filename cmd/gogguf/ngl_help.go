//go:build !metal

package main

// nglUsage is the -ngl flag help (default build: CUDA only, macOS offload rejected)
const nglUsage = "GPU layers (CUDA, -tags cuda); on macOS build with -tags metal for the Metal decode path (see docs/platforms.md)"
