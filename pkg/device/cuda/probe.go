//go:build cuda

package cuda

/*
#cgo LDFLAGS: -ldl -lm
#include "driver.h"
#include <stdlib.h>
*/
import "C"
import (
	"fmt"
	"unsafe"
)

// KernelsPTXForTarget exports PTX matmul (diagnostics)
func KernelsPTXForTarget(target int) string {
	return kernelsPTXForTarget(target)
}

// MatmulVecPTXForTarget matmul_vec only
func MatmulVecPTXForTarget(target int) string {
	return ptxHeaderForTarget(target) + matmulVecKernel
}

// MatmulQ8PTXForTarget matmul_vec_q8_0 only
func MatmulQ8PTXForTarget(target int) string {
	return ptxHeaderForTarget(target) + matmulQ8Kernel
}

// MatmulQ4PTXForTarget matmul_vec_q4_0 only
func MatmulQ4PTXForTarget(target int) string {
	return ptxHeaderForTarget(target) + matmulQ4Kernel
}

// MatmulQ4KPTXForTarget matmul_vec_q4_k only
func MatmulQ4KPTXForTarget(target int) string {
	return ptxHeaderForTarget(target) + matmulQ4KKernel
}

// MatmulQ5KPTXForTarget matmul_vec_q5_k only
func MatmulQ5KPTXForTarget(target int) string {
	return ptxHeaderForTarget(target) + matmulQ5KKernel
}

// MatmulQ6KPTXForTarget matmul_vec_q6_k only
func MatmulQ6KPTXForTarget(target int) string {
	return ptxHeaderForTarget(target) + matmulQ6KKernel
}

// ProbeLoadPTX tries to load PTX on GPU 0
func ProbeLoadPTX(ptx string) error {
	var drv C.cuda_driver_t
	var lib unsafe.Pointer
	var ctx C.CUcontext
	var nameBuf [256]C.char
	var initErr [512]C.char
	var cc C.int
	if rc := C.gguf_cuda_init(&drv, &lib, &ctx, 0, &nameBuf[0], C.size_t(len(nameBuf)), &initErr[0], C.size_t(len(initErr)), &cc); rc != 0 {
		return fmt.Errorf("init: %s", C.GoString(&initErr[0]))
	}
	defer C.gguf_cuda_shutdown(&drv, ctx)

	cptx := C.CString(ptx)
	defer C.free(unsafe.Pointer(cptx))

	var module C.CUmodule
	var errBuf [8192]C.char
	rc := C.gguf_cuda_load_module(&drv, ctx, cptx, &module, nil, nil, nil, nil, nil, &errBuf[0], C.size_t(len(errBuf)))
	if rc != 0 {
		return fmt.Errorf("%s", C.GoString(&errBuf[0]))
	}

	return nil
}
