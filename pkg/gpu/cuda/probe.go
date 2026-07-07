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

// KernelsPTXForTarget экспортирует PTX matmul (диагностика)
func KernelsPTXForTarget(target int) string {
	return kernelsPTXForTarget(target)
}

// MatmulVecPTXForTarget только matmul_vec
func MatmulVecPTXForTarget(target int) string {
	return ptxHeaderForTarget(target) + matmulVecKernel
}

// MatmulQ8PTXForTarget только matmul_vec_q8_0
func MatmulQ8PTXForTarget(target int) string {
	return ptxHeaderForTarget(target) + matmulQ8Kernel
}

// ProbeLoadPTX пробует загрузить PTX на GPU 0
func ProbeLoadPTX(ptx string) error {
	var drv C.cuda_driver_t
	var lib unsafe.Pointer
	var ctx C.CUcontext
	var nameBuf [256]C.char
	var initErr [512]C.char
	var cc C.int
	if rc := C.gguf_cuda_init(&drv, &lib, &ctx, &nameBuf[0], C.size_t(len(nameBuf)), &initErr[0], C.size_t(len(initErr)), &cc); rc != 0 {
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
