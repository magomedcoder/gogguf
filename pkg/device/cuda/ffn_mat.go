//go:build cuda

package cuda

/*
#include "driver.h"
*/
import "C"

import (
	"fmt"
	"unsafe"

	"github.com/magomedcoder/gogguf/pkg/dtype"
)

// FFNSwiGLUMatCached runs batch FFN on device: one HtoD of x[B*embd], gate/up/SwiGLU/down, one DtoH
func (b *Backend) FFNSwiGLUMatCached(gateName, upName, downName string, gateW, upW, downW, x, out []float32, embd, ffn, batch int) error {
	if !b.hasSwiGLU {
		return fmt.Errorf("cuda: SwiGLU kernel unavailable")
	}

	if err := b.ffnMatValidate(x, out, embd, ffn, batch); err != nil {
		return err
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	gateM, err := b.ensureFP32Matrix(gateName, gateW, ffn, embd)
	if err != nil {
		return err
	}

	upM, err := b.ensureFP32Matrix(upName, upW, ffn, embd)
	if err != nil {
		return err
	}

	downM, err := b.ensureFP32Matrix(downName, downW, embd, ffn)
	if err != nil {
		return err
	}

	b.prepareVecUpload(x[:batch*embd])
	var fnMat C.CUfunction
	if b.hasMatF32 {
		fnMat = b.fnMatF32
	}

	rc := C.gguf_cuda_ffn_swiglu_mat_device(
		&b.drv,
		b.ctx,
		b.fn,
		fnMat,
		b.fnSwiGLU,
		&b.matmulPool,
		gateM.ptr,
		upM.ptr,
		downM.ptr,
		(*C.float)(unsafe.Pointer(&x[0])),
		(*C.float)(unsafe.Pointer(&out[0])),
		C.int(embd),
		C.int(ffn),
		C.int(batch),
	)
	b.lastVecAddr = 0
	b.lastVecLen = 0
	if rc != 0 {
		return fmt.Errorf("cuda: ffn_swiglu_mat: code %d", int(rc))
	}

	return nil
}

// FFNSwiGLUMatQuantCached is the quantized-weight variant of FFNSwiGLUMatCached
func (b *Backend) FFNSwiGLUMatQuantCached(t dtype.DType, gateName, upName, downName string, gateRaw, upRaw, downRaw []byte, x, out []float32, embd, ffn, batch int) error {
	if !b.hasSwiGLU {
		return fmt.Errorf("cuda: SwiGLU kernel unavailable")
	}
	if err := b.ffnMatValidate(x, out, embd, ffn, batch); err != nil {
		return err
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	_, fnMatmul, err := b.quantCache(t)
	if err != nil {
		return err
	}

	gateM, err := b.ensureQuantMatrix(t, gateName, gateRaw, ffn, embd)
	if err != nil {
		return err
	}

	upM, err := b.ensureQuantMatrix(t, upName, upRaw, ffn, embd)
	if err != nil {
		return err
	}

	downM, err := b.ensureQuantMatrix(t, downName, downRaw, embd, ffn)
	if err != nil {
		return err
	}

	b.prepareVecUpload(x[:batch*embd])
	var fnMat C.CUfunction
	if (t == dtype.Q8_0 || t == dtype.Q8_1) && b.hasMatQ8 {
		fnMat = b.fnMatQ8
	}

	rc := C.gguf_cuda_ffn_swiglu_mat_device(
		&b.drv,
		b.ctx,
		fnMatmul,
		fnMat,
		b.fnSwiGLU,
		&b.matmulPool,
		gateM.ptr,
		upM.ptr,
		downM.ptr,
		(*C.float)(unsafe.Pointer(&x[0])),
		(*C.float)(unsafe.Pointer(&out[0])),
		C.int(embd),
		C.int(ffn),
		C.int(batch),
	)
	b.lastVecAddr = 0
	b.lastVecLen = 0
	if rc != 0 {
		return fmt.Errorf("cuda: ffn_swiglu_mat %s: code %d", t, int(rc))
	}
	return nil
}

func (b *Backend) ffnMatValidate(x, out []float32, embd, ffn, batch int) error {
	if batch < 1 || embd < 1 || ffn < 1 {
		return fmt.Errorf("cuda: ffn_mat: batch=%d embd=%d ffn=%d", batch, embd, ffn)
	}

	if len(x) < batch*embd || len(out) < batch*embd {
		return fmt.Errorf("cuda: ffn_mat: buffers too short")
	}

	return nil
}
