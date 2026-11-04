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

// AttnOutResidualNormQuantCached: WO(attn)+resid+= + RMSNorm -> MoE snapshot
func (b *Backend) AttnOutResidualNormQuantCached(t dtype.DType, woName, ffnNormName string, woRaw []byte, ffnNorm, x, attn []float32, embd, attnDim int, eps float32) error {
	if !b.hasRMS || !b.hasAdd {
		return fmt.Errorf("cuda: attn_out_norm kernels unavailable")
	}

	if len(ffnNorm) < embd {
		return fmt.Errorf("cuda: AttnOutResidualNormQuantCached: ffn_norm too short")
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	resident := b.hiddenResident && len(x) == 0
	if !resident {
		if len(x) < embd {
			return fmt.Errorf("cuda: AttnOutResidualNormQuantCached: x too short")
		}
		b.clearResidencyLocked()
	}

	_, fnMatmul, err := b.quantCache(t)
	if err != nil {
		return err
	}

	woM, err := b.ensureQuantMatrix(t, woName, woRaw, embd, attnDim)
	if err != nil {
		return err
	}

	normM, err := b.ensureFP32Matrix(ffnNormName, ffnNorm, embd, 1)
	if err != nil {
		return err
	}

	var xPtr, attnPtr *C.float
	if len(x) >= embd {
		xPtr = (*C.float)(unsafe.Pointer(&x[0]))
	}
	if len(attn) > 0 {
		attnPtr = (*C.float)(unsafe.Pointer(&attn[0]))
	}

	var dirty C.int
	rc := C.gguf_cuda_attn_out_norm_device(
		&b.drv, b.ctx, fnMatmul, b.fnRMS, b.fnAdd, &b.matmulPool,
		woM.ptr, normM.ptr, xPtr, attnPtr,
		C.int(embd), C.int(attnDim), C.float(eps), &dirty,
	)
	b.lastVecAddr = 0
	b.lastVecLen = 0
	if rc != 0 {
		if resident && dirty != 0 {
			b.clearResidencyLocked()
		}
		return fmt.Errorf("cuda: attn_out_norm %s: code %d", t, int(rc))
	}

	b.hiddenResident = true
	b.matmulPool.keep_resid_device = 1

	return nil
}

// MoEHiddenNorm: RMSNorm(d_resid) -> MoE snapshot
func (b *Backend) MoEHiddenNorm(ffnNormName string, ffnNorm []float32, embd int, eps float32) error {
	if !b.hasRMS {
		return fmt.Errorf("cuda: MoEHiddenNorm: rmsnorm unavailable")
	}

	if len(ffnNorm) < embd {
		return fmt.Errorf("cuda: MoEHiddenNorm: ffn_norm too short")
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	if !b.hiddenResident {
		return fmt.Errorf("cuda: MoEHiddenNorm: residency not active")
	}

	normM, err := b.ensureFP32Matrix(ffnNormName, ffnNorm, embd, 1)
	if err != nil {
		return err
	}

	rc := C.gguf_cuda_moe_hidden_norm(&b.drv, b.ctx, b.fnRMS, &b.matmulPool, normM.ptr, C.int(embd), C.float(eps))
	b.lastVecAddr = 0
	b.lastVecLen = 0
	if rc != 0 {
		return fmt.Errorf("cuda: moe_hidden_norm: code %d", int(rc))
	}

	return nil
}

// MatMulVecFromMoEHiddenQuant: router matmul from MoE snapshot
func (b *Backend) MatMulVecFromMoEHiddenQuant(t dtype.DType, name string, raw []byte, rows, cols int, out []float32) error {
	if len(out) < rows {
		return fmt.Errorf("cuda: MatMulVecFromMoEHiddenQuant: out too short")
	}

	_, fn, err := b.quantCache(t)
	if err != nil {
		return err
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	gm, err := b.ensureQuantMatrix(t, name, raw, rows, cols)
	if err != nil {
		return err
	}

	rc := C.gguf_cuda_matmul_vec_from_moe_h(&b.drv, b.ctx, fn, &b.matmulPool, gm.ptr, (*C.float)(unsafe.Pointer(&out[0])), C.int(rows), C.int(cols))
	b.lastVecAddr = 0
	b.lastVecLen = 0
	if rc != 0 {
		return fmt.Errorf("cuda: matmul_from_moe_h %s: code %d", t, int(rc))
	}

	return nil
}

// MatMulVecFromMoEHiddenF32: FP32 router from MoE snapshot
func (b *Backend) MatMulVecFromMoEHiddenF32(name string, matrix []float32, rows, cols int, out []float32) error {
	if len(out) < rows || len(matrix) < rows*cols {
		return fmt.Errorf("cuda: MatMulVecFromMoEHiddenF32: buffers too short")
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	m, err := b.ensureFP32Matrix(name, matrix, rows, cols)
	if err != nil {
		return err
	}

	rc := C.gguf_cuda_matmul_vec_from_moe_h(&b.drv, b.ctx, b.fn, &b.matmulPool, m.ptr, (*C.float)(unsafe.Pointer(&out[0])), C.int(rows), C.int(cols))
	b.lastVecAddr = 0
	b.lastVecLen = 0
	if rc != 0 {
		return fmt.Errorf("cuda: matmul_from_moe_h f32: code %d", int(rc))
	}

	return nil
}

// FFNSwiGLUMoEAccumulateQuant: ExpertFFN from MoE snapshot; resid += scale * out
func (b *Backend) FFNSwiGLUMoEAccumulateQuant(t dtype.DType, gateName, upName, downName string, gateRaw, upRaw, downRaw []byte, scale float32, embd, ffn int) error {
	if !b.hasSwiGLU || !b.hasAdd || (scale != 1 && !b.hasAddScaled) {
		return fmt.Errorf("cuda: FFNSwiGLUMoEAccumulate: kernels unavailable")
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	if !b.hiddenResident {
		return fmt.Errorf("cuda: FFNSwiGLUMoEAccumulate: residency not active")
	}

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

	rc := C.gguf_cuda_ffn_swiglu_moe_accum(
		&b.drv, b.ctx, fnMatmul, b.fnSwiGLU, b.fnAdd, b.fnAddScaled, &b.matmulPool,
		gateM.ptr, upM.ptr, downM.ptr, C.float(scale), C.int(embd), C.int(ffn),
	)
	b.lastVecAddr = 0
	b.lastVecLen = 0
	if rc != 0 {
		return fmt.Errorf("cuda: ffn_moe_accum %s: code %d", t, int(rc))
	}

	return nil
}

// ResidualAddScaled: resid += scale * src (host)
func (b *Backend) ResidualAddScaled(src []float32, scale float32) error {
	if !b.hasAdd {
		return fmt.Errorf("cuda: ResidualAddScaled: add kernel unavailable")
	}

	if len(src) == 0 {
		return fmt.Errorf("cuda: ResidualAddScaled: empty src")
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	if !b.hiddenResident {
		return fmt.Errorf("cuda: ResidualAddScaled: residency not active")
	}

	rc := C.gguf_cuda_residual_add_scaled(
		&b.drv, b.ctx, b.fnAdd, b.fnAddScaled, &b.matmulPool,
		(*C.float)(unsafe.Pointer(&src[0])), C.float(scale), C.int(len(src)),
	)
	b.lastVecAddr = 0
	b.lastVecLen = 0
	if rc != 0 {
		return fmt.Errorf("cuda: residual_add_scaled: code %d", int(rc))
	}

	return nil
}
