// AVX2 scale-mul для RMSNorm: dst[i] = x[i] * scale * weight[i]

//go:build amd64

#include "textflag.h"

// func rmsnormScaleMulAVX2Asm(dst, x, weight []float32, scale float32, n int)
// n кратно 8
TEXT ·rmsnormScaleMulAVX2Asm(SB), NOSPLIT, $0-84
	MOVQ  dst_base+0(FP), DX
	MOVQ  x_base+24(FP), SI
	MOVQ  weight_base+48(FP), DI
	VBROADCASTSS scale+72(FP), Y0
	MOVQ  n+80(FP), CX

loop:
	CMPQ CX, $0
	JLE done
	VMOVUPS (SI), Y1
	VMULPS  Y0, Y1, Y1
	VMULPS  (DI), Y1, Y2
	VMOVUPS Y2, (DX)
	ADDQ $32, SI
	ADDQ $32, DI
	ADDQ $32, DX
	SUBQ $8, CX
	JMP loop

done:
	VZEROUPPER
	RET
