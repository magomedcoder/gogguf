// AVX2 dot product для одного Q8_0-блока (32*int8 * float32)

//go:build amd64

#include "textflag.h"

// func cpuHasAVX2() bool
TEXT ·cpuHasAVX2(SB), NOSPLIT, $0-1
	MOVQ $0, AX
	CPUID
	MOVQ $1, AX
	MOVQ $0, CX
	CPUID
	MOVL CX, DX
	ANDL $0x18000000, DX
	CMPL DX, $0x18000000
	JNE   noavx2
	MOVQ  $7, AX
	MOVQ  $0, CX
	CPUID
	TESTL $32, BX
	JZ    noavx2
	MOVB  $1, ret+0(FP)
	RET
noavx2:
	MOVB  $0, ret+0(FP)
	RET

// func dotBlockQ8_0AVX2Asm(d float32, q *byte, x *float32) float32
TEXT ·dotBlockQ8_0AVX2Asm(SB), NOSPLIT, $0-28
	MOVSS   d+0(FP), X0
	VBROADCASTSS X0, Y7

	MOVQ    q+8(FP), SI
	MOVQ    x+16(FP), DI

	VXORPS  Y6, Y6, Y6

	// 4* по 8 элементов
	MOVQ    $4, CX
chunk:
	VPMOVSXBD 0(SI), X1
	VPMOVSXBD 4(SI), X2
	VINSERTF128 $1, X2, Y1, Y1
	VCVTDQ2PS Y1, Y1
	VMULPS  Y1, Y7, Y1
	VMOVUPS (DI), Y2
	VFMADD231PS Y1, Y2, Y6

	ADDQ    $8, SI
	ADDQ    $32, DI
	SUBQ    $1, CX
	JNZ     chunk

	// horizontal sum Y6 (8 floats)
	VEXTRACTF128 $1, Y6, X1
	VADDPS       X6, X1, X6
	VSHUFPS      $0x4E, X6, X6, X1
	VADDPS       X6, X1, X6
	VSHUFPS      $0xB1, X6, X6, X1
	VADDSS       X6, X1, X6
	MOVSS        X6, ret+24(FP)
	VZEROUPPER
	RET
