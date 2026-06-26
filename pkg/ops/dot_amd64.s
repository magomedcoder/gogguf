// AVX2 dot product и проверка CPUID для amd64

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

// func dotAVX2Asm(a []float32, b []float32, n int) float32
// n кратно 8
TEXT ·dotAVX2Asm(SB), NOSPLIT, $0-28
	MOVQ  a_base+0(FP), SI
	MOVQ  b_base+24(FP), DI
	MOVQ  n+48(FP), CX
	VXORPS Y0, Y0, Y0

loop:
	CMPQ CX, $0
	JLE done
	VMOVUPS (SI), Y1
	VFMADD231PS (DI), Y1, Y0
	ADDQ $32, SI
	ADDQ $32, DI
	SUBQ $8, CX
	JMP loop

done:
	VEXTRACTF128 $1, Y0, X1
	VADDPS       X0, X1, X0
	VSHUFPS      $0x4E, X0, X0, X1
	VADDPS       X0, X1, X0
	VSHUFPS      $0xB1, X0, X0, X1
	VADDSS       X0, X1, X0
	MOVSS        X0, ret+56(FP)
	VZEROUPPER
	RET
