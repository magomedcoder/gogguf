// AVX2 vecMulInPlace и addInPlace

//go:build amd64

#include "textflag.h"

// func vecMulInPlaceAVX2Asm(a, b []float32, n int)
TEXT ·vecMulInPlaceAVX2Asm(SB), NOSPLIT, $0-52
	MOVQ  a_base+0(FP), SI
	MOVQ  b_base+24(FP), DI
	MOVQ  n+48(FP), CX

loop_mul:
	CMPQ CX, $0
	JLE done_mul
	VMOVUPS (SI), Y0
	VMULPS  (DI), Y0, Y0
	VMOVUPS Y0, (SI)
	ADDQ $32, SI
	ADDQ $32, DI
	SUBQ $8, CX
	JMP loop_mul

done_mul:
	VZEROUPPER
	RET

// func addInPlaceAVX2Asm(a, b []float32, n int)
TEXT ·addInPlaceAVX2Asm(SB), NOSPLIT, $0-52
	MOVQ  a_base+0(FP), SI
	MOVQ  b_base+24(FP), DI
	MOVQ  n+48(FP), CX

loop_add:
	CMPQ CX, $0
	JLE done_add
	VMOVUPS (SI), Y0
	VADDPS  (DI), Y0, Y0
	VMOVUPS Y0, (SI)
	ADDQ $32, SI
	ADDQ $32, DI
	SUBQ $8, CX
	JMP loop_add

done_add:
	VZEROUPPER
	RET

// func vectorMaxAVX2Asm(x []float32, n int) float32
TEXT ·vectorMaxAVX2Asm(SB), NOSPLIT, $0-36
	MOVQ  x_base+0(FP), SI
	MOVQ  n+24(FP), CX
	VMOVUPS (SI), Y0
	SUBQ  $8, CX
	ADDQ  $32, SI

loop_max:
	CMPQ CX, $0
	JLE reduce_max
	VMOVUPS (SI), Y1
	VMAXPS Y1, Y0, Y0
	ADDQ $32, SI
	SUBQ $8, CX
	JMP loop_max

reduce_max:
	VEXTRACTF128 $1, Y0, X1
	VMAXPS       X0, X1, X0
	VSHUFPS      $0x4E, X0, X0, X1
	VMAXPS       X0, X1, X0
	VSHUFPS      $0xB1, X0, X0, X1
	VMAXSS       X1, X0, X0
	MOVSS        X0, ret+32(FP)
	VZEROUPPER
	RET

// func vecScaleInPlaceAVX2Asm(x []float32, scale float32, n int)
TEXT ·vecScaleInPlaceAVX2Asm(SB), NOSPLIT, $0-36
	MOVQ  x_base+0(FP), SI
	VBROADCASTSS scale+24(FP), Y0
	MOVQ  n+32(FP), CX

loop_scale:
	CMPQ CX, $0
	JLE done_scale
	VMOVUPS (SI), Y1
	VMULPS  Y0, Y1, Y1
	VMOVUPS Y1, (SI)
	ADDQ $32, SI
	SUBQ $8, CX
	JMP loop_scale

done_scale:
	VZEROUPPER
	RET
