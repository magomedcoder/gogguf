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
