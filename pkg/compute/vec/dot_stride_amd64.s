// AVX2 strided weighted row sum for amd64

//go:build amd64

#include "textflag.h"

// func dotStrideBlock32AVX2Asm(dst, w, v *float32, stride, n, d int)
// d is a multiple of 32; dst[i] = sum_t w[t]*v[t*stride+i]
TEXT ·dotStrideBlock32AVX2Asm(SB), NOSPLIT, $0-48
	MOVQ  dst+0(FP), R10
	MOVQ  w+8(FP), R11
	MOVQ  v+16(FP), R12
	MOVQ  stride+24(FP), R8
	MOVQ  n+32(FP), R13
	MOVQ  d+40(FP), R14
	SHLQ  $2, R8              // stride in bytes

outer32:
	CMPQ R14, $0
	JLE  done32
	VXORPS Y0, Y0, Y0
	VXORPS Y1, Y1, Y1
	VXORPS Y2, Y2, Y2
	VXORPS Y3, Y3, Y3
	MOVQ R11, SI
	MOVQ R12, DI
	MOVQ R13, CX

inner32:
	VBROADCASTSS (SI), Y4
	VFMADD231PS (DI), Y4, Y0
	VFMADD231PS 32(DI), Y4, Y1
	VFMADD231PS 64(DI), Y4, Y2
	VFMADD231PS 96(DI), Y4, Y3
	ADDQ $4, SI
	ADDQ R8, DI
	DECQ CX
	JNZ  inner32

	VMOVUPS Y0, (R10)
	VMOVUPS Y1, 32(R10)
	VMOVUPS Y2, 64(R10)
	VMOVUPS Y3, 96(R10)
	ADDQ $128, R10
	ADDQ $128, R12
	SUBQ $32, R14
	JMP  outer32

done32:
	VZEROUPPER
	RET

// func dotStrideBlock8AVX2Asm(dst, w, v *float32, stride, n, d int)
// d is a multiple of 8
TEXT ·dotStrideBlock8AVX2Asm(SB), NOSPLIT, $0-48
	MOVQ  dst+0(FP), R10
	MOVQ  w+8(FP), R11
	MOVQ  v+16(FP), R12
	MOVQ  stride+24(FP), R8
	MOVQ  n+32(FP), R13
	MOVQ  d+40(FP), R14
	SHLQ  $2, R8

outer8:
	CMPQ R14, $0
	JLE  done8
	VXORPS Y0, Y0, Y0
	VXORPS Y1, Y1, Y1
	MOVQ R11, SI
	MOVQ R12, DI
	MOVQ R13, CX

inner8:
	VBROADCASTSS (SI), Y4
	VFMADD231PS (DI), Y4, Y0
	ADDQ $4, SI
	ADDQ R8, DI
	DECQ CX
	JNZ  inner8

	VMOVUPS Y0, (R10)
	ADDQ $32, R10
	ADDQ $32, R12
	SUBQ $8, R14
	JMP  outer8

done8:
	VZEROUPPER
	RET
