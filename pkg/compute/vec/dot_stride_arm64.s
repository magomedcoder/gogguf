// NEON strided weighted row sum for arm64

//go:build arm64

#include "textflag.h"

// func dotStrideBlock16NEONAsm(dst, w, v *float32, stride, n, d int)
// d is a multiple of 16; dst[i] = sum_t w[t]*v[t*stride+i]
TEXT ·dotStrideBlock16NEONAsm(SB), NOSPLIT, $0-48
	MOVD	dst+0(FP), R0
	MOVD	w+8(FP), R1
	MOVD	v+16(FP), R2
	MOVD	stride+24(FP), R3
	MOVD	n+32(FP), R4
	MOVD	d+40(FP), R5
	LSL	$2, R3, R3

outer16:
	CMP	ZR, R5
	BLE	done16
	VEOR	V0.B16, V0.B16, V0.B16
	VEOR	V1.B16, V1.B16, V1.B16
	VEOR	V2.B16, V2.B16, V2.B16
	VEOR	V3.B16, V3.B16, V3.B16
	MOVD	R1, R6
	MOVD	R2, R7
	MOVD	R4, R8

inner16:
	VLD1R.P	4(R6), [V4.S4]
	VLD1	(R7), [V16.S4, V17.S4, V18.S4, V19.S4]
	VFMLA	V16.S4, V4.S4, V0.S4
	VFMLA	V17.S4, V4.S4, V1.S4
	VFMLA	V18.S4, V4.S4, V2.S4
	VFMLA	V19.S4, V4.S4, V3.S4
	ADD	R3, R7
	SUB	$1, R8
	CBNZ	R8, inner16

	VST1.P	[V0.S4, V1.S4, V2.S4, V3.S4], 64(R0)
	ADD	$64, R2
	SUB	$16, R5
	B	outer16

done16:
	RET

// func dotStrideBlock4NEONAsm(dst, w, v *float32, stride, n, d int)
// d is a multiple of 4
TEXT ·dotStrideBlock4NEONAsm(SB), NOSPLIT, $0-48
	MOVD	dst+0(FP), R0
	MOVD	w+8(FP), R1
	MOVD	v+16(FP), R2
	MOVD	stride+24(FP), R3
	MOVD	n+32(FP), R4
	MOVD	d+40(FP), R5
	LSL	$2, R3, R3

outer4:
	CMP	ZR, R5
	BLE	done4
	VEOR	V0.B16, V0.B16, V0.B16
	MOVD	R1, R6
	MOVD	R2, R7
	MOVD	R4, R8

inner4:
	VLD1R.P	4(R6), [V4.S4]
	VLD1	(R7), [V16.S4]
	VFMLA	V16.S4, V4.S4, V0.S4
	ADD	R3, R7
	SUB	$1, R8
	CBNZ	R8, inner4

	VST1.P	[V0.S4], 16(R0)
	ADD	$16, R2
	SUB	$4, R5
	B	outer4

done4:
	RET
