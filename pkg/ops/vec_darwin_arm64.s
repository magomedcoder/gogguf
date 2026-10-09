// NEON vecMulInPlace and addInPlace

//go:build arm64 && darwin

#include "textflag.h"

// func vecMulInPlaceNEONAsm(a, b []float32, n int)
TEXT ·vecMulInPlaceNEONAsm(SB), NOSPLIT, $0-56
	MOVD	a_base+0(FP), R0
	MOVD	b_base+24(FP), R1
	MOVD	n+48(FP), R2

loop_mul:
	CMP	ZR, R2
	BLE	done_mul
	VLD1	(R0), [V0.S4]
	VLD1	(R1), [V1.S4]
	VFMUL	V0.S4, V1.S4, V0.S4
	VST1.P	[V0.S4], 16(R0)
	ADD	$16, R1
	SUB	$4, R2
	B	loop_mul

done_mul:
	RET

// func addInPlaceNEONAsm(a, b []float32, n int)
TEXT ·addInPlaceNEONAsm(SB), NOSPLIT, $0-56
	MOVD	a_base+0(FP), R0
	MOVD	b_base+24(FP), R1
	MOVD	n+48(FP), R2

loop_add:
	CMP	ZR, R2
	BLE	done_add
	VLD1	(R0), [V0.S4]
	VLD1	(R1), [V1.S4]
	VFADD	V0.S4, V1.S4, V0.S4
	VST1.P	[V0.S4], 16(R0)
	ADD	$16, R1
	SUB	$4, R2
	B	loop_add

done_add:
	RET

// func vectorMaxNEONAsm(x []float32, n int) float32
TEXT ·vectorMaxNEONAsm(SB), NOSPLIT, $0-36
	MOVD	x_base+0(FP), R0
	MOVD	n+24(FP), R2
	VLD1.P	16(R0), [V0.S4]
	SUB	$4, R2

loop_max:
	CMP	ZR, R2
	BLE	done_max
	VLD1.P	16(R0), [V1.S4]
	VFMAX	V0.S4, V1.S4, V0.S4
	SUB	$4, R2
	B	loop_max

done_max:
	VFMAXV	V0.S4, V0
	FMOVS	F0, ret+32(FP)
	RET

// func vecScaleInPlaceNEONAsm(x []float32, scale float32, n int)
TEXT ·vecScaleInPlaceNEONAsm(SB), NOSPLIT, $0-40
	MOVD	x_base+0(FP), R0
	MOVD	scale+24(FP), R9
	VMOV	R9, V0.S[0]
	VDUP	V0.S[0], V0.S4
	MOVD	n+32(FP), R2

loop_scale:
	CMP	ZR, R2
	BLE	done_scale
	VLD1	(R0), [V1.S4]
	VFMUL	V1.S4, V0.S4, V1.S4
	VST1.P	[V1.S4], 16(R0)
	SUB	$4, R2
	B	loop_scale

done_scale:
	RET
