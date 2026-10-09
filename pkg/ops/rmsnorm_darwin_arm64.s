// NEON scale-mul для RMSNorm: dst[i] = x[i] * scale * weight[i]

//go:build arm64 && darwin

#include "textflag.h"

// func rmsnormScaleMulNEONAsm(dst, x, weight []float32, scale float32, n int)
// n кратно 4
TEXT ·rmsnormScaleMulNEONAsm(SB), NOSPLIT, $0-88
	MOVD	dst_base+0(FP), R3
	MOVD	x_base+24(FP), R0
	MOVD	weight_base+48(FP), R1
	MOVD	scale+72(FP), R9
	VMOV	R9, V0.S[0]
	VDUP	V0.S[0], V0.S4
	MOVD	n+80(FP), R2

loop:
	CMP	ZR, R2
	BLE	done
	VLD1	(R0), [V1.S4]
	VLD1	(R1), [V2.S4]
	VFMUL	V1.S4, V0.S4, V1.S4
	VFMUL	V2.S4, V1.S4, V2.S4
	VST1.P	[V2.S4], 16(R3)
	ADD	$16, R0
	ADD	$16, R1
	SUB	$4, R2
	B	loop

done:
	RET
