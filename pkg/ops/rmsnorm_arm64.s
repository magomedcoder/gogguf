// NEON scale-mul для RMSNorm: dst[i] = x[i] * scale * weight[i]

//go:build arm64

#include "textflag.h"

// func rmsnormScaleMulNEONAsm(dst, x, weight []float32, scale float32, n int)
// n кратно 4
TEXT ·rmsnormScaleMulNEONAsm(SB), NOSPLIT, $0-84
	MOVD  dst_base+0(FP), R3
	MOVD  x_base+24(FP), R0
	MOVD  weight_base+48(FP), R1
	VMOV  scale+72(FP), V0.S[0]
	MOVD  n+80(FP), R2
	DUP   V0.S[0], V0.S4

loop:
	CMP   R2, $0
	BLE   done
	VLD1.P {V1.4S}, [R0], #16
	VLD1.P {V2.4S}, [R1], #16
	VMUL  V1.S4, V0.S4, V1.S4
	VMUL  V2.S4, V1.S4, V1.S4
	VST1.P {V1.4S}, [R3], #16
	SUB   $4, R2
	B     loop

done:
	RET
