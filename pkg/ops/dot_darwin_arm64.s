// NEON dot product for arm64 (Go assembly syntax; portable across linux/darwin arm64)

//go:build arm64 && darwin

#include "textflag.h"

// func dotNEONAsm(a []float32, b []float32, n int) float32
// n кратно 4
TEXT ·dotNEONAsm(SB), NOSPLIT, $0-60
	MOVD	a_base+0(FP), R0
	MOVD	b_base+24(FP), R1
	MOVD	n+48(FP), R2
	VMOV	ZR, V0.D[0]
	VMOV	ZR, V0.D[1]

loop:
	CMP	ZR, R2
	BLE	done
	VLD1.P	16(R0), [V1.S4]
	VLD1.P	16(R1), [V2.S4]
	VFMLA	V1.S4, V2.S4, V0.S4
	SUB	$4, R2
	B	loop

done:
	// reduce the 4 accumulator lanes of V0 to a scalar sum in F4
	VMOV	V0.S[0], R3
	VMOV	V0.S[1], R4
	VMOV	V0.S[2], R5
	VMOV	V0.S[3], R6
	FMOVS	R3, F4
	FMOVS	R4, F5
	FMOVS	R5, F6
	FMOVS	R6, F7
	FADDS	F4, F5, F4
	FADDS	F6, F7, F6
	FADDS	F4, F6, F4
	FMOVS	F4, ret+56(FP)
	RET
