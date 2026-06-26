// NEON dot product для arm64

//go:build arm64

#include "textflag.h"

// func dotNEONAsm(a []float32, b []float32, n int) float32
// n кратно 4
TEXT ·dotNEONAsm(SB), NOSPLIT, $0-28
	MOVD  a_base+0(FP), R0
	MOVD  b_base+24(FP), R1
	MOVD  n+48(FP), R2
	FMOVD ZR, V0

loop:
	CMP   R2, $0
	BLE   done
	VLD1.P {V1.4S}, [R0], #16
	VLD1.P {V2.4S}, [R1], #16
	VFMLA V0.S4, V1.S4, V2.S4
	SUB   $4, R2
	B     loop

done:
	FADDP V0.S4, V0.S4, V0.S2
	FADDP V0.S2, V0.S2, V0.S1
	VMOV  V0.S[0], ret+56(FP)
	RET
