// NEON vecMulInPlace и addInPlace

//go:build arm64

#include "textflag.h"

// func vecMulInPlaceNEONAsm(a, b []float32, n int)
TEXT ·vecMulInPlaceNEONAsm(SB), NOSPLIT, $0-52
	MOVD  a_base+0(FP), R0
	MOVD  b_base+24(FP), R1
	MOVD  n+48(FP), R2

loop_mul:
	CMP   R2, $0
	BLE   done_mul
	VLD1  {V0.4S}, [R0]
	VLD1  {V1.4S}, [R1]
	VMUL  V0.S4, V1.S4, V0.S4
	VST1  {V0.4S}, [R0], #16
	ADD   $16, R1
	SUB   $4, R2
	B     loop_mul

done_mul:
	RET

// func addInPlaceNEONAsm(a, b []float32, n int)
TEXT ·addInPlaceNEONAsm(SB), NOSPLIT, $0-52
	MOVD  a_base+0(FP), R0
	MOVD  b_base+24(FP), R1
	MOVD  n+48(FP), R2

loop_add:
	CMP   R2, $0
	BLE   done_add
	VLD1  {V0.4S}, [R0]
	VLD1  {V1.4S}, [R1]
	VADD  V0.S4, V1.S4, V0.S4
	VST1  {V0.4S}, [R0], #16
	ADD   $16, R1
	SUB   $4, R2
	B     loop_add

done_add:
	RET

// func vectorMaxNEONAsm(x []float32, n int) float32
TEXT ·vectorMaxNEONAsm(SB), NOSPLIT, $0-36
	MOVD  x_base+0(FP), R0
	MOVD  n+24(FP), R2
	VLD1  {V0.4S}, [R0], #16
	SUB   $4, R2

loop_max:
	CMP   R2, $0
	BLE   reduce_max
	VLD1  {V1.4S}, [R0], #16
	VMAX  V0.S4, V1.S4, V0.S4
	SUB   $4, R2
	B     loop_max

reduce_max:
	FMAX  V0.S4, V0.S4, V0.S2
	FMAX  V0.S2, V0.S2, V0.S1
	VMOV  V0.S[0], ret+32(FP)
	RET

// func vecScaleInPlaceNEONAsm(x []float32, scale float32, n int)
TEXT ·vecScaleInPlaceNEONAsm(SB), NOSPLIT, $0-36
	MOVD  x_base+0(FP), R0
	VMOV  scale+24(FP), V0.S[0]
	DUP   V0.S[0], V0.S4
	MOVD  n+32(FP), R2

loop_scale:
	CMP   R2, $0
	BLE   done_scale
	VLD1  {V1.4S}, [R0]
	VMUL  V1.S4, V0.S4, V1.S4
	VST1  {V1.4S}, [R0], #16
	SUB   $4, R2
	B     loop_scale

done_scale:
	RET
