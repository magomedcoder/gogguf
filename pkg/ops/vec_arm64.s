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
