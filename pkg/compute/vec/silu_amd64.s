// AVX2 SiLU: x / (1 + exp(-x)) with a Cephes-style expf polynomial (rel. err ~1e-7)

//go:build amd64

#include "textflag.h"

// float32 constants (bit patterns)
DATA siluConst<>+0(SB)/4, $0x3fb8aa3b   // log2(e)
DATA siluConst<>+4(SB)/4, $0x3f318000   // ln2 high
DATA siluConst<>+8(SB)/4, $0xb95e8083   // ln2 low
DATA siluConst<>+12(SB)/4, $0x39506967  // p0
DATA siluConst<>+16(SB)/4, $0x3ab743ce  // p1
DATA siluConst<>+20(SB)/4, $0x3c088908  // p2
DATA siluConst<>+24(SB)/4, $0x3d2aa9c1  // p3
DATA siluConst<>+28(SB)/4, $0x3e2aaaaa  // p4
DATA siluConst<>+32(SB)/4, $0x3f000000  // p5
DATA siluConst<>+36(SB)/4, $0x3f800000  // 1.0
DATA siluConst<>+40(SB)/4, $0x42b00000  // 88.0
DATA siluConst<>+44(SB)/4, $0xc2b00000  // -88.0
DATA siluConst<>+48(SB)/4, $0x80000000  // sign mask
DATA siluConst<>+52(SB)/4, $127         // exponent bias
GLOBL siluConst<>(SB), RODATA|NOPTR, $56

// func siluInPlaceAVX2Asm(x []float32, n int)
// n is a multiple of 8
TEXT ·siluInPlaceAVX2Asm(SB), NOSPLIT, $0-32
	MOVQ  x_base+0(FP), SI
	MOVQ  n+24(FP), CX
	LEAQ  siluConst<>(SB), DX

	VBROADCASTSS 0(DX), Y8     // log2e
	VBROADCASTSS 4(DX), Y9     // ln2 hi
	VBROADCASTSS 8(DX), Y10    // ln2 lo
	VBROADCASTSS 36(DX), Y11   // 1
	VBROADCASTSS 40(DX), Y12   // +88
	VBROADCASTSS 44(DX), Y13   // -88
	VBROADCASTSS 48(DX), Y14   // sign
	VBROADCASTSS 52(DX), Y15   // 127

loop:
	CMPQ CX, $0
	JLE  done
	VMOVUPS (SI), Y0           // x
	VXORPS  Y14, Y0, Y1        // t = -x
	VMINPS  Y12, Y1, Y1
	VMAXPS  Y13, Y1, Y1        // clamp t to [-88, 88]
	VMULPS  Y8, Y1, Y2         // t*log2e
	VCVTPS2DQ Y2, Y3           // n = round(t*log2e)
	VCVTDQ2PS Y3, Y4           // float(n)
	VFNMADD231PS Y4, Y9, Y1    // r = t - n*ln2hi
	VFNMADD231PS Y4, Y10, Y1   // r -= n*ln2lo
	VBROADCASTSS 12(DX), Y5    // p0
	VBROADCASTSS 16(DX), Y6
	VFMADD213PS Y6, Y1, Y5     // y = y*r + p1
	VBROADCASTSS 20(DX), Y6
	VFMADD213PS Y6, Y1, Y5
	VBROADCASTSS 24(DX), Y6
	VFMADD213PS Y6, Y1, Y5
	VBROADCASTSS 28(DX), Y6
	VFMADD213PS Y6, Y1, Y5
	VBROADCASTSS 32(DX), Y6
	VFMADD213PS Y6, Y1, Y5     // y = y*r + p5
	VMULPS  Y1, Y1, Y6         // r*r
	VFMADD213PS Y1, Y6, Y5     // y = y*r^2 + r
	VADDPS  Y11, Y5, Y5        // y += 1
	VPADDD  Y15, Y3, Y3        // n + 127
	VPSLLD  $23, Y3, Y3        // 2^n
	VMULPS  Y3, Y5, Y5         // exp(t)
	VADDPS  Y11, Y5, Y5        // 1 + exp(t)
	VDIVPS  Y5, Y0, Y0         // x / (1 + exp(-x))
	VMOVUPS Y0, (SI)
	ADDQ $32, SI
	SUBQ $8, CX
	JMP  loop

done:
	VZEROUPPER
	RET
