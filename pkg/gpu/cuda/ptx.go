//go:build cuda

package cuda

// kernelsPTX выбирает PTX matmul-ядер по compute capability GPU
func kernelsPTX(cc int) string {
	if cc >= 120 {
		return ptxHeader120 + matmulKernelsBody
	}

	return ptxHeader60 + matmulKernelsBody
}

// opsPTX выбирает PTX op-ядер (RMSNorm, ...)
func opsPTX(cc int) string {
	if cc >= 120 {
		return ptxHeader120 + opsKernelsBody
	}

	return ptxHeader60 + opsKernelsBody
}

const ptxHeader60 = `.version 7.0
.target sm_60
.address_size 64
`

const ptxHeader120 = `.version 8.0
.target sm_120
.address_size 64
`

const matmulKernelsBody = `
.visible .entry matmul_vec(
    .param .u64 param_matrix,
    .param .u64 param_vec,
    .param .u64 param_out,
    .param .u32 param_rows,
    .param .u32 param_cols
)
{
    .reg .pred      %p<2>;
    .reg .b32       %r<8>;
    .reg .b64       %rd<14>;
    .reg .f32       %f<4>;

    mov.u32         %r1, %tid.x;
    mov.u32         %r2, %ctaid.x;
    mov.u32         %r3, %ntid.x;
    mad.lo.u32      %r4, %r2, %r3, %r1;

    ld.param.u32    %r5, [param_rows];
    setp.ge.u32     %p0, %r4, %r5;
    @%p0            bra EXIT;

    ld.param.u64    %rd1, [param_matrix];
    ld.param.u64    %rd2, [param_vec];
    ld.param.u64    %rd3, [param_out];
    ld.param.u32    %r6, [param_cols];

    mov.f32         %f1, 0f00000000;
    mov.u32         %r7, 0;

LOOP:
    setp.ge.u32     %p1, %r7, %r6;
    @%p1            bra DONE;

    mul.wide.u32    %rd4, %r4, %r6;
    cvt.u64.u32     %rd12, %r7;
    add.u64         %rd5, %rd4, %rd12;
    shl.b64         %rd6, %rd5, 2;
    add.u64         %rd7, %rd1, %rd6;
    ld.global.f32   %f2, [%rd7];

    cvt.u64.u32     %rd8, %r7;
    shl.b64         %rd9, %rd8, 2;
    add.u64         %rd10, %rd2, %rd9;
    ld.global.f32   %f3, [%rd10];

    fma.rn.f32      %f1, %f2, %f3, %f1;

    add.u32         %r7, %r7, 1;
    bra LOOP;

DONE:
    cvt.u64.u32     %rd11, %r4;
    shl.b64         %rd12, %rd11, 2;
    add.u64         %rd13, %rd3, %rd12;
    st.global.f32   [%rd13], %f1;

EXIT:
    ret;
}

.visible .entry matmul_vec_q8_0(
    .param .u64 param_matrix,
    .param .u64 param_vec,
    .param .u64 param_out,
    .param .u32 param_rows,
    .param .u32 param_cols
)
{
    .reg .pred      %p<3>;
    .reg .b16       %h<1>;
    .reg .s8        %s<1>;
    .reg .b32       %r<12>;
    .reg .b64       %rd<16>;
    .reg .f32       %f<5>;

    mov.u32         %r1, %tid.x;
    mov.u32         %r2, %ctaid.x;
    mov.u32         %r3, %ntid.x;
    mad.lo.u32      %r4, %r2, %r3, %r1;

    ld.param.u32    %r5, [param_rows];
    setp.ge.u32     %p0, %r4, %r5;
    @%p0            bra Q8_EXIT;

    ld.param.u64    %rd1, [param_matrix];
    ld.param.u64    %rd2, [param_vec];
    ld.param.u64    %rd3, [param_out];
    ld.param.u32    %r6, [param_cols];

    shr.u32         %r7, %r6, 5;
    mov.f32         %f1, 0f00000000;
    mov.u32         %r8, 0;

Q8_BLOCK:
    setp.ge.u32     %p1, %r8, %r7;
    @%p1            bra Q8_DONE;

    mul.lo.u32      %r9, %r4, %r7;
    add.u32         %r9, %r9, %r8;
    mul.wide.u32    %rd4, %r9, 34;
    add.u64         %rd5, %rd1, %rd4;

    ld.global.b16   %h0, [%rd5];
    cvt.rn.f32.f16  %f2, %h0;

    mov.u32         %r11, 0;

Q8_INNER:
    setp.ge.u32     %p2, %r11, 32;
    @%p2            bra Q8_INNER_DONE;

    add.u64         %rd6, %rd5, 2;
    cvt.u64.u32     %rd7, %r11;
    add.u64         %rd8, %rd6, %rd7;
    ld.global.s8    %s0, [%rd8];
    cvt.s32.s8      %r12, %s0;
    cvt.rn.f32.s32  %f3, %r12;

    mul.lo.u32      %r9, %r8, 32;
    add.u32         %r9, %r9, %r11;
    mul.wide.u32    %rd9, %r9, 4;
    add.u64         %rd10, %rd2, %rd9;
    ld.global.f32   %f4, [%rd10];

    mul.f32         %f5, %f2, %f3;
    fma.rn.f32      %f1, %f5, %f4, %f1;

    add.u32         %r11, %r11, 1;
    bra Q8_INNER;

Q8_INNER_DONE:
    add.u32         %r8, %r8, 1;
    bra Q8_BLOCK;

Q8_DONE:
    cvt.u64.u32     %rd11, %r4;
    shl.b64         %rd12, %rd11, 2;
    add.u64         %rd13, %rd3, %rd12;
    st.global.f32   [%rd13], %f1;

Q8_EXIT:
    ret;
}
`

const opsKernelsBody = `
.visible .entry rmsnorm(
    .param .u64 param_x,
    .param .u64 param_weight,
    .param .u64 param_out,
    .param .u32 param_n,
    .param .f32 param_eps
)
{
    .reg .pred      %p<1>;
    .reg .b32       %r<4>;
    .reg .b64       %rd<10>;
    .reg .f32       %f<6>;

    mov.u32         %r1, %tid.x;
    setp.ne.u32     %p0, %r1, 0;
    @%p0            bra RN_EXIT;

    ld.param.u64    %rd1, [param_x];
    ld.param.u64    %rd2, [param_weight];
    ld.param.u64    %rd3, [param_out];
    ld.param.u32    %r2, [param_n];
    ld.param.f32    %f1, [param_eps];

    mov.f32         %f2, 0f00000000;
    mov.u32         %r3, 0;

RN_SUM:
    setp.ge.u32     %p0, %r3, %r2;
    @%p0            bra RN_SCALE;

    mul.wide.u32    %rd4, %r3, 4;
    add.u64         %rd5, %rd1, %rd4;
    ld.global.f32   %f3, [%rd5];
    mul.f32         %f3, %f3, %f3;
    add.f32         %f2, %f2, %f3;

    add.u32         %r3, %r3, 1;
    bra RN_SUM;

RN_SCALE:
    cvt.rn.f32.u32  %f4, %r2;
    div.rn.f32      %f4, %f2, %f4;
    add.f32         %f4, %f4, %f1;
    rsqrt.approx.f32 %f5, %f4;
    mov.u32         %r3, 0;

RN_OUT:
    setp.ge.u32     %p0, %r3, %r2;
    @%p0            bra RN_EXIT;

    mul.wide.u32    %rd4, %r3, 4;
    add.u64         %rd5, %rd1, %rd4;
    ld.global.f32   %f3, [%rd5];
    add.u64         %rd6, %rd2, %rd4;
    ld.global.f32   %f4, [%rd6];
    mul.f32         %f3, %f3, %f5;
    mul.f32         %f3, %f3, %f4;
    add.u64         %rd7, %rd3, %rd4;
    st.global.f32   [%rd7], %f3;

    add.u32         %r3, %r3, 1;
    bra RN_OUT;

RN_EXIT:
    ret;
}

.visible .entry rope_heads(
    .param .u64 param_v,
    .param .u64 param_cos,
    .param .u64 param_sin,
    .param .u32 param_nheads,
    .param .u32 param_headdim,
    .param .u32 param_half
)
{
    .reg .pred      %p<2>;
    .reg .b32       %r<16>;
    .reg .b64       %rd<16>;
    .reg .f32       %f<8>;

    mov.u32         %r1, %tid.x;
    mov.u32         %r2, %ctaid.x;
    mov.u32         %r3, %ntid.x;
    mad.lo.u32      %r4, %r2, %r3, %r1;

    ld.param.u32    %r5, [param_nheads];
    ld.param.u32    %r6, [param_headdim];
    ld.param.u32    %r7, [param_half];
    mul.lo.u32      %r8, %r5, %r7;

    setp.ge.u32     %p0, %r4, %r8;
    @%p0            bra RH_EXIT;

    ld.param.u64    %rd1, [param_v];
    ld.param.u64    %rd2, [param_cos];
    ld.param.u64    %rd3, [param_sin];

    div.u32         %r9, %r4, %r7;
    rem.u32         %r10, %r4, %r7;

    mul.lo.u32      %r11, %r9, %r6;
    add.u32         %r12, %r11, %r10;
    mul.lo.u32      %r13, %r12, 4;
    cvt.u64.u32     %rd4, %r13;
    add.u64         %rd5, %rd1, %rd4;
    ld.global.f32   %f1, [%rd5];

    add.u32         %r14, %r11, %r7;
    add.u32         %r14, %r14, %r10;
    mul.lo.u32      %r15, %r14, 4;
    cvt.u64.u32     %rd6, %r15;
    add.u64         %rd7, %rd1, %rd6;
    ld.global.f32   %f2, [%rd7];

    mul.wide.u32    %rd8, %r10, 4;
    add.u64         %rd9, %rd2, %rd8;
    ld.global.f32   %f3, [%rd9];
    add.u64         %rd10, %rd3, %rd8;
    ld.global.f32   %f4, [%rd10];

    mul.f32         %f5, %f1, %f3;
    mul.f32         %f6, %f2, %f4;
    sub.f32         %f5, %f5, %f6;
    st.global.f32   [%rd5], %f5;

    mul.f32         %f6, %f1, %f4;
    mul.f32         %f7, %f2, %f3;
    add.f32         %f6, %f6, %f7;
    st.global.f32   [%rd7], %f6;

RH_EXIT:
    ret;
}
`
