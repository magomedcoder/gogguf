//go:build cuda

package cuda

// kernelsPTX — FP32 matmul_vec и Q8_0 matmul_vec_q8_0 (sm_70+, f16 scale)
const kernelsPTX = `.version 7.0
.target sm_70
.address_size 64

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

    ld.global.u16   %r10, [%rd5];
    mov.b16         %h0, %r10;
    cvt.rn.f32.f16  %f2, %h0;

    mov.u32         %r11, 0;

Q8_INNER:
    setp.ge.u32     %p2, %r11, 32;
    @%p2            bra Q8_INNER_DONE;

    add.u64         %rd6, %rd5, 2;
    cvt.u64.u32     %rd7, %r11;
    add.u64         %rd8, %rd6, %rd7;
    ld.global.s8    %r12, [%rd8];
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
