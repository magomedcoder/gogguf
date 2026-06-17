//go:build cuda

package cuda

// matmulVecPTX - PTX kernel matmul_vec (FP32 matrix * vector)
const matmulVecPTX = `.version 7.0
.target sm_52
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
}`
