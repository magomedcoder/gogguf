//go:build cuda

package cuda

// ptxTargets returns list of SM targets to try for JIT
func ptxTargets(cc int) []int {
	switch {
	case cc >= 120:
		return []int{120, 90, 75, 60}
	case cc >= 90:
		return []int{90, 75, 60}
	case cc >= 75:
		return []int{75, 60}
	default:
		return []int{60}
	}
}

func kernelsPTXForTarget(target int) string {
	return ptxHeaderForTarget(target) + matmulKernelsBody
}

func opsPTXForTarget(target int) string {
	return ptxHeaderForTarget(target) + opsKernelsBody
}

// kernelsPTX selects PTX matmul kernels by GPU compute capability
func kernelsPTX(cc int) string {
	targets := ptxTargets(cc)
	return kernelsPTXForTarget(targets[0])
}

// opsPTX selects PTX op kernels (RMSNorm, ...)
func opsPTX(cc int) string {
	targets := ptxTargets(cc)
	return opsPTXForTarget(targets[0])
}

func ptxHeaderForTarget(target int) string {
	switch {
	case target >= 120:
		// Blackwell (sm_120) requires PTX >= 8.7
		return `.version 8.7
.target sm_120
.address_size 64
`
	case target >= 90:
		return `.version 8.0
.target sm_90
.address_size 64
`
	case target >= 75:
		return `.version 7.4
.target sm_75
.address_size 64
`
	default:
		return ptxHeader60
	}
}

const ptxHeader60 = `.version 7.0
.target sm_60
.address_size 64
`

const matmulVecKernel = `
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

`

const matmulQ8Kernel = `
.visible .entry matmul_vec_q8_0(
    .param .u64 param_matrix,
    .param .u64 param_vec,
    .param .u64 param_out,
    .param .u32 param_rows,
    .param .u32 param_cols
)
{
    .reg .pred      %p<3>;
    .reg .s8        %s<1>;
    .reg .b32       %r<13>;
    .reg .b64       %rd<20>;
    .reg .f32       %f<6>;

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
    mul.wide.u32    %rd4, %r9, 36;
    add.u64         %rd5, %rd1, %rd4;

    ld.global.f32   %f2, [%rd5];

    mov.u32         %r11, 0;

Q8_INNER:
    setp.ge.u32     %p2, %r11, 32;
    @%p2            bra Q8_INNER_DONE;

    add.u64         %rd6, %rd5, 4;
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

// matmul_mat_q8_0: W[rows,cols] * X[batch,cols] -> out[batch,rows]; tid = b*rows+r
const matmulMatQ8Kernel = `
.visible .entry matmul_mat_q8_0(
    .param .u64 param_matrix,
    .param .u64 param_x,
    .param .u64 param_out,
    .param .u32 param_rows,
    .param .u32 param_cols,
    .param .u32 param_batch
)
{
    .reg .pred      %p<3>;
    .reg .s8        %s<1>;
    .reg .b32       %r<18>;
    .reg .b64       %rd<24>;
    .reg .f32       %f<6>;

    mov.u32         %r1, %tid.x;
    mov.u32         %r2, %ctaid.x;
    mov.u32         %r3, %ntid.x;
    mad.lo.u32      %r10, %r2, %r3, %r1;

    ld.param.u32    %r5, [param_rows];
    ld.param.u32    %r14, [param_batch];
    mul.lo.u32      %r15, %r5, %r14;
    setp.ge.u32     %p0, %r10, %r15;
    @%p0            bra MQ8_EXIT;

    rem.u32         %r4, %r10, %r5;
    div.u32         %r13, %r10, %r5;

    ld.param.u64    %rd1, [param_matrix];
    ld.param.u64    %rd2, [param_x];
    ld.param.u64    %rd3, [param_out];
    ld.param.u32    %r6, [param_cols];

    mul.wide.u32    %rd14, %r13, %r6;
    shl.b64         %rd14, %rd14, 2;
    add.u64         %rd2, %rd2, %rd14;

    shr.u32         %r7, %r6, 5;
    mov.f32         %f1, 0f00000000;
    mov.u32         %r8, 0;

MQ8_BLOCK:
    setp.ge.u32     %p1, %r8, %r7;
    @%p1            bra MQ8_DONE;

    mul.lo.u32      %r9, %r4, %r7;
    add.u32         %r9, %r9, %r8;
    mul.wide.u32    %rd4, %r9, 36;
    add.u64         %rd5, %rd1, %rd4;

    ld.global.f32   %f2, [%rd5];

    mov.u32         %r11, 0;

MQ8_INNER:
    setp.ge.u32     %p2, %r11, 32;
    @%p2            bra MQ8_INNER_DONE;

    add.u64         %rd6, %rd5, 4;
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
    bra MQ8_INNER;

MQ8_INNER_DONE:
    add.u32         %r8, %r8, 1;
    bra MQ8_BLOCK;

MQ8_DONE:
    mul.lo.u32      %r16, %r13, %r5;
    add.u32         %r16, %r16, %r4;
    cvt.u64.u32     %rd11, %r16;
    shl.b64         %rd12, %rd11, 2;
    add.u64         %rd13, %rd3, %rd12;
    st.global.f32   [%rd13], %f1;

MQ8_EXIT:
    ret;
}
`

const matmulQ4Kernel = `
.visible .entry matmul_vec_q4_0(
    .param .u64 param_matrix,
    .param .u64 param_vec,
    .param .u64 param_out,
    .param .u32 param_rows,
    .param .u32 param_cols
)
{
    .reg .pred      %p<4>;
    .reg .b32       %r<16>;
    .reg .b64       %rd<20>;
    .reg .f32       %f<6>;

    mov.u32         %r1, %tid.x;
    mov.u32         %r2, %ctaid.x;
    mov.u32         %r3, %ntid.x;
    mad.lo.u32      %r4, %r2, %r3, %r1;

    ld.param.u32    %r5, [param_rows];
    setp.ge.u32     %p0, %r4, %r5;
    @%p0            bra Q4_EXIT;

    ld.param.u64    %rd1, [param_matrix];
    ld.param.u64    %rd2, [param_vec];
    ld.param.u64    %rd3, [param_out];
    ld.param.u32    %r6, [param_cols];

    shr.u32         %r7, %r6, 5;
    mov.f32         %f1, 0f00000000;
    mov.u32         %r8, 0;

Q4_BLOCK:
    setp.ge.u32     %p1, %r8, %r7;
    @%p1            bra Q4_DONE;

    mul.lo.u32      %r9, %r4, %r7;
    add.u32         %r9, %r9, %r8;
    mul.wide.u32    %rd4, %r9, 20;
    add.u64         %rd5, %rd1, %rd4;

    ld.global.f32   %f2, [%rd5];

    mov.u32         %r11, 0;

Q4_INNER:
    setp.ge.u32     %p2, %r11, 32;
    @%p2            bra Q4_INNER_DONE;

    shr.u32         %r12, %r11, 1;
    add.u64         %rd6, %rd5, 4;
    cvt.u64.u32     %rd7, %r12;
    add.u64         %rd8, %rd6, %rd7;
    ld.global.u8    %r13, [%rd8];

    and.b32         %r14, %r11, 1;
    setp.ne.u32     %p3, %r14, 0;
    @%p3            bra Q4_HIGH;
    and.b32         %r15, %r13, 15;
    bra Q4_HAVE_Q;

Q4_HIGH:
    shr.u32         %r15, %r13, 4;

Q4_HAVE_Q:
    sub.s32         %r15, %r15, 8;
    cvt.rn.f32.s32  %f3, %r15;

    mul.lo.u32      %r9, %r8, 32;
    add.u32         %r9, %r9, %r11;
    mul.wide.u32    %rd9, %r9, 4;
    add.u64         %rd10, %rd2, %rd9;
    ld.global.f32   %f4, [%rd10];

    mul.f32         %f5, %f2, %f3;
    fma.rn.f32      %f1, %f5, %f4, %f1;

    add.u32         %r11, %r11, 1;
    bra Q4_INNER;

Q4_INNER_DONE:
    add.u32         %r8, %r8, 1;
    bra Q4_BLOCK;

Q4_DONE:
    cvt.u64.u32     %rd11, %r4;
    shl.b64         %rd12, %rd11, 2;
    add.u64         %rd13, %rd3, %rd12;
    st.global.f32   [%rd13], %f1;

Q4_EXIT:
    ret;
}
`

const matmulQ4KKernel = `
.visible .entry matmul_vec_q4_k(
    .param .u64 param_matrix,
    .param .u64 param_vec,
    .param .u64 param_out,
    .param .u32 param_rows,
    .param .u32 param_cols
)
{
    .reg .pred      %p<5>;
    .reg .b32       %r<20>;
    .reg .b64       %rd<24>;
    .reg .f32       %f<8>;

    mov.u32         %r1, %tid.x;
    mov.u32         %r2, %ctaid.x;
    mov.u32         %r3, %ntid.x;
    mad.lo.u32      %r4, %r2, %r3, %r1;

    ld.param.u32    %r5, [param_rows];
    setp.ge.u32     %p0, %r4, %r5;
    @%p0            bra Q4K_EXIT;

    ld.param.u64    %rd1, [param_matrix];
    ld.param.u64    %rd2, [param_vec];
    ld.param.u64    %rd3, [param_out];
    ld.param.u32    %r6, [param_cols];

    shr.u32         %r7, %r6, 8;
    mov.f32         %f1, 0f00000000;
    mov.u32         %r8, 0;

Q4K_BLOCK:
    setp.ge.u32     %p1, %r8, %r7;
    @%p1            bra Q4K_DONE;

    mul.lo.u32      %r9, %r4, %r7;
    add.u32         %r9, %r9, %r8;
    mul.wide.u32    %rd4, %r9, 192;
    add.u64         %rd5, %rd1, %rd4;

    mov.u32         %r10, 0;

Q4K_GROUP:
    setp.ge.u32     %p2, %r10, 8;
    @%p2            bra Q4K_GROUP_DONE;

    mul.wide.u32    %rd6, %r10, 4;
    add.u64         %rd7, %rd5, %rd6;
    ld.global.f32   %f2, [%rd7];
    add.u64         %rd8, %rd7, 32;
    ld.global.f32   %f3, [%rd8];

    shr.u32         %r11, %r10, 1;
    mul.lo.u32      %r12, %r11, 32;
    add.u64         %rd9, %rd5, 64;
    cvt.u64.u32     %rd10, %r12;
    add.u64         %rd11, %rd9, %rd10;

    and.b32         %r19, %r10, 1;
    mov.u32         %r14, 0;

Q4K_INNER:
    setp.ge.u32     %p3, %r14, 32;
    @%p3            bra Q4K_INNER_DONE;

    cvt.u64.u32     %rd12, %r14;
    add.u64         %rd13, %rd11, %rd12;
    ld.global.u8    %r15, [%rd13];

    setp.ne.u32     %p4, %r19, 0;
    @%p4            bra Q4K_HIGH;
    and.b32         %r16, %r15, 15;
    bra Q4K_HAVE_Q;

Q4K_HIGH:
    shr.u32         %r16, %r15, 4;

Q4K_HAVE_Q:
    cvt.rn.f32.u32  %f4, %r16;
    mul.f32         %f5, %f2, %f4;
    sub.f32         %f5, %f5, %f3;

    mul.lo.u32      %r17, %r8, 256;
    mul.lo.u32      %r18, %r10, 32;
    add.u32         %r17, %r17, %r18;
    add.u32         %r17, %r17, %r14;
    mul.wide.u32    %rd14, %r17, 4;
    add.u64         %rd15, %rd2, %rd14;
    ld.global.f32   %f6, [%rd15];

    fma.rn.f32      %f1, %f5, %f6, %f1;

    add.u32         %r14, %r14, 1;
    bra Q4K_INNER;

Q4K_INNER_DONE:
    add.u32         %r10, %r10, 1;
    bra Q4K_GROUP;

Q4K_GROUP_DONE:
    add.u32         %r8, %r8, 1;
    bra Q4K_BLOCK;

Q4K_DONE:
    cvt.u64.u32     %rd16, %r4;
    shl.b64         %rd17, %rd16, 2;
    add.u64         %rd18, %rd3, %rd17;
    st.global.f32   [%rd18], %f1;

Q4K_EXIT:
    ret;
}
`

const matmulQ5KKernel = `
.visible .entry matmul_vec_q5_k(
    .param .u64 param_matrix,
    .param .u64 param_vec,
    .param .u64 param_out,
    .param .u32 param_rows,
    .param .u32 param_cols
)
{
    .reg .pred      %p<6>;
    .reg .b32       %r<24>;
    .reg .b64       %rd<28>;
    .reg .f32       %f<8>;

    mov.u32         %r1, %tid.x;
    mov.u32         %r2, %ctaid.x;
    mov.u32         %r3, %ntid.x;
    mad.lo.u32      %r4, %r2, %r3, %r1;

    ld.param.u32    %r5, [param_rows];
    setp.ge.u32     %p0, %r4, %r5;
    @%p0            bra Q5K_EXIT;

    ld.param.u64    %rd1, [param_matrix];
    ld.param.u64    %rd2, [param_vec];
    ld.param.u64    %rd3, [param_out];
    ld.param.u32    %r6, [param_cols];

    shr.u32         %r7, %r6, 8;
    mov.f32         %f1, 0f00000000;
    mov.u32         %r8, 0;

Q5K_BLOCK:
    setp.ge.u32     %p1, %r8, %r7;
    @%p1            bra Q5K_DONE;

    mul.lo.u32      %r9, %r4, %r7;
    add.u32         %r9, %r9, %r8;
    mul.wide.u32    %rd4, %r9, 224;
    add.u64         %rd5, %rd1, %rd4;

    mov.u32         %r10, 0;

Q5K_GROUP:
    setp.ge.u32     %p2, %r10, 8;
    @%p2            bra Q5K_GROUP_DONE;

    mul.wide.u32    %rd6, %r10, 4;
    add.u64         %rd7, %rd5, %rd6;
    ld.global.f32   %f2, [%rd7];
    add.u64         %rd8, %rd7, 32;
    ld.global.f32   %f3, [%rd8];

    shr.u32         %r11, %r10, 1;
    mul.lo.u32      %r12, %r11, 32;
    add.u64         %rd9, %rd5, 96;
    cvt.u64.u32     %rd10, %r12;
    add.u64         %rd11, %rd9, %rd10;

    add.u64         %rd16, %rd5, 64;

    shl.b32         %r20, %r11, 1;
    and.b32         %r19, %r10, 1;
    add.u32         %r20, %r20, %r19;
    mov.u32         %r21, 1;
    shl.b32         %r21, %r21, %r20;

    mov.u32         %r14, 0;

Q5K_INNER:
    setp.ge.u32     %p3, %r14, 32;
    @%p3            bra Q5K_INNER_DONE;

    cvt.u64.u32     %rd12, %r14;
    add.u64         %rd13, %rd11, %rd12;
    ld.global.u8    %r15, [%rd13];

    add.u64         %rd17, %rd16, %rd12;
    ld.global.u8    %r22, [%rd17];
    and.b32         %r22, %r22, %r21;
    setp.ne.u32     %p5, %r22, 0;

    setp.ne.u32     %p4, %r19, 0;
    @%p4            bra Q5K_HIGH;
    and.b32         %r16, %r15, 15;
    bra Q5K_HAVE_Q;

Q5K_HIGH:
    shr.u32         %r16, %r15, 4;

Q5K_HAVE_Q:
    @%p5            add.u32 %r16, %r16, 16;
    cvt.rn.f32.u32  %f4, %r16;
    mul.f32         %f5, %f2, %f4;
    sub.f32         %f5, %f5, %f3;

    mul.lo.u32      %r17, %r8, 256;
    mul.lo.u32      %r18, %r10, 32;
    add.u32         %r17, %r17, %r18;
    add.u32         %r17, %r17, %r14;
    mul.wide.u32    %rd14, %r17, 4;
    add.u64         %rd15, %rd2, %rd14;
    ld.global.f32   %f6, [%rd15];

    fma.rn.f32      %f1, %f5, %f6, %f1;

    add.u32         %r14, %r14, 1;
    bra Q5K_INNER;

Q5K_INNER_DONE:
    add.u32         %r10, %r10, 1;
    bra Q5K_GROUP;

Q5K_GROUP_DONE:
    add.u32         %r8, %r8, 1;
    bra Q5K_BLOCK;

Q5K_DONE:
    cvt.u64.u32     %rd18, %r4;
    shl.b64         %rd19, %rd18, 2;
    add.u64         %rd20, %rd3, %rd19;
    st.global.f32   [%rd20], %f1;

Q5K_EXIT:
    ret;
}
`

// GPU block: 16*f32 scales (d*int8) + 128 ql + 64 qh = 256
const matmulQ6KKernel = `
.visible .entry matmul_vec_q6_k(
    .param .u64 param_matrix,
    .param .u64 param_vec,
    .param .u64 param_out,
    .param .u32 param_rows,
    .param .u32 param_cols
)
{
    .reg .pred      %p<4>;
    .reg .b32       %r<32>;
    .reg .b64       %rd<32>;
    .reg .f32       %f<12>;

    mov.u32         %r1, %tid.x;
    mov.u32         %r2, %ctaid.x;
    mov.u32         %r3, %ntid.x;
    mad.lo.u32      %r4, %r2, %r3, %r1;

    ld.param.u32    %r5, [param_rows];
    setp.ge.u32     %p0, %r4, %r5;
    @%p0            bra Q6K_EXIT;

    ld.param.u64    %rd1, [param_matrix];
    ld.param.u64    %rd2, [param_vec];
    ld.param.u64    %rd3, [param_out];
    ld.param.u32    %r6, [param_cols];

    shr.u32         %r7, %r6, 8;
    mov.f32         %f1, 0f00000000;
    mov.u32         %r8, 0;

Q6K_BLOCK:
    setp.ge.u32     %p1, %r8, %r7;
    @%p1            bra Q6K_DONE;

    mul.lo.u32      %r9, %r4, %r7;
    add.u32         %r9, %r9, %r8;
    mul.wide.u32    %rd4, %r9, 256;
    add.u64         %rd5, %rd1, %rd4;

    mov.u32         %r10, 0;

Q6K_HALF:
    setp.ge.u32     %p2, %r10, 2;
    @%p2            bra Q6K_HALF_DONE;

    mul.lo.u32      %r11, %r10, 64;
    add.u64         %rd6, %rd5, 64;
    cvt.u64.u32     %rd7, %r11;
    add.u64         %rd8, %rd6, %rd7;

    mul.lo.u32      %r12, %r10, 32;
    add.u64         %rd9, %rd5, 192;
    cvt.u64.u32     %rd10, %r12;
    add.u64         %rd11, %rd9, %rd10;

    mul.lo.u32      %r13, %r10, 8;
    mul.wide.u32    %rd12, %r13, 4;
    add.u64         %rd13, %rd5, %rd12;

    mul.lo.u32      %r14, %r8, 256;
    mul.lo.u32      %r15, %r10, 128;
    add.u32         %r16, %r14, %r15;

    mov.u32         %r17, 0;

Q6K_INNER:
    setp.ge.u32     %p3, %r17, 32;
    @%p3            bra Q6K_INNER_DONE;

    cvt.u64.u32     %rd14, %r17;
    add.u64         %rd15, %rd8, %rd14;
    ld.global.u8    %r18, [%rd15];
    add.u64         %rd16, %rd15, 32;
    ld.global.u8    %r19, [%rd16];

    add.u64         %rd17, %rd11, %rd14;
    ld.global.u8    %r20, [%rd17];

    shr.u32         %r21, %r17, 4;
    mul.wide.u32    %rd18, %r21, 4;
    add.u64         %rd19, %rd13, %rd18;
    ld.global.f32   %f2, [%rd19];
    add.u64         %rd20, %rd19, 8;
    ld.global.f32   %f3, [%rd20];
    add.u64         %rd21, %rd19, 16;
    ld.global.f32   %f4, [%rd21];
    add.u64         %rd22, %rd19, 24;
    ld.global.f32   %f5, [%rd22];

    and.b32         %r22, %r18, 15;
    and.b32         %r23, %r20, 3;
    shl.b32         %r23, %r23, 4;
    or.b32          %r22, %r22, %r23;
    sub.s32         %r22, %r22, 32;
    cvt.rn.f32.s32  %f6, %r22;
    mul.f32         %f6, %f2, %f6;

    and.b32         %r24, %r19, 15;
    shr.u32         %r25, %r20, 2;
    and.b32         %r25, %r25, 3;
    shl.b32         %r25, %r25, 4;
    or.b32          %r24, %r24, %r25;
    sub.s32         %r24, %r24, 32;
    cvt.rn.f32.s32  %f7, %r24;
    mul.f32         %f7, %f3, %f7;

    shr.u32         %r26, %r18, 4;
    shr.u32         %r27, %r20, 4;
    and.b32         %r27, %r27, 3;
    shl.b32         %r27, %r27, 4;
    or.b32          %r26, %r26, %r27;
    sub.s32         %r26, %r26, 32;
    cvt.rn.f32.s32  %f8, %r26;
    mul.f32         %f8, %f4, %f8;

    shr.u32         %r28, %r19, 4;
    shr.u32         %r29, %r20, 6;
    and.b32         %r29, %r29, 3;
    shl.b32         %r29, %r29, 4;
    or.b32          %r28, %r28, %r29;
    sub.s32         %r28, %r28, 32;
    cvt.rn.f32.s32  %f9, %r28;
    mul.f32         %f9, %f5, %f9;

    add.u32         %r30, %r16, %r17;
    mul.wide.u32    %rd23, %r30, 4;
    add.u64         %rd24, %rd2, %rd23;
    ld.global.f32   %f10, [%rd24];
    fma.rn.f32      %f1, %f6, %f10, %f1;

    add.u32         %r30, %r30, 32;
    mul.wide.u32    %rd23, %r30, 4;
    add.u64         %rd24, %rd2, %rd23;
    ld.global.f32   %f10, [%rd24];
    fma.rn.f32      %f1, %f7, %f10, %f1;

    add.u32         %r30, %r30, 32;
    mul.wide.u32    %rd23, %r30, 4;
    add.u64         %rd24, %rd2, %rd23;
    ld.global.f32   %f10, [%rd24];
    fma.rn.f32      %f1, %f8, %f10, %f1;

    add.u32         %r30, %r30, 32;
    mul.wide.u32    %rd23, %r30, 4;
    add.u64         %rd24, %rd2, %rd23;
    ld.global.f32   %f10, [%rd24];
    fma.rn.f32      %f1, %f9, %f10, %f1;

    add.u32         %r17, %r17, 1;
    bra Q6K_INNER;

Q6K_INNER_DONE:
    add.u32         %r10, %r10, 1;
    bra Q6K_HALF;

Q6K_HALF_DONE:
    add.u32         %r8, %r8, 1;
    bra Q6K_BLOCK;

Q6K_DONE:
    cvt.u64.u32     %rd25, %r4;
    shl.b64         %rd26, %rd25, 2;
    add.u64         %rd27, %rd3, %rd26;
    st.global.f32   [%rd27], %f1;

Q6K_EXIT:
    ret;
}
`

// GPU block: f32 d + f32 m + 16 qs = 24
const matmulQ41Kernel = `
.visible .entry matmul_vec_q4_1(
    .param .u64 param_matrix,
    .param .u64 param_vec,
    .param .u64 param_out,
    .param .u32 param_rows,
    .param .u32 param_cols
)
{
    .reg .pred      %p<8>;
    .reg .s8        %s<1>;
    .reg .b32       %r<32>;
    .reg .b64       %rd<20>;
    .reg .f32       %f<10>;

    mov.u32         %r1, %tid.x;
    mov.u32         %r2, %ctaid.x;
    mov.u32         %r3, %ntid.x;
    mad.lo.u32      %r4, %r2, %r3, %r1;

    ld.param.u32    %r5, [param_rows];
    setp.ge.u32     %p0, %r4, %r5;
    @%p0            bra Q41_EXIT;

    ld.param.u64    %rd1, [param_matrix];
    ld.param.u64    %rd2, [param_vec];
    ld.param.u64    %rd3, [param_out];
    ld.param.u32    %r6, [param_cols];

    shr.u32         %r7, %r6, 5;
    mov.f32         %f1, 0f00000000;
    mov.u32         %r8, 0;

Q41_BLOCK:
    setp.ge.u32     %p1, %r8, %r7;
    @%p1            bra Q41_DONE;

    mul.lo.u32      %r9, %r4, %r7;
    add.u32         %r9, %r9, %r8;
    mul.wide.u32    %rd4, %r9, 24;
    add.u64         %rd5, %rd1, %rd4;

    ld.global.f32   %f2, [%rd5];
    ld.global.f32   %f3, [%rd5+4];
    add.u64         %rd6, %rd5, 8;
    shl.b32         %r10, %r8, 5;
    mov.u32         %r11, 0;

Q41_INNER:
    setp.ge.u32     %p2, %r11, 32;
    @%p2            bra Q41_INNER_DONE;

    shr.u32         %r12, %r11, 1;
    cvt.u64.u32     %rd7, %r12;
    add.u64         %rd8, %rd6, %rd7;
    ld.global.u8    %r13, [%rd8];

    and.b32         %r14, %r11, 1;
    shl.b32         %r14, %r14, 2;
    shr.u32         %r15, %r13, %r14;
    and.b32         %r15, %r15, 15;
    cvt.rn.f32.u32  %f4, %r15;
    fma.rn.f32      %f5, %f2, %f4, %f3;

    add.u32         %r16, %r10, %r11;
    mul.wide.u32    %rd9, %r16, 4;
    add.u64         %rd10, %rd2, %rd9;
    ld.global.f32   %f6, [%rd10];
    fma.rn.f32      %f1, %f5, %f6, %f1;

    add.u32         %r11, %r11, 1;
    bra Q41_INNER;

Q41_INNER_DONE:
    add.u32         %r8, %r8, 1;
    bra Q41_BLOCK;

Q41_DONE:
    cvt.u64.u32     %rd14, %r4;
    shl.b64         %rd15, %rd14, 2;
    add.u64         %rd16, %rd3, %rd15;
    st.global.f32   [%rd16], %f1;

Q41_EXIT:
    ret;
}
`

// GPU block: f32 d + u32 qh + 16 qs = 24
const matmulQ50Kernel = `
.visible .entry matmul_vec_q5_0(
    .param .u64 param_matrix,
    .param .u64 param_vec,
    .param .u64 param_out,
    .param .u32 param_rows,
    .param .u32 param_cols
)
{
    .reg .pred      %p<8>;
    .reg .s8        %s<1>;
    .reg .b32       %r<32>;
    .reg .b64       %rd<20>;
    .reg .f32       %f<10>;

    mov.u32         %r1, %tid.x;
    mov.u32         %r2, %ctaid.x;
    mov.u32         %r3, %ntid.x;
    mad.lo.u32      %r4, %r2, %r3, %r1;

    ld.param.u32    %r5, [param_rows];
    setp.ge.u32     %p0, %r4, %r5;
    @%p0            bra Q50_EXIT;

    ld.param.u64    %rd1, [param_matrix];
    ld.param.u64    %rd2, [param_vec];
    ld.param.u64    %rd3, [param_out];
    ld.param.u32    %r6, [param_cols];

    shr.u32         %r7, %r6, 5;
    mov.f32         %f1, 0f00000000;
    mov.u32         %r8, 0;

Q50_BLOCK:
    setp.ge.u32     %p1, %r8, %r7;
    @%p1            bra Q50_DONE;

    mul.lo.u32      %r9, %r4, %r7;
    add.u32         %r9, %r9, %r8;
    mul.wide.u32    %rd4, %r9, 24;
    add.u64         %rd5, %rd1, %rd4;

    ld.global.f32   %f2, [%rd5];
    ld.global.u32   %r10, [%rd5+4];
    add.u64         %rd6, %rd5, 8;
    shl.b32         %r11, %r8, 5;
    mov.u32         %r12, 0;

Q50_INNER:
    setp.ge.u32     %p2, %r12, 16;
    @%p2            bra Q50_INNER_DONE;

    cvt.u64.u32     %rd7, %r12;
    add.u64         %rd8, %rd6, %rd7;
    ld.global.u8    %r13, [%rd8];

    shr.u32         %r14, %r10, %r12;
    shl.b32         %r14, %r14, 4;
    and.b32         %r14, %r14, 16;
    add.u32         %r15, %r12, 12;
    shr.u32         %r15, %r10, %r15;
    and.b32         %r15, %r15, 16;

    and.b32         %r16, %r13, 15;
    or.b32          %r16, %r16, %r14;
    sub.s32         %r16, %r16, 16;
    cvt.rn.f32.s32  %f3, %r16;
    mul.f32         %f3, %f2, %f3;

    shr.u32         %r17, %r13, 4;
    or.b32          %r17, %r17, %r15;
    sub.s32         %r17, %r17, 16;
    cvt.rn.f32.s32  %f4, %r17;
    mul.f32         %f4, %f2, %f4;

    add.u32         %r18, %r11, %r12;
    mul.wide.u32    %rd9, %r18, 4;
    add.u64         %rd10, %rd2, %rd9;
    ld.global.f32   %f5, [%rd10];
    fma.rn.f32      %f1, %f3, %f5, %f1;
    ld.global.f32   %f6, [%rd10+64];
    fma.rn.f32      %f1, %f4, %f6, %f1;

    add.u32         %r12, %r12, 1;
    bra Q50_INNER;

Q50_INNER_DONE:
    add.u32         %r8, %r8, 1;
    bra Q50_BLOCK;

Q50_DONE:
    cvt.u64.u32     %rd14, %r4;
    shl.b64         %rd15, %rd14, 2;
    add.u64         %rd16, %rd3, %rd15;
    st.global.f32   [%rd16], %f1;

Q50_EXIT:
    ret;
}
`

// GPU block: f32 d + f32 m + u32 qh + 16 qs = 28
const matmulQ51Kernel = `
.visible .entry matmul_vec_q5_1(
    .param .u64 param_matrix,
    .param .u64 param_vec,
    .param .u64 param_out,
    .param .u32 param_rows,
    .param .u32 param_cols
)
{
    .reg .pred      %p<8>;
    .reg .s8        %s<1>;
    .reg .b32       %r<32>;
    .reg .b64       %rd<20>;
    .reg .f32       %f<10>;

    mov.u32         %r1, %tid.x;
    mov.u32         %r2, %ctaid.x;
    mov.u32         %r3, %ntid.x;
    mad.lo.u32      %r4, %r2, %r3, %r1;

    ld.param.u32    %r5, [param_rows];
    setp.ge.u32     %p0, %r4, %r5;
    @%p0            bra Q51_EXIT;

    ld.param.u64    %rd1, [param_matrix];
    ld.param.u64    %rd2, [param_vec];
    ld.param.u64    %rd3, [param_out];
    ld.param.u32    %r6, [param_cols];

    shr.u32         %r7, %r6, 5;
    mov.f32         %f1, 0f00000000;
    mov.u32         %r8, 0;

Q51_BLOCK:
    setp.ge.u32     %p1, %r8, %r7;
    @%p1            bra Q51_DONE;

    mul.lo.u32      %r9, %r4, %r7;
    add.u32         %r9, %r9, %r8;
    mul.wide.u32    %rd4, %r9, 28;
    add.u64         %rd5, %rd1, %rd4;

    ld.global.f32   %f2, [%rd5];
    ld.global.f32   %f7, [%rd5+4];
    ld.global.u32   %r10, [%rd5+8];
    add.u64         %rd6, %rd5, 12;
    shl.b32         %r11, %r8, 5;
    mov.u32         %r12, 0;

Q51_INNER:
    setp.ge.u32     %p2, %r12, 16;
    @%p2            bra Q51_INNER_DONE;

    cvt.u64.u32     %rd7, %r12;
    add.u64         %rd8, %rd6, %rd7;
    ld.global.u8    %r13, [%rd8];

    shr.u32         %r14, %r10, %r12;
    shl.b32         %r14, %r14, 4;
    and.b32         %r14, %r14, 16;
    add.u32         %r15, %r12, 12;
    shr.u32         %r15, %r10, %r15;
    and.b32         %r15, %r15, 16;

    and.b32         %r16, %r13, 15;
    or.b32          %r16, %r16, %r14;
    cvt.rn.f32.u32  %f3, %r16;
    fma.rn.f32      %f3, %f2, %f3, %f7;

    shr.u32         %r17, %r13, 4;
    or.b32          %r17, %r17, %r15;
    cvt.rn.f32.u32  %f4, %r17;
    fma.rn.f32      %f4, %f2, %f4, %f7;

    add.u32         %r18, %r11, %r12;
    mul.wide.u32    %rd9, %r18, 4;
    add.u64         %rd10, %rd2, %rd9;
    ld.global.f32   %f5, [%rd10];
    fma.rn.f32      %f1, %f3, %f5, %f1;
    ld.global.f32   %f6, [%rd10+64];
    fma.rn.f32      %f1, %f4, %f6, %f1;

    add.u32         %r12, %r12, 1;
    bra Q51_INNER;

Q51_INNER_DONE:
    add.u32         %r8, %r8, 1;
    bra Q51_BLOCK;

Q51_DONE:
    cvt.u64.u32     %rd14, %r4;
    shl.b64         %rd15, %rd14, 2;
    add.u64         %rd16, %rd3, %rd15;
    st.global.f32   [%rd16], %f1;

Q51_EXIT:
    ret;
}
`

// GPU block: 16*f32 dl + 16*f32 ml + 64 qs = 192
const matmulQ2KKernel = `
.visible .entry matmul_vec_q2_k(
    .param .u64 param_matrix,
    .param .u64 param_vec,
    .param .u64 param_out,
    .param .u32 param_rows,
    .param .u32 param_cols
)
{
    .reg .pred      %p<8>;
    .reg .s8        %s<1>;
    .reg .b32       %r<32>;
    .reg .b64       %rd<20>;
    .reg .f32       %f<10>;

    mov.u32         %r1, %tid.x;
    mov.u32         %r2, %ctaid.x;
    mov.u32         %r3, %ntid.x;
    mad.lo.u32      %r4, %r2, %r3, %r1;

    ld.param.u32    %r5, [param_rows];
    setp.ge.u32     %p0, %r4, %r5;
    @%p0            bra Q2K_EXIT;

    ld.param.u64    %rd1, [param_matrix];
    ld.param.u64    %rd2, [param_vec];
    ld.param.u64    %rd3, [param_out];
    ld.param.u32    %r6, [param_cols];

    shr.u32         %r7, %r6, 8;
    mov.f32         %f1, 0f00000000;
    mov.u32         %r8, 0;

Q2K_BLOCK:
    setp.ge.u32     %p1, %r8, %r7;
    @%p1            bra Q2K_DONE;

    mul.lo.u32      %r9, %r4, %r7;
    add.u32         %r9, %r9, %r8;
    mul.wide.u32    %rd4, %r9, 192;
    add.u64         %rd5, %rd1, %rd4;

    shl.b32         %r10, %r8, 8;
    mov.u32         %r11, 0;

Q2K_CHUNK:
    setp.ge.u32     %p2, %r11, 16;
    @%p2            bra Q2K_CHUNK_DONE;

    mul.wide.u32    %rd6, %r11, 4;
    add.u64         %rd7, %rd5, %rd6;
    ld.global.f32   %f2, [%rd7];
    ld.global.f32   %f3, [%rd7+64];

    shr.u32         %r12, %r11, 3;
    shl.b32         %r12, %r12, 5;
    and.b32         %r13, %r11, 1;
    shl.b32         %r13, %r13, 4;
    add.u32         %r12, %r12, %r13;
    shr.u32         %r14, %r11, 1;
    and.b32         %r14, %r14, 3;
    shl.b32         %r14, %r14, 1;

    cvt.u64.u32     %rd8, %r12;
    add.u64         %rd9, %rd5, 128;
    add.u64         %rd9, %rd9, %rd8;

    shl.b32         %r15, %r11, 4;
    add.u32         %r15, %r15, %r10;
    mov.u32         %r16, 0;

Q2K_INNER:
    setp.ge.u32     %p3, %r16, 16;
    @%p3            bra Q2K_INNER_DONE;

    cvt.u64.u32     %rd10, %r16;
    add.u64         %rd11, %rd9, %rd10;
    ld.global.u8    %r17, [%rd11];
    shr.u32         %r17, %r17, %r14;
    and.b32         %r17, %r17, 3;
    cvt.rn.f32.u32  %f4, %r17;
    mul.f32         %f4, %f2, %f4;
    sub.f32         %f4, %f4, %f3;

    add.u32         %r18, %r15, %r16;
    mul.wide.u32    %rd12, %r18, 4;
    add.u64         %rd13, %rd2, %rd12;
    ld.global.f32   %f5, [%rd13];
    fma.rn.f32      %f1, %f4, %f5, %f1;

    add.u32         %r16, %r16, 1;
    bra Q2K_INNER;

Q2K_INNER_DONE:
    add.u32         %r11, %r11, 1;
    bra Q2K_CHUNK;

Q2K_CHUNK_DONE:
    add.u32         %r8, %r8, 1;
    bra Q2K_BLOCK;

Q2K_DONE:
    cvt.u64.u32     %rd14, %r4;
    shl.b64         %rd15, %rd14, 2;
    add.u64         %rd16, %rd3, %rd15;
    st.global.f32   [%rd16], %f1;

Q2K_EXIT:
    ret;
}
`

// GPU block: 16*f32 scales + 32 hmask + 64 qs = 160
const matmulQ3KKernel = `
.visible .entry matmul_vec_q3_k(
    .param .u64 param_matrix,
    .param .u64 param_vec,
    .param .u64 param_out,
    .param .u32 param_rows,
    .param .u32 param_cols
)
{
    .reg .pred      %p<8>;
    .reg .s8        %s<1>;
    .reg .b32       %r<32>;
    .reg .b64       %rd<20>;
    .reg .f32       %f<10>;

    mov.u32         %r1, %tid.x;
    mov.u32         %r2, %ctaid.x;
    mov.u32         %r3, %ntid.x;
    mad.lo.u32      %r4, %r2, %r3, %r1;

    ld.param.u32    %r5, [param_rows];
    setp.ge.u32     %p0, %r4, %r5;
    @%p0            bra Q3K_EXIT;

    ld.param.u64    %rd1, [param_matrix];
    ld.param.u64    %rd2, [param_vec];
    ld.param.u64    %rd3, [param_out];
    ld.param.u32    %r6, [param_cols];

    shr.u32         %r7, %r6, 8;
    mov.f32         %f1, 0f00000000;
    mov.u32         %r8, 0;

Q3K_BLOCK:
    setp.ge.u32     %p1, %r8, %r7;
    @%p1            bra Q3K_DONE;

    mul.lo.u32      %r9, %r4, %r7;
    add.u32         %r9, %r9, %r8;
    mul.wide.u32    %rd4, %r9, 160;
    add.u64         %rd5, %rd1, %rd4;

    shl.b32         %r10, %r8, 8;
    mov.u32         %r11, 0;

Q3K_CHUNK:
    setp.ge.u32     %p2, %r11, 16;
    @%p2            bra Q3K_CHUNK_DONE;

    mul.wide.u32    %rd6, %r11, 4;
    add.u64         %rd7, %rd5, %rd6;
    ld.global.f32   %f2, [%rd7];

    shr.u32         %r12, %r11, 3;
    shl.b32         %r12, %r12, 5;
    and.b32         %r13, %r11, 1;
    shl.b32         %r13, %r13, 4;
    add.u32         %r12, %r12, %r13;
    shr.u32         %r14, %r11, 1;
    and.b32         %r14, %r14, 3;
    shl.b32         %r14, %r14, 1;

    cvt.u64.u32     %rd8, %r12;
    add.u64         %rd9, %rd5, 96;
    add.u64         %rd9, %rd9, %rd8;
    shr.u32         %r19, %r11, 3;
    shl.b32         %r19, %r19, 2;
    shr.u32         %r20, %r11, 1;
    and.b32         %r20, %r20, 3;
    add.u32         %r19, %r19, %r20;
    mov.u32         %r21, 1;
    shl.b32         %r21, %r21, %r19;
    add.u64         %rd17, %rd5, 64;
    cvt.u64.u32     %rd18, %r13;
    add.u64         %rd17, %rd17, %rd18;

    shl.b32         %r15, %r11, 4;
    add.u32         %r15, %r15, %r10;
    mov.u32         %r16, 0;

Q3K_INNER:
    setp.ge.u32     %p3, %r16, 16;
    @%p3            bra Q3K_INNER_DONE;

    cvt.u64.u32     %rd10, %r16;
    add.u64         %rd11, %rd9, %rd10;
    ld.global.u8    %r17, [%rd11];
    shr.u32         %r17, %r17, %r14;
    and.b32         %r17, %r17, 3;
    add.u64         %rd19, %rd17, %rd10;
    ld.global.u8    %r22, [%rd19];
    and.b32         %r22, %r22, %r21;
    setp.eq.u32     %p4, %r22, 0;
    @%p4            sub.s32 %r17, %r17, 4;
    cvt.rn.f32.s32  %f4, %r17;
    mul.f32         %f4, %f2, %f4;

    add.u32         %r18, %r15, %r16;
    mul.wide.u32    %rd12, %r18, 4;
    add.u64         %rd13, %rd2, %rd12;
    ld.global.f32   %f5, [%rd13];
    fma.rn.f32      %f1, %f4, %f5, %f1;

    add.u32         %r16, %r16, 1;
    bra Q3K_INNER;

Q3K_INNER_DONE:
    add.u32         %r11, %r11, 1;
    bra Q3K_CHUNK;

Q3K_CHUNK_DONE:
    add.u32         %r8, %r8, 1;
    bra Q3K_BLOCK;

Q3K_DONE:
    cvt.u64.u32     %rd14, %r4;
    shl.b64         %rd15, %rd14, 2;
    add.u64         %rd16, %rd3, %rd15;
    st.global.f32   [%rd16], %f1;

Q3K_EXIT:
    ret;
}
`

// GPU block: f32 d + 256 qs = 260
const matmulQ8KKernel = `
.visible .entry matmul_vec_q8_k(
    .param .u64 param_matrix,
    .param .u64 param_vec,
    .param .u64 param_out,
    .param .u32 param_rows,
    .param .u32 param_cols
)
{
    .reg .pred      %p<8>;
    .reg .s8        %s<1>;
    .reg .b32       %r<32>;
    .reg .b64       %rd<20>;
    .reg .f32       %f<10>;

    mov.u32         %r1, %tid.x;
    mov.u32         %r2, %ctaid.x;
    mov.u32         %r3, %ntid.x;
    mad.lo.u32      %r4, %r2, %r3, %r1;

    ld.param.u32    %r5, [param_rows];
    setp.ge.u32     %p0, %r4, %r5;
    @%p0            bra Q8K_EXIT;

    ld.param.u64    %rd1, [param_matrix];
    ld.param.u64    %rd2, [param_vec];
    ld.param.u64    %rd3, [param_out];
    ld.param.u32    %r6, [param_cols];

    shr.u32         %r7, %r6, 8;
    mov.f32         %f1, 0f00000000;
    mov.u32         %r8, 0;

Q8K_BLOCK:
    setp.ge.u32     %p1, %r8, %r7;
    @%p1            bra Q8K_DONE;

    mul.lo.u32      %r9, %r4, %r7;
    add.u32         %r9, %r9, %r8;
    mul.wide.u32    %rd4, %r9, 260;
    add.u64         %rd5, %rd1, %rd4;

    ld.global.f32   %f2, [%rd5];
    add.u64         %rd6, %rd5, 4;
    shl.b32         %r10, %r8, 8;
    mov.u32         %r11, 0;

Q8K_INNER:
    setp.ge.u32     %p2, %r11, 256;
    @%p2            bra Q8K_INNER_DONE;

    cvt.u64.u32     %rd7, %r11;
    add.u64         %rd8, %rd6, %rd7;
    ld.global.s8    %s0, [%rd8];
    cvt.s32.s8      %r12, %s0;
    cvt.rn.f32.s32  %f3, %r12;
    mul.f32         %f4, %f2, %f3;

    add.u32         %r13, %r10, %r11;
    mul.wide.u32    %rd9, %r13, 4;
    add.u64         %rd10, %rd2, %rd9;
    ld.global.f32   %f5, [%rd10];
    fma.rn.f32      %f1, %f4, %f5, %f1;

    add.u32         %r11, %r11, 1;
    bra Q8K_INNER;

Q8K_INNER_DONE:
    add.u32         %r8, %r8, 1;
    bra Q8K_BLOCK;

Q8K_DONE:
    cvt.u64.u32     %rd14, %r4;
    shl.b64         %rd15, %rd14, 2;
    add.u64         %rd16, %rd3, %rd15;
    st.global.f32   [%rd16], %f1;

Q8K_EXIT:
    ret;
}
`

const matmulKernelsBody = matmulVecKernel + matmulQ8Kernel + matmulMatQ8Kernel + matmulQ4Kernel + matmulQ4KKernel + matmulQ5KKernel + matmulQ6KKernel +
	matmulQ41Kernel + matmulQ50Kernel + matmulQ51Kernel + matmulQ2KKernel + matmulQ3KKernel + matmulQ8KKernel

// opsKernelsBody - RMSNorm, RoPE, SwiGLU, attention, softmax, add.
// Comments inside PTX must be ASCII only: ptxas rejects non-ASCII.
const opsKernelsBody = `
// rmsnorm: 1 block - sum on tid0, apply in parallel (stride)
.visible .entry rmsnorm(
    .param .u64 param_x,
    .param .u64 param_weight,
    .param .u64 param_out,
    .param .u32 param_n,
    .param .f32 param_eps
)
{
    .reg .pred      %p<3>;
    .reg .b32       %r<10>;
    .reg .b64       %rd<14>;
    .reg .f32       %f<8>;
    .shared .align 4 .b32 scale_s;

    mov.u32         %r1, %tid.x;
    mov.u32         %r2, %ntid.x;
    ld.param.u32    %r3, [param_n];
    ld.param.u64    %rd1, [param_x];
    ld.param.u64    %rd2, [param_weight];
    ld.param.u64    %rd3, [param_out];
    ld.param.f32    %f1, [param_eps];

    setp.ne.u32     %p0, %r1, 0;
    @%p0            bra RN_WAIT;

    mov.f32         %f2, 0f00000000;
    mov.u32         %r4, 0;

RN_SUM:
    setp.ge.u32     %p1, %r4, %r3;
    @%p1            bra RN_SCALE;
    mul.wide.u32    %rd4, %r4, 4;
    add.u64         %rd5, %rd1, %rd4;
    ld.global.f32   %f3, [%rd5];
    fma.rn.f32      %f2, %f3, %f3, %f2;
    add.u32         %r4, %r4, 1;
    bra RN_SUM;

RN_SCALE:
    cvt.rn.f32.u32  %f4, %r3;
    div.rn.f32      %f4, %f2, %f4;
    add.f32         %f4, %f4, %f1;
    rsqrt.approx.f32 %f5, %f4;
    st.shared.f32   [scale_s], %f5;

RN_WAIT:
    bar.sync        0;
    ld.shared.f32   %f5, [scale_s];
    mov.u32         %r4, %r1;

RN_APPLY:
    setp.ge.u32     %p2, %r4, %r3;
    @%p2            bra RN_EXIT;
    mul.wide.u32    %rd4, %r4, 4;
    add.u64         %rd5, %rd1, %rd4;
    ld.global.f32   %f3, [%rd5];
    add.u64         %rd6, %rd2, %rd4;
    ld.global.f32   %f4, [%rd6];
    mul.f32         %f3, %f3, %f5;
    mul.f32         %f3, %f3, %f4;
    add.u64         %rd7, %rd3, %rd4;
    st.global.f32   [%rd7], %f3;
    add.u32         %r4, %r4, %r2;
    bra RN_APPLY;

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
    .reg .b64       %rd<20>;
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

.visible .entry rope_heads_norm(
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
    .reg .b64       %rd<20>;
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
    @%p0            bra RHN_EXIT;

    ld.param.u64    %rd1, [param_v];
    ld.param.u64    %rd2, [param_cos];
    ld.param.u64    %rd3, [param_sin];

    div.u32         %r9, %r4, %r7;
    rem.u32         %r10, %r4, %r7;

    mul.lo.u32      %r11, %r9, %r6;
    mul.lo.u32      %r12, %r10, 2;
    add.u32         %r12, %r11, %r12;
    mul.lo.u32      %r13, %r12, 4;
    cvt.u64.u32     %rd4, %r13;
    add.u64         %rd5, %rd1, %rd4;
    ld.global.f32   %f1, [%rd5];

    add.u32         %r14, %r12, 1;
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

RHN_EXIT:
    ret;
}

.visible .entry swiglu(
    .param .u64 param_gate,
    .param .u64 param_up,
    .param .u32 param_n
)
{
    .reg .pred      %p<1>;
    .reg .b32       %r<8>;
    .reg .b64       %rd<8>;
    .reg .f32       %f<12>;

    mov.u32         %r1, %tid.x;
    mov.u32         %r2, %ctaid.x;
    mov.u32         %r3, %ntid.x;
    mad.lo.u32      %r4, %r2, %r3, %r1;

    ld.param.u32    %r5, [param_n];
    setp.ge.u32     %p0, %r4, %r5;
    @%p0            bra SG_EXIT;

    ld.param.u64    %rd1, [param_gate];
    ld.param.u64    %rd2, [param_up];

    mul.wide.u32    %rd3, %r4, 4;
    add.u64         %rd4, %rd1, %rd3;
    ld.global.f32   %f1, [%rd4];

    add.u64         %rd5, %rd2, %rd3;
    ld.global.f32   %f2, [%rd5];

    mov.f32         %f3, 0f00000000;
    sub.f32         %f4, %f3, %f1;
    mov.f32         %f5, 0f3fb8aa3b;
    mul.f32         %f6, %f4, %f5;
    ex2.approx.f32  %f7, %f6;
    mov.f32         %f8, 0f3f800000;
    add.f32         %f9, %f7, %f8;
    div.rn.f32      %f10, %f1, %f9;
    mul.f32         %f11, %f10, %f2;
    st.global.f32   [%rd4], %f11;

SG_EXIT:
    ret;
}

// geglu: gelu(gate)*up -> gate (Gemma / Phi FFN)
// tanh-approx GELU: 0.5*g*(1+tanh(0.7978845608*(g+0.044715*g^3)));
// tanh(z) = 1 - 2/(exp(2z)+1), exp via ex2.approx (no tanh.approx before sm_75)
.visible .entry geglu(
    .param .u64 param_gate,
    .param .u64 param_up,
    .param .u32 param_n
)
{
    .reg .pred      %p<1>;
    .reg .b32       %r<8>;
    .reg .b64       %rd<8>;
    .reg .f32       %f<20>;

    mov.u32         %r1, %tid.x;
    mov.u32         %r2, %ctaid.x;
    mov.u32         %r3, %ntid.x;
    mad.lo.u32      %r4, %r2, %r3, %r1;

    ld.param.u32    %r5, [param_n];
    setp.ge.u32     %p0, %r4, %r5;
    @%p0            bra GG_EXIT;

    ld.param.u64    %rd1, [param_gate];
    ld.param.u64    %rd2, [param_up];

    mul.wide.u32    %rd3, %r4, 4;
    add.u64         %rd4, %rd1, %rd3;
    ld.global.f32   %f1, [%rd4];

    add.u64         %rd5, %rd2, %rd3;
    ld.global.f32   %f2, [%rd5];

    // inner = 0.7978845608 * (g + 0.044715*g^3)
    mul.f32         %f3, %f1, %f1;
    mul.f32         %f3, %f3, %f1;
    mov.f32         %f4, 0f3D372713;
    mul.f32         %f3, %f3, %f4;
    add.f32         %f3, %f3, %f1;
    mov.f32         %f5, 0f3F4C422A;
    mul.f32         %f3, %f3, %f5;

    // e = exp(2*inner) = ex2(2*inner*log2(e))
    add.f32         %f6, %f3, %f3;
    mov.f32         %f7, 0f3FB8AA3B;
    mul.f32         %f6, %f6, %f7;
    ex2.approx.f32  %f8, %f6;

    // tanh = 1 - 2/(e+1)
    mov.f32         %f9, 0f3F800000;
    add.f32         %f10, %f8, %f9;
    mov.f32         %f11, 0f40000000;
    div.rn.f32      %f12, %f11, %f10;
    sub.f32         %f13, %f9, %f12;

    // gelu(g) = 0.5*g*(1+tanh)
    add.f32         %f14, %f13, %f9;
    mul.f32         %f15, %f1, %f14;
    mov.f32         %f16, 0f3F000000;
    mul.f32         %f15, %f15, %f16;

    mul.f32         %f17, %f15, %f2;
    st.global.f32   [%rd4], %f17;

GG_EXIT:
    ret;
}

.visible .entry attn_qk(
    .param .u64 param_q,
    .param .u64 param_k,
    .param .u64 param_scores,
    .param .u32 param_seq_len,
    .param .u32 param_head_dim,
    .param .u32 param_kv_stride,
    .param .u32 param_kv_off,
    .param .u32 param_q_off,
    .param .f32 param_scale
)
{
    .reg .pred      %p<2>;
    .reg .b32       %r<16>;
    .reg .b64       %rd<20>;
    .reg .f32       %f<6>;

    mov.u32         %r1, %tid.x;
    mov.u32         %r2, %ctaid.x;
    mov.u32         %r3, %ntid.x;
    mad.lo.u32      %r4, %r2, %r3, %r1;

    ld.param.u32    %r5, [param_seq_len];
    setp.ge.u32     %p0, %r4, %r5;
    @%p0            bra AQ_EXIT;

    ld.param.u64    %rd1, [param_q];
    ld.param.u64    %rd2, [param_k];
    ld.param.u64    %rd3, [param_scores];
    ld.param.u32    %r6, [param_head_dim];
    ld.param.u32    %r7, [param_kv_stride];
    ld.param.u32    %r8, [param_kv_off];
    ld.param.u32    %r9, [param_q_off];
    ld.param.f32    %f1, [param_scale];

    mul.wide.u32    %rd4, %r4, %r7;
    cvt.u64.u32     %rd5, %r8;
    add.u64         %rd6, %rd4, %rd5;
    shl.b64         %rd7, %rd6, 2;
    add.u64         %rd8, %rd2, %rd7;

    mov.f32         %f2, 0f00000000;
    mov.u32         %r10, 0;

AQ_LOOP:
    setp.ge.u32     %p1, %r10, %r6;
    @%p1            bra AQ_DONE;

    cvt.u64.u32     %rd9, %r9;
    cvt.u64.u32     %rd10, %r10;
    add.u64         %rd11, %rd9, %rd10;
    shl.b64         %rd12, %rd11, 2;
    add.u64         %rd13, %rd1, %rd12;
    ld.global.f32   %f3, [%rd13];

    cvt.u64.u32     %rd14, %r10;
    shl.b64         %rd15, %rd14, 2;
    add.u64         %rd16, %rd8, %rd15;
    ld.global.f32   %f4, [%rd16];

    fma.rn.f32      %f2, %f3, %f4, %f2;
    add.u32         %r10, %r10, 1;
    bra AQ_LOOP;

AQ_DONE:
    mul.f32         %f5, %f2, %f1;
    mul.wide.u32    %rd4, %r4, 4;
    add.u64         %rd5, %rd3, %rd4;
    st.global.f32   [%rd5], %f5;

AQ_EXIT:
    ret;
}

.visible .entry attn_v(
    .param .u64 param_scores,
    .param .u64 param_v,
    .param .u64 param_out,
    .param .u32 param_seq_len,
    .param .u32 param_head_dim,
    .param .u32 param_kv_stride,
    .param .u32 param_kv_off
)
{
    .reg .pred      %p<2>;
    .reg .b32       %r<14>;
    .reg .b64       %rd<20>;
    .reg .f32       %f<5>;

    mov.u32         %r1, %tid.x;
    mov.u32         %r2, %ctaid.x;
    mov.u32         %r3, %ntid.x;
    mad.lo.u32      %r4, %r2, %r3, %r1;

    ld.param.u32    %r5, [param_head_dim];
    setp.ge.u32     %p0, %r4, %r5;
    @%p0            bra AV_EXIT;

    ld.param.u64    %rd1, [param_scores];
    ld.param.u64    %rd2, [param_v];
    ld.param.u64    %rd3, [param_out];
    ld.param.u32    %r6, [param_seq_len];
    ld.param.u32    %r7, [param_kv_stride];
    ld.param.u32    %r8, [param_kv_off];

    mov.f32         %f1, 0f00000000;
    mov.u32         %r9, 0;

AV_LOOP:
    setp.ge.u32     %p1, %r9, %r6;
    @%p1            bra AV_DONE;

    mul.wide.u32    %rd4, %r9, 4;
    add.u64         %rd5, %rd1, %rd4;
    ld.global.f32   %f2, [%rd5];

    mul.wide.u32    %rd6, %r9, %r7;
    cvt.u64.u32     %rd7, %r8;
    add.u64         %rd8, %rd6, %rd7;
    cvt.u64.u32     %rd9, %r4;
    add.u64         %rd10, %rd8, %rd9;
    shl.b64         %rd11, %rd10, 2;
    add.u64         %rd12, %rd2, %rd11;
    ld.global.f32   %f3, [%rd12];

    fma.rn.f32      %f1, %f2, %f3, %f1;
    add.u32         %r9, %r9, 1;
    bra AV_LOOP;

AV_DONE:
    mul.wide.u32    %rd4, %r4, 4;
    add.u64         %rd5, %rd3, %rd4;
    st.global.f32   [%rd5], %f1;

AV_EXIT:
    ret;
}

// attn_flash: fused QK dot + online softmax + weighted V sum, one launch.
// grid = (n_heads, n_queries), block = 256 threads (fixed), head_dim <= 256.
// Block (h, qi) attends over seq = seq_base + qi positions of K/V (causal batch: seq_base = past_len + 1; single decode query: n_queries = 1, seq_base = seq_len).
// q/out rows are [qi][h][head_dim]. K/V are [pos][kv_stride] with kv_off = (h/group)*head_dim.
// Keys are processed in chunks of 256 (one position per thread); scores live in shared memory, running max/denominator use the flash-attention rescale (corr = exp(m_old - m_new)).
.visible .entry attn_flash(
    .param .u64 param_q,
    .param .u64 param_k,
    .param .u64 param_v,
    .param .u64 param_out,
    .param .u32 param_seq_base,
    .param .u32 param_head_dim,
    .param .u32 param_kv_stride,
    .param .u32 param_group,
    .param .f32 param_scale
)
{
    .reg .pred      %p<8>;
    .reg .b32       %r<32>;
    .reg .b64       %rd<12>;
    .reg .f32       %f<16>;
    .shared .align 4 .b8 af_q[1024];
    .shared .align 4 .b8 af_s[1024];
    .shared .align 4 .b8 af_red[1024];

    mov.u32         %r1, %tid.x;
    mov.u32         %r2, %ctaid.x;
    mov.u32         %r3, %ctaid.y;
    mov.u32         %r4, %nctaid.x;
    ld.param.u64    %rd1, [param_q];
    ld.param.u64    %rd2, [param_k];
    ld.param.u64    %rd3, [param_v];
    ld.param.u64    %rd4, [param_out];
    ld.param.u32    %r5, [param_seq_base];
    ld.param.u32    %r6, [param_head_dim];
    ld.param.u32    %r7, [param_kv_stride];
    ld.param.u32    %r8, [param_group];
    ld.param.f32    %f1, [param_scale];

    add.u32         %r9, %r5, %r3;
    div.u32         %r10, %r2, %r8;
    mul.lo.u32      %r11, %r10, %r6;
    mad.lo.u32      %r12, %r3, %r4, %r2;
    mul.lo.u32      %r12, %r12, %r6;
    mul.wide.u32    %rd5, %r12, 4;
    add.u64         %rd1, %rd1, %rd5;
    add.u64         %rd4, %rd4, %rd5;

    mov.u32         %r20, af_q;
    mov.u32         %r21, af_s;
    mov.u32         %r22, af_red;
    shl.b32         %r25, %r1, 2;
    add.u32         %r26, %r21, %r25;
    add.u32         %r27, %r22, %r25;
    setp.lt.u32     %p6, %r1, %r6;

    mov.u32         %r15, %r1;
AF_LOADQ:
    setp.ge.u32     %p0, %r15, %r6;
    @%p0            bra AF_LOADQ_DONE;
    mul.wide.u32    %rd5, %r15, 4;
    add.u64         %rd6, %rd1, %rd5;
    ld.global.f32   %f7, [%rd6];
    shl.b32         %r23, %r15, 2;
    add.u32         %r24, %r20, %r23;
    st.shared.f32   [%r24], %f7;
    add.u32         %r15, %r15, 256;
    bra             AF_LOADQ;
AF_LOADQ_DONE:
    bar.sync        0;

    mov.f32         %f2, 0fFF800000;
    mov.f32         %f3, 0f00000000;
    mov.f32         %f4, 0f00000000;
    mov.f32         %f14, 0f3FB8AA3B;
    mov.u32         %r13, 0;

AF_CHUNK:
    setp.ge.u32     %p0, %r13, %r9;
    @%p0            bra AF_FINISH;
    add.u32         %r14, %r13, %r1;
    mov.f32         %f5, 0fFF800000;
    setp.ge.u32     %p1, %r14, %r9;
    @%p1            bra AF_SCORE_DONE;

    mul.wide.u32    %rd5, %r14, %r7;
    cvt.u64.u32     %rd6, %r11;
    add.u64         %rd5, %rd5, %rd6;
    shl.b64         %rd5, %rd5, 2;
    add.u64         %rd7, %rd2, %rd5;
    mov.f32         %f6, 0f00000000;
    mov.u32         %r15, 0;
AF_DOT:
    setp.ge.u32     %p2, %r15, %r6;
    @%p2            bra AF_DOT_DONE;
    shl.b32         %r23, %r15, 2;
    add.u32         %r24, %r20, %r23;
    ld.shared.f32   %f7, [%r24];
    mul.wide.u32    %rd8, %r15, 4;
    add.u64         %rd9, %rd7, %rd8;
    ld.global.f32   %f8, [%rd9];
    fma.rn.f32      %f6, %f7, %f8, %f6;
    add.u32         %r15, %r15, 1;
    bra             AF_DOT;
AF_DOT_DONE:
    mul.f32         %f5, %f6, %f1;
AF_SCORE_DONE:
    st.shared.f32   [%r27], %f5;
    bar.sync        0;

    mov.u32         %r16, 128;
AF_MAXRED:
    setp.ge.u32     %p3, %r1, %r16;
    @%p3            bra AF_MAXRED_SYNC;
    add.u32         %r23, %r1, %r16;
    shl.b32         %r23, %r23, 2;
    add.u32         %r24, %r22, %r23;
    ld.shared.f32   %f8, [%r24];
    ld.shared.f32   %f9, [%r27];
    max.f32         %f9, %f9, %f8;
    st.shared.f32   [%r27], %f9;
AF_MAXRED_SYNC:
    bar.sync        0;
    shr.u32         %r16, %r16, 1;
    setp.ne.u32     %p4, %r16, 0;
    @%p4            bra AF_MAXRED;

    ld.shared.f32   %f9, [%r22];
    max.f32         %f10, %f2, %f9;
    sub.f32         %f11, %f2, %f10;
    mul.f32         %f11, %f11, %f14;
    ex2.approx.f32  %f11, %f11;
    mov.f32         %f2, %f10;
    sub.f32         %f12, %f5, %f2;
    mul.f32         %f12, %f12, %f14;
    ex2.approx.f32  %f12, %f12;
    fma.rn.f32      %f3, %f3, %f11, %f12;
    st.shared.f32   [%r26], %f12;
    bar.sync        0;

    @!%p6           bra AF_V_DONE;
    mul.f32         %f4, %f4, %f11;
    sub.u32         %r18, %r9, %r13;
    min.u32         %r18, %r18, 256;
    mov.u32         %r17, 0;
AF_VLOOP:
    setp.ge.u32     %p5, %r17, %r18;
    @%p5            bra AF_V_DONE;
    shl.b32         %r23, %r17, 2;
    add.u32         %r24, %r21, %r23;
    ld.shared.f32   %f12, [%r24];
    add.u32         %r19, %r13, %r17;
    mul.wide.u32    %rd5, %r19, %r7;
    cvt.u64.u32     %rd6, %r11;
    add.u64         %rd5, %rd5, %rd6;
    cvt.u64.u32     %rd6, %r1;
    add.u64         %rd5, %rd5, %rd6;
    shl.b64         %rd5, %rd5, 2;
    add.u64         %rd6, %rd3, %rd5;
    ld.global.f32   %f13, [%rd6];
    fma.rn.f32      %f4, %f12, %f13, %f4;
    add.u32         %r17, %r17, 1;
    bra             AF_VLOOP;
AF_V_DONE:
    bar.sync        0;
    add.u32         %r13, %r13, 256;
    bra             AF_CHUNK;

AF_FINISH:
    st.shared.f32   [%r27], %f3;
    bar.sync        0;
    mov.u32         %r16, 128;
AF_SUMRED:
    setp.ge.u32     %p3, %r1, %r16;
    @%p3            bra AF_SUMRED_SYNC;
    add.u32         %r23, %r1, %r16;
    shl.b32         %r23, %r23, 2;
    add.u32         %r24, %r22, %r23;
    ld.shared.f32   %f8, [%r24];
    ld.shared.f32   %f9, [%r27];
    add.f32         %f9, %f9, %f8;
    st.shared.f32   [%r27], %f9;
AF_SUMRED_SYNC:
    bar.sync        0;
    shr.u32         %r16, %r16, 1;
    setp.ne.u32     %p4, %r16, 0;
    @%p4            bra AF_SUMRED;

    ld.shared.f32   %f9, [%r22];
    @!%p6           bra AF_EXIT;
    div.rn.f32      %f4, %f4, %f9;
    mul.wide.u32    %rd5, %r1, 4;
    add.u64         %rd6, %rd4, %rd5;
    st.global.f32   [%rd6], %f4;
AF_EXIT:
    ret;
}

// mla_attn: absorbed MLA decode attention (DeepSeek-V2/V3), one block per head, 256 threads.
//   q   : [n_heads][nope + rope]  (q_nope unrotated, q_pe already rotated on host)
//   kb  : [n_heads][kvlora][nope] f32 (attn_k_b, ggml layout {nope, kvlora, n_heads})
//   wvb : [n_heads][vhead][kvlora] f32 (attn_v_b)
//   kc  : [pos][kvlora + rope]  K cache (kv_cmpr, k_pe)   vc : [pos][kvlora]  V cache (kv_cmpr)
//   out : [n_heads][vhead]
// Steps: q_abs[m] = sum_k kb[m][k]*q_nope[k]; scores = q_abs|q_pe . K row (online softmax, chunks of 256); ctx = softmax-weighted V (up to 4 dims per thread, kvlora <= 1024); out = wvb . ctx.
// Limits: kvlora <= 1024, kvlora + rope <= 1280.
.visible .entry mla_attn(
    .param .u64 param_q,
    .param .u64 param_kb,
    .param .u64 param_wvb,
    .param .u64 param_kc,
    .param .u64 param_vc,
    .param .u64 param_out,
    .param .u32 param_seq,
    .param .u32 param_nope,
    .param .u32 param_kvlora,
    .param .u32 param_rope,
    .param .u32 param_vhead,
    .param .f32 param_scale
)
{
    .reg .pred      %p<12>;
    .reg .b32       %r<40>;
    .reg .b64       %rd<24>;
    .reg .f32       %f<24>;
    .shared .align 4 .b8 ml_q[5120];
    .shared .align 4 .b8 ml_s[1024];
    .shared .align 4 .b8 ml_red[1024];

    mov.u32         %r1, %tid.x;
    mov.u32         %r2, %ctaid.x;
    ld.param.u64    %rd1, [param_q];
    ld.param.u64    %rd2, [param_kb];
    ld.param.u64    %rd3, [param_wvb];
    ld.param.u64    %rd4, [param_kc];
    ld.param.u64    %rd5, [param_vc];
    ld.param.u64    %rd6, [param_out];
    ld.param.u32    %r3, [param_seq];
    ld.param.u32    %r4, [param_nope];
    ld.param.u32    %r5, [param_kvlora];
    ld.param.u32    %r6, [param_rope];
    ld.param.u32    %r7, [param_vhead];
    ld.param.f32    %f1, [param_scale];

    add.u32         %r8, %r5, %r6;
    add.u32         %r9, %r4, %r6;

    mul.wide.u32    %rd7, %r2, %r9;
    shl.b64         %rd7, %rd7, 2;
    add.u64         %rd1, %rd1, %rd7;

    mul.lo.u32      %r10, %r5, %r4;
    mul.wide.u32    %rd7, %r2, %r10;
    shl.b64         %rd7, %rd7, 2;
    add.u64         %rd2, %rd2, %rd7;

    mul.lo.u32      %r10, %r7, %r5;
    mul.wide.u32    %rd7, %r2, %r10;
    shl.b64         %rd7, %rd7, 2;
    add.u64         %rd3, %rd3, %rd7;

    mul.wide.u32    %rd7, %r2, %r7;
    shl.b64         %rd7, %rd7, 2;
    add.u64         %rd6, %rd6, %rd7;

    mov.u32         %r20, ml_q;
    mov.u32         %r21, ml_s;
    mov.u32         %r22, ml_red;
    shl.b32         %r25, %r1, 2;
    add.u32         %r26, %r21, %r25;
    add.u32         %r27, %r22, %r25;
    add.u32         %r28, %r20, %r25;
    setp.lt.u32     %p6, %r1, %r5;
    add.u32         %r29, %r1, 256;
    setp.lt.u32     %p7, %r29, %r5;
    add.u32         %r29, %r1, 512;
    setp.lt.u32     %p8, %r29, %r5;
    add.u32         %r29, %r1, 768;
    setp.lt.u32     %p9, %r29, %r5;
    mov.f32         %f16, 0f00000000;
    mov.f32         %f17, 0f00000000;
    mov.f32         %f18, 0f00000000;
    mov.f32         %f19, 0f00000000;

    mov.u32         %r15, %r1;
ML_ABS:
    setp.ge.u32     %p0, %r15, %r5;
    @%p0            bra ML_ROPE;
    mul.wide.u32    %rd8, %r15, %r4;
    shl.b64         %rd8, %rd8, 2;
    add.u64         %rd8, %rd2, %rd8;
    mov.f32         %f6, 0f00000000;
    mov.u32         %r16, 0;
ML_ABS_DOT:
    setp.ge.u32     %p1, %r16, %r4;
    @%p1            bra ML_ABS_DOT_DONE;
    mul.wide.u32    %rd9, %r16, 4;
    add.u64         %rd10, %rd8, %rd9;
    ld.global.f32   %f7, [%rd10];
    add.u64         %rd11, %rd1, %rd9;
    ld.global.f32   %f8, [%rd11];
    fma.rn.f32      %f6, %f7, %f8, %f6;
    add.u32         %r16, %r16, 1;
    bra             ML_ABS_DOT;
ML_ABS_DOT_DONE:
    shl.b32         %r23, %r15, 2;
    add.u32         %r24, %r20, %r23;
    st.shared.f32   [%r24], %f6;
    add.u32         %r15, %r15, 256;
    bra             ML_ABS;

ML_ROPE:
    mov.u32         %r15, %r1;
ML_ROPE_LOOP:
    setp.ge.u32     %p0, %r15, %r6;
    @%p0            bra ML_ROPE_DONE;
    add.u32         %r16, %r15, %r4;
    mul.wide.u32    %rd9, %r16, 4;
    add.u64         %rd10, %rd1, %rd9;
    ld.global.f32   %f7, [%rd10];
    add.u32         %r17, %r15, %r5;
    shl.b32         %r23, %r17, 2;
    add.u32         %r24, %r20, %r23;
    st.shared.f32   [%r24], %f7;
    add.u32         %r15, %r15, 256;
    bra             ML_ROPE_LOOP;
ML_ROPE_DONE:
    bar.sync        0;

    mov.f32         %f2, 0fFF800000;
    mov.f32         %f3, 0f00000000;
    mov.f32         %f20, 0f3FB8AA3B;
    mov.u32         %r13, 0;

ML_CHUNK:
    setp.ge.u32     %p0, %r13, %r3;
    @%p0            bra ML_FINISH;
    add.u32         %r14, %r13, %r1;
    mov.f32         %f5, 0fFF800000;
    setp.ge.u32     %p1, %r14, %r3;
    @%p1            bra ML_SCORE_DONE;

    mul.wide.u32    %rd12, %r14, %r8;
    shl.b64         %rd12, %rd12, 2;
    add.u64         %rd13, %rd4, %rd12;
    mov.f32         %f6, 0f00000000;
    mov.u32         %r15, 0;
ML_DOT:
    setp.ge.u32     %p2, %r15, %r8;
    @%p2            bra ML_DOT_DONE;
    shl.b32         %r23, %r15, 2;
    add.u32         %r24, %r20, %r23;
    ld.shared.f32   %f7, [%r24];
    mul.wide.u32    %rd9, %r15, 4;
    add.u64         %rd10, %rd13, %rd9;
    ld.global.f32   %f8, [%rd10];
    fma.rn.f32      %f6, %f7, %f8, %f6;
    add.u32         %r15, %r15, 1;
    bra             ML_DOT;
ML_DOT_DONE:
    mul.f32         %f5, %f6, %f1;
ML_SCORE_DONE:
    st.shared.f32   [%r27], %f5;
    bar.sync        0;

    mov.u32         %r16, 128;
ML_MAXRED:
    setp.ge.u32     %p3, %r1, %r16;
    @%p3            bra ML_MAXRED_SYNC;
    add.u32         %r23, %r1, %r16;
    shl.b32         %r23, %r23, 2;
    add.u32         %r24, %r22, %r23;
    ld.shared.f32   %f8, [%r24];
    ld.shared.f32   %f9, [%r27];
    max.f32         %f9, %f9, %f8;
    st.shared.f32   [%r27], %f9;
ML_MAXRED_SYNC:
    bar.sync        0;
    shr.u32         %r16, %r16, 1;
    setp.ne.u32     %p4, %r16, 0;
    @%p4            bra ML_MAXRED;

    ld.shared.f32   %f9, [%r22];
    max.f32         %f10, %f2, %f9;
    sub.f32         %f11, %f2, %f10;
    mul.f32         %f11, %f11, %f20;
    ex2.approx.f32  %f11, %f11;
    mov.f32         %f2, %f10;
    sub.f32         %f12, %f5, %f2;
    mul.f32         %f12, %f12, %f20;
    ex2.approx.f32  %f12, %f12;
    fma.rn.f32      %f3, %f3, %f11, %f12;
    st.shared.f32   [%r26], %f12;
    bar.sync        0;

    mul.f32         %f16, %f16, %f11;
    mul.f32         %f17, %f17, %f11;
    mul.f32         %f18, %f18, %f11;
    mul.f32         %f19, %f19, %f11;
    sub.u32         %r18, %r3, %r13;
    min.u32         %r18, %r18, 256;
    mov.u32         %r17, 0;
ML_VLOOP:
    setp.ge.u32     %p5, %r17, %r18;
    @%p5            bra ML_V_DONE;
    shl.b32         %r23, %r17, 2;
    add.u32         %r24, %r21, %r23;
    ld.shared.f32   %f12, [%r24];
    add.u32         %r19, %r13, %r17;
    mul.wide.u32    %rd14, %r19, %r5;
    cvt.u64.u32     %rd15, %r1;
    add.u64         %rd14, %rd14, %rd15;
    shl.b64         %rd14, %rd14, 2;
    add.u64         %rd15, %rd5, %rd14;
    @%p6            ld.global.f32   %f13, [%rd15];
    @%p6            fma.rn.f32      %f16, %f12, %f13, %f16;
    @%p7            ld.global.f32   %f14, [%rd15+1024];
    @%p7            fma.rn.f32      %f17, %f12, %f14, %f17;
    @%p8            ld.global.f32   %f15, [%rd15+2048];
    @%p8            fma.rn.f32      %f18, %f12, %f15, %f18;
    @%p9            ld.global.f32   %f21, [%rd15+3072];
    @%p9            fma.rn.f32      %f19, %f12, %f21, %f19;
    add.u32         %r17, %r17, 1;
    bra             ML_VLOOP;
ML_V_DONE:
    bar.sync        0;
    add.u32         %r13, %r13, 256;
    bra             ML_CHUNK;

ML_FINISH:
    st.shared.f32   [%r27], %f3;
    bar.sync        0;
    mov.u32         %r16, 128;
ML_SUMRED:
    setp.ge.u32     %p3, %r1, %r16;
    @%p3            bra ML_SUMRED_SYNC;
    add.u32         %r23, %r1, %r16;
    shl.b32         %r23, %r23, 2;
    add.u32         %r24, %r22, %r23;
    ld.shared.f32   %f8, [%r24];
    ld.shared.f32   %f9, [%r27];
    add.f32         %f9, %f9, %f8;
    st.shared.f32   [%r27], %f9;
ML_SUMRED_SYNC:
    bar.sync        0;
    shr.u32         %r16, %r16, 1;
    setp.ne.u32     %p4, %r16, 0;
    @%p4            bra ML_SUMRED;

    ld.shared.f32   %f9, [%r22];
    @%p6            div.rn.f32      %f16, %f16, %f9;
    @%p6            st.shared.f32   [%r28], %f16;
    @%p7            div.rn.f32      %f17, %f17, %f9;
    @%p7            st.shared.f32   [%r28+1024], %f17;
    @%p8            div.rn.f32      %f18, %f18, %f9;
    @%p8            st.shared.f32   [%r28+2048], %f18;
    @%p9            div.rn.f32      %f19, %f19, %f9;
    @%p9            st.shared.f32   [%r28+3072], %f19;
    bar.sync        0;

    mov.u32         %r15, %r1;
ML_OUT:
    setp.ge.u32     %p0, %r15, %r7;
    @%p0            bra ML_EXIT;
    mul.wide.u32    %rd8, %r15, %r5;
    shl.b64         %rd8, %rd8, 2;
    add.u64         %rd9, %rd3, %rd8;
    mov.f32         %f6, 0f00000000;
    mov.u32         %r16, 0;
ML_OUT_DOT:
    setp.ge.u32     %p1, %r16, %r5;
    @%p1            bra ML_OUT_DOT_DONE;
    shl.b32         %r23, %r16, 2;
    add.u32         %r24, %r20, %r23;
    ld.shared.f32   %f7, [%r24];
    mul.wide.u32    %rd10, %r16, 4;
    add.u64         %rd11, %rd9, %rd10;
    ld.global.f32   %f8, [%rd11];
    fma.rn.f32      %f6, %f7, %f8, %f6;
    add.u32         %r16, %r16, 1;
    bra             ML_OUT_DOT;
ML_OUT_DOT_DONE:
    mul.wide.u32    %rd10, %r15, 4;
    add.u64         %rd11, %rd6, %rd10;
    st.global.f32   [%rd11], %f6;
    add.u32         %r15, %r15, 256;
    bra             ML_OUT;
ML_EXIT:
    ret;
}

// softmax - single-thread in-place softmax over scores (decode: short seq)
.visible .entry softmax(
    .param .u64 param_x,
    .param .u32 param_n
)
{
    .reg .pred      %p<3>;
    .reg .b32       %r<8>;
    .reg .b64       %rd<8>;
    .reg .f32       %f<16>;

    mov.u32         %r1, %tid.x;
    mov.u32         %r2, %ctaid.x;
    or.b32          %r3, %r1, %r2;
    setp.ne.u32     %p0, %r3, 0;
    @%p0            bra SM_EXIT;

    ld.param.u32    %r4, [param_n];
    setp.eq.u32     %p1, %r4, 0;
    @%p1            bra SM_EXIT;

    ld.param.u64    %rd1, [param_x];
    ld.global.f32   %f1, [%rd1];
    mov.u32         %r5, 1;

SM_MAX:
    setp.ge.u32     %p2, %r5, %r4;
    @%p2            bra SM_EXP;
    mul.wide.u32    %rd2, %r5, 4;
    add.u64         %rd3, %rd1, %rd2;
    ld.global.f32   %f2, [%rd3];
    max.f32         %f1, %f1, %f2;
    add.u32         %r5, %r5, 1;
    bra SM_MAX;

SM_EXP:
    mov.f32         %f3, 0f00000000;
    mov.u32         %r5, 0;

SM_EXP_LOOP:
    setp.ge.u32     %p2, %r5, %r4;
    @%p2            bra SM_NORM;
    mul.wide.u32    %rd2, %r5, 4;
    add.u64         %rd3, %rd1, %rd2;
    ld.global.f32   %f2, [%rd3];
    sub.f32         %f4, %f2, %f1;
    mov.f32         %f5, 0f3fb8aa3b;
    mul.f32         %f4, %f4, %f5;
    ex2.approx.f32  %f4, %f4;
    st.global.f32   [%rd3], %f4;
    add.f32         %f3, %f3, %f4;
    add.u32         %r5, %r5, 1;
    bra SM_EXP_LOOP;

SM_NORM:
    mov.f32         %f6, 0f3f800000;
    div.rn.f32      %f6, %f6, %f3;
    mov.u32         %r5, 0;

SM_NORM_LOOP:
    setp.ge.u32     %p2, %r5, %r4;
    @%p2            bra SM_EXIT;
    mul.wide.u32    %rd2, %r5, 4;
    add.u64         %rd3, %rd1, %rd2;
    ld.global.f32   %f2, [%rd3];
    mul.f32         %f2, %f2, %f6;
    st.global.f32   [%rd3], %f2;
    add.u32         %r5, %r5, 1;
    bra SM_NORM_LOOP;

SM_EXIT:
    ret;
}

// add_inplace: a[i] += b[i]
.visible .entry add_inplace(
    .param .u64 param_a,
    .param .u64 param_b,
    .param .u32 param_n
)
{
    .reg .pred      %p<2>;
    .reg .b32       %r<8>;
    .reg .b64       %rd<10>;
    .reg .f32       %f<4>;

    mov.u32         %r1, %tid.x;
    mov.u32         %r2, %ctaid.x;
    mov.u32         %r3, %ntid.x;
    mad.lo.u32      %r4, %r2, %r3, %r1;

    ld.param.u32    %r5, [param_n];
    setp.ge.u32     %p0, %r4, %r5;
    @%p0            bra AI_EXIT;

    ld.param.u64    %rd1, [param_a];
    ld.param.u64    %rd2, [param_b];
    mul.wide.u32    %rd3, %r4, 4;
    add.u64         %rd4, %rd1, %rd3;
    add.u64         %rd5, %rd2, %rd3;
    ld.global.f32   %f1, [%rd4];
    ld.global.f32   %f2, [%rd5];
    add.f32         %f1, %f1, %f2;
    st.global.f32   [%rd4], %f1;

AI_EXIT:
    ret;
}

// embed_gather_q8_0: dequant one Q8_0 emb row (token_id) into d_resid[embd]
.visible .entry embed_gather_q8_0(
    .param .u64 param_matrix,
    .param .u64 param_out,
    .param .u32 param_token,
    .param .u32 param_embd
)
{
    .reg .pred      %p<3>;
    .reg .s8        %s<1>;
    .reg .b32       %r<14>;
    .reg .b64       %rd<16>;
    .reg .f32       %f<4>;

    mov.u32         %r1, %tid.x;
    mov.u32         %r2, %ctaid.x;
    mov.u32         %r3, %ntid.x;
    mad.lo.u32      %r4, %r2, %r3, %r1;

    ld.param.u32    %r5, [param_embd];
    setp.ge.u32     %p0, %r4, %r5;
    @%p0            bra EG_EXIT;

    ld.param.u32    %r6, [param_token];
    shr.u32         %r7, %r5, 5;
    mul.lo.u32      %r8, %r6, %r7;
    shr.u32         %r9, %r4, 5;
    add.u32         %r8, %r8, %r9;
    mul.wide.u32    %rd4, %r8, 36;
    ld.param.u64    %rd1, [param_matrix];
    add.u64         %rd5, %rd1, %rd4;
    ld.global.f32   %f2, [%rd5];

    and.b32         %r10, %r4, 31;
    add.u64         %rd6, %rd5, 4;
    cvt.u64.u32     %rd7, %r10;
    add.u64         %rd8, %rd6, %rd7;
    ld.global.s8    %s0, [%rd8];
    cvt.s32.s8      %r11, %s0;
    cvt.rn.f32.s32  %f3, %r11;
    mul.f32         %f1, %f2, %f3;

    ld.param.u64    %rd3, [param_out];
    mul.wide.u32    %rd9, %r4, 4;
    add.u64         %rd10, %rd3, %rd9;
    st.global.f32   [%rd10], %f1;

EG_EXIT:
    ret;
}

// logits_argmax: single-block reduce; writes idx and value to out[0], out[1]
.visible .entry logits_argmax(
    .param .u64 param_logits,
    .param .u64 param_out,
    .param .u32 param_n
)
{
    .reg .pred      %p<3>;
    .reg .b32       %r<12>;
    .reg .b64       %rd<10>;
    .reg .f32       %f<4>;
    .shared .align 4 .b32 best_val;
    .shared .align 4 .b32 best_idx;

    mov.u32         %r1, %tid.x;
    mov.u32         %r2, %ntid.x;
    ld.param.u32    %r3, [param_n];
    ld.param.u64    %rd1, [param_logits];

    setp.ne.u32     %p0, %r1, 0;
    @%p0            bra LA_WAIT;

    mov.f32         %f1, 0ff8000000;
    mov.u32         %r4, 0;
    mov.u32         %r5, 0;

LA_LOOP:
    setp.ge.u32     %p1, %r5, %r3;
    @%p1            bra LA_STORE;
    mul.wide.u32    %rd2, %r5, 4;
    add.u64         %rd3, %rd1, %rd2;
    ld.global.f32   %f2, [%rd3];
    setp.gt.f32     %p2, %f2, %f1;
    @!%p2           bra LA_NEXT;
    mov.f32         %f1, %f2;
    mov.u32         %r4, %r5;
LA_NEXT:
    add.u32         %r5, %r5, 1;
    bra LA_LOOP;

LA_STORE:
    st.shared.f32   [best_val], %f1;
    st.shared.u32   [best_idx], %r4;

LA_WAIT:
    bar.sync        0;
    setp.ne.u32     %p0, %r1, 0;
    @%p0            bra LA_EXIT;
    ld.shared.f32   %f1, [best_val];
    ld.shared.u32   %r4, [best_idx];
    ld.param.u64    %rd4, [param_out];
    cvt.rn.f32.s32  %f3, %r4;
    st.global.f32   [%rd4], %f3;
    add.u64         %rd5, %rd4, 4;
    st.global.f32   [%rd5], %f1;

LA_EXIT:
    ret;
}
`
