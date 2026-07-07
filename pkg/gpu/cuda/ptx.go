//go:build cuda

package cuda

// ptxTargets возвращает список SM-таргетов для попытки JIT
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

// kernelsPTX выбирает PTX matmul-ядер по compute capability GPU
func kernelsPTX(cc int) string {
	targets := ptxTargets(cc)
	return kernelsPTXForTarget(targets[0])
}

// opsPTX выбирает PTX op-ядер (RMSNorm, ...)
func opsPTX(cc int) string {
	targets := ptxTargets(cc)
	return opsPTXForTarget(targets[0])
}

func ptxHeaderForTarget(target int) string {
	switch {
	case target >= 120:
		// Blackwell (sm_120) требует PTX >= 8.7
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

const matmulKernelsBody = matmulVecKernel + matmulQ8Kernel

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
`
