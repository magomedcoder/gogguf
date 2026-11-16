package metal

import (
	"os"
	"regexp"
	"strings"
	"testing"
)

// The darwin driver cannot be compiled on other OSes, so these static checks keep the Go <-> C <-> MSL surface consistent everywhere (they read the sources next to this file; no GPU, no cgo)

// allKernels lists every MSL entry point the driver must compile (B1 + B2 + B3)
func allKernels() []string {
	return append(append(append([]string(nil), B1Kernels...), B2Kernels...), B3Kernels...)
}

func readSource(t *testing.T, name string) string {
	t.Helper()

	b, err := os.ReadFile(name)
	if err != nil {
		t.Fatal(err)
	}

	return string(b)
}

// every kernel named in B1Kernels / B2Kernels / B3Kernels gets a pipeline state in gguf_metal_open
func TestDriverCompilesEveryKernel(t *testing.T) {
	m := readSource(t, "driver_darwin.m")
	for _, name := range allKernels() {
		if !strings.Contains(m, `make_pso(m, "`+name+`")`) {
			t.Fatalf("driver_darwin.m does not build a pipeline for kernel %q", name)
		}
	}
}

// every C function the Go runtime calls is declared in driver.h and defined in driver_darwin.m
func TestDriverSurfaceMatchesGoRuntime(t *testing.T) {
	h, m := readSource(t, "driver.h"), readSource(t, "driver_darwin.m")
	call := regexp.MustCompile(`C\.(gguf_metal_[a-z0-9_]+)\(`)

	seen := 0
	for _, file := range []string{"runtime_darwin.go", "fused_darwin.go", "batch_darwin.go"} {
		for _, mm := range call.FindAllStringSubmatch(readSource(t, file), -1) {
			seen++

			if !regexp.MustCompile(`\b` + mm[1] + `\(`).MatchString(h) {
				t.Fatalf("%s calls C.%s, not declared in driver.h", file, mm[1])
			}

			if !regexp.MustCompile(`(?m)^[a-z_ *0-9]+\b` + mm[1] + `\(.*\{`).MatchString(m) {
				t.Fatalf("%s calls C.%s, not defined in driver_darwin.m", file, mm[1])
			}
		}
	}

	if seen == 0 {
		t.Fatal("found no C calls in the darwin runtime sources")
	}
}

// the MSL param structs and the C ones in the driver have the same fields (all 4-byte scalars)
func TestDriverParamStructsMatchMSL(t *testing.T) {
	m, mslSource := readSource(t, "driver_darwin.m"), readSource(t, "msl.go")
	for _, name := range []string{"MatParams", "MatMatParams", "RMSParams", "RopeParams", "SwigluParams", "AddParams", "AttnParams"} {
		msl := regexp.MustCompile(`struct ` + name + ` \{([^}]*)\}`).FindStringSubmatch(mslSource)
		c := regexp.MustCompile(`typedef struct \{([^}]*)\} ` + name + `;`).FindStringSubmatch(m)
		if msl == nil || c == nil {
			t.Fatalf("param struct %s missing (msl=%v c=%v)", name, msl != nil, c != nil)
		}

		norm := func(s string) string {
			s = strings.NewReplacer("uint32_t", "uint").Replace(s)
			return strings.Join(strings.Fields(s), " ")
		}

		if norm(msl[1]) != norm(c[1]) {
			t.Fatalf("%s differs: msl %q vs driver %q", name, norm(msl[1]), norm(c[1]))
		}
	}
}

// B3: the GEMM entry point and the offset attention launch exist in the C surface and the MSL, and the kernel keeps the CUDA thread-per-(batch,row) shape
func TestB3DriverAndMSLSurface(t *testing.T) {
	h, m, msl := readSource(t, "driver.h"), readSource(t, "driver_darwin.m"), readSource(t, "msl.go")
	for _, fn := range []string{"gguf_metal_matmul_mat_q8_0", "gguf_metal_attention_off"} {
		if !strings.Contains(h, fn+"(") || !strings.Contains(m, fn+"(") {
			t.Fatalf("%s missing from driver.h / driver_darwin.m", fn)
		}
	}

	for _, want := range []string{
		"kernel void matmul_mat_q8_0(",
		"constant MatMatParams &p",
		"uint r = gid % p.rows;",
		"uint b = gid / p.rows;",
	} {
		if !strings.Contains(msl, want) {
			t.Fatalf("msl.go lacks %q", want)
		}
	}

	if strings.Contains(msl, "simdgroup_float8x8") || strings.Contains(msl, "simdgroup_matrix<") {
		t.Fatal("B3 matmul_mat_q8_0 must stay the thread-per-output kernel (simdgroup_matrix is a follow-up)")
	}

	// the GEMM launch is one thread per (batch, row): grid = batch*rows rounded up to threadgroups
	if !strings.Contains(m, "(uint64_t)rows * (uint64_t)batch") {
		t.Fatal("driver_darwin.m: matmul_mat_q8_0 grid is not batch*rows")
	}
}

// the B3 Go surface (Backend wrappers, runtime file) is wired: the runtime implements the three gpu entry points and encodes one batch per call
func TestB3RuntimeEncodesOneBatchPerCall(t *testing.T) {
	rt := readSource(t, "batch_darwin.go")
	for _, want := range []string{
		"func (r *mtl) matMulMatQ8_0(",
		"func (r *mtl) ffnSwiGLUMatQ8_0(",
		"func (r *mtl) attentionKVBatch(",
		"C.gguf_metal_matmul_mat_q8_0(",
		"C.gguf_metal_attention_off(",
		"C.gguf_metal_swiglu(",
	} {
		if !strings.Contains(rt, want) {
			t.Fatalf("batch_darwin.go lacks %q", want)
		}
	}

	if got := strings.Count(rt, "r.batch(func() error"); got != 2 {
		t.Fatalf("batch_darwin.go encodes %d command-buffer batches, want 2 (FFN, attention)", got)
	}
}

// Q5_K / Q6_K matvec kinds and the NORM rope launch are wired through driver.h, driver_darwin.m and the Go runtime
func TestDriverQ5Q6AndRoPENormSurface(t *testing.T) {
	h, m := readSource(t, "driver.h"), readSource(t, "driver_darwin.m")
	for _, want := range []string{"GGUF_METAL_MAT_Q4_K = 3,", "GGUF_METAL_MAT_Q5_K = 4,", "GGUF_METAL_MAT_Q6_K = 5,"} {
		if !strings.Contains(h, want) {
			t.Fatalf("driver.h lacks %q", want)
		}
	}

	for _, want := range []string{
		`make_pso(m, "matmul_vec_q5_k")`,
		`make_pso(m, "matmul_vec_q6_k")`,
		`make_pso(m, "rope_heads_norm")`,
		"case GGUF_METAL_MAT_Q5_K:",
		"pso = m.matQ5K;",
		"case GGUF_METAL_MAT_Q6_K:",
		"pso = m.matQ6K;",
		"m.ropeNorm",
	} {
		if !strings.Contains(m, want) {
			t.Fatalf("driver_darwin.m lacks %q", want)
		}
	}

	for _, fn := range []string{"gguf_metal_rope_heads_norm_off", "gguf_metal_rope_heads_norm"} {
		if !strings.Contains(h, fn+"(") || !regexp.MustCompile(`(?m)^int `+fn+`\(.*\{`).MatchString(m) {
			t.Fatalf("%s missing from driver.h / driver_darwin.m", fn)
		}
	}

	rt, fused := readSource(t, "runtime_darwin.go"), readSource(t, "fused_darwin.go")
	for _, want := range []string{"C.GGUF_METAL_MAT_Q5_K", "C.GGUF_METAL_MAT_Q6_K", "q5_k: make(map", "q6_k: make(map"} {
		if !strings.Contains(rt, want) {
			t.Fatalf("runtime_darwin.go lacks %q", want)
		}
	}

	// the fused encoders take the matvec kind from the weight type and the rope kernel from the mode: no hardcoded Q8_0 / NeoX
	if strings.Contains(fused, "GGUF_METAL_MAT_Q8_0") || strings.Contains(fused, "weightQ8") {
		t.Fatal("fused_darwin.go still hardcodes Q8_0")
	}

	for _, want := range []string{"func (r *mtl) weightQuant(t dtype.DType", "func (r *mtl) encRoPE(mode compute.RoPEMode", "C.gguf_metal_rope_heads_norm_off(", "r.encRoPE(a.mode, "} {
		if !strings.Contains(fused, want) {
			t.Fatalf("fused_darwin.go lacks %q", want)
		}
	}

	if !strings.Contains(rt, "C.gguf_metal_rope_heads") && !strings.Contains(fused, "C.gguf_metal_rope_heads_off(") {
		t.Fatal("NeoX rope launch no longer reachable")
	}
}
