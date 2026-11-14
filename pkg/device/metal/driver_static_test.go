package metal

import (
	"os"
	"regexp"
	"strings"
	"testing"
)

// The darwin driver cannot be compiled on other OSes, so these static checks keep the Go <-> C <-> MSL surface consistent everywhere (they read the sources next to this file; no GPU, no cgo)

func readSource(t *testing.T, name string) string {
	t.Helper()

	b, err := os.ReadFile(name)
	if err != nil {
		t.Fatal(err)
	}

	return string(b)
}

// every kernel named in B1Kernels / B2Kernels gets a pipeline state in gguf_metal_open
func TestDriverCompilesEveryKernel(t *testing.T) {
	m := readSource(t, "driver_darwin.m")
	for _, name := range append(append([]string(nil), B1Kernels...), B2Kernels...) {
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
	for _, file := range []string{"runtime_darwin.go", "fused_darwin.go"} {
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
	for _, name := range []string{"MatParams", "RMSParams", "RopeParams", "SwigluParams", "AddParams", "AttnParams"} {
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
