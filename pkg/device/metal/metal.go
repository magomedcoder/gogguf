package metal

import (
	"errors"
	"fmt"

	"github.com/magomedcoder/gogguf/pkg/compute"
	"github.com/magomedcoder/gogguf/pkg/dtype"
)

// ErrUnavailable Metal backend does not compute yet (every compute entry point wraps it, see unavailable)
var ErrUnavailable = errors.New("metal: backend not implemented (scaffold)")

// unavailable wraps ErrUnavailable with the entry point name and the planned kernel from PlannedKernels (errors.Is(err, ErrUnavailable) holds)
func unavailable(op string) error {
	if k := plannedKernelFor(op); k != "" {
		return fmt.Errorf("%w: %s needs MSL kernel %q (planned, see metal.PlannedKernels); use -ngl 0", ErrUnavailable, op, k)
	}

	return fmt.Errorf("%w: %s (planned, see metal.PlannedKernels); use -ngl 0", ErrUnavailable, op)
}

// Ready reports whether the backend can run model layers.
// False while kernels are scaffold; flips to true (KernelsImplemented) once MSL kernels are wired, which is what enables -ngl on darwin in pkg/model.Options.Normalize.
func (b *Backend) Ready() bool {
	return KernelsImplemented
}

// Backend - Metal device stub: implements full device.Backend interface but any compute call returns ErrUnavailable
type Backend struct {
	name string
}

// Name returns device name, e.g. "Metal (scaffold)"
func (b *Backend) Name() string {
	if b == nil || b.name == "" {
		return "Metal (scaffold)"
	}

	return b.name
}

// VRAMInfo: unified memory size not queried yet
func (b *Backend) VRAMInfo() (used, total uint64, err error) {
	return 0, 0, nil
}

func (b *Backend) MatMulVec([]float32, int, int, []float32) ([]float32, error) {
	return nil, unavailable("MatMulVec")
}

func (b *Backend) MatMulVecCached(string, []float32, int, int, []float32) ([]float32, error) {
	return nil, unavailable("MatMulVecCached")
}

func (b *Backend) MatMulVecQ8_0Cached(string, []byte, int, int, []float32) ([]float32, error) {
	return nil, unavailable("MatMulVecQ8_0Cached")
}

func (b *Backend) MatMulVecQ4_0Cached(string, []byte, int, int, []float32) ([]float32, error) {
	return nil, unavailable("MatMulVecQ4_0Cached")
}

func (b *Backend) MatMulVecQ4_KCached(string, []byte, int, int, []float32) ([]float32, error) {
	return nil, unavailable("MatMulVecQ4_KCached")
}

func (b *Backend) MatMulVecQ5_KCached(string, []byte, int, int, []float32) ([]float32, error) {
	return nil, unavailable("MatMulVecQ5_KCached")
}

func (b *Backend) MatMulVecQ6_KCached(string, []byte, int, int, []float32) ([]float32, error) {
	return nil, unavailable("MatMulVecQ6_KCached")
}

func (b *Backend) MatMulVecQ4_1Cached(string, []byte, int, int, []float32) ([]float32, error) {
	return nil, unavailable("MatMulVecQ4_1Cached")
}

func (b *Backend) MatMulVecQ5_0Cached(string, []byte, int, int, []float32) ([]float32, error) {
	return nil, unavailable("MatMulVecQ5_0Cached")
}

func (b *Backend) MatMulVecQ5_1Cached(string, []byte, int, int, []float32) ([]float32, error) {
	return nil, unavailable("MatMulVecQ5_1Cached")
}

func (b *Backend) MatMulVecQ2_KCached(string, []byte, int, int, []float32) ([]float32, error) {
	return nil, unavailable("MatMulVecQ2_KCached")
}

func (b *Backend) MatMulVecQ3_KCached(string, []byte, int, int, []float32) ([]float32, error) {
	return nil, unavailable("MatMulVecQ3_KCached")
}

func (b *Backend) MatMulVecQ8_KCached(string, []byte, int, int, []float32) ([]float32, error) {
	return nil, unavailable("MatMulVecQ8_KCached")
}

func (b *Backend) RMSNormInto(_, _, _ []float32, _ float32) error {
	return unavailable("RMSNormInto")
}

func (b *Backend) ApplyRoPEHeads([]float32, int, int, int, float32) error {
	return unavailable("ApplyRoPEHeads")
}

func (b *Backend) ApplyRoPEHeadsNorm([]float32, int, int, int, float32) error {
	return unavailable("ApplyRoPEHeadsNorm")
}

func (b *Backend) SwiGLUInPlace(_, _ []float32) error {
	return unavailable("SwiGLUInPlace")
}

func (b *Backend) FFNSwiGLUCached(_, _, _ string, _, _, _, _, _ []float32, _, _ int) error {
	return unavailable("FFNSwiGLUCached")
}

func (b *Backend) FFNSwiGLUQ8_0Cached(_, _, _ string, _, _, _ []byte, _, _ []float32, _, _ int) error {
	return unavailable("FFNSwiGLUQ8_0Cached")
}

func (b *Backend) AttnFFNResidualCached(_, _, _, _, _ string, _, _, _, _, _, _, _ []float32, _, _, _ int, _ float32) error {
	return unavailable("AttnFFNResidualCached")
}

func (b *Backend) AttnFFNResidualQ8_0Cached(_, _, _, _, _ string, _, _, _, _ []byte, _, _, _ []float32, _, _, _ int, _ float32) error {
	return unavailable("AttnFFNResidualQ8_0Cached")
}

func (b *Backend) QKVRoPEAttentionCached(_, _, _, _, _ string, _, _, _, _, _, _, _, _, _, _, _ []float32, _, _, _, _, _, _, _ int, _ float32) error {
	return unavailable("QKVRoPEAttentionCached")
}

func (b *Backend) QKVRoPEAttentionQ8_0Cached(_, _, _, _, _ string, _, _, _ []byte, _, _, _, _, _, _, _, _ []float32, _, _, _, _, _, _, _ int, _ float32) error {
	return unavailable("QKVRoPEAttentionQ8_0Cached")
}

func (b *Backend) FFNSwiGLUQuantCached(dtype.DType, string, string, string, []byte, []byte, []byte, []float32, []float32, int, int) error {
	return unavailable("FFNSwiGLUQuantCached")
}

func (b *Backend) FFNGeGLUQuantCached(dtype.DType, string, string, string, []byte, []byte, []byte, []float32, []float32, int, int) error {
	return unavailable("FFNGeGLUQuantCached")
}

func (b *Backend) AttnFFNResidualQuantCached(_ dtype.DType, _, _, _, _, _ string, _, _, _, _ []byte, _, _, _ []float32, _, _, _ int, _ float32) error {
	return unavailable("AttnFFNResidualQuantCached")
}

func (b *Backend) QKVRoPEAttentionQuantCached(_ dtype.DType, _ compute.RoPEMode, _, _, _, _, _, _ string, _, _, _ []byte, _, _, _, _, _, _, _, _, _ []float32, _, _, _, _, _, _, _ int, _ float32) error {
	return unavailable("QKVRoPEAttentionQuantCached")
}

// HiddenResident device-resident hidden state will require Metal kernels
func (b *Backend) HiddenResident() bool {
	return false
}

func (b *Backend) HiddenActive() bool {
	return false
}

func (b *Backend) HiddenUpload([]float32) error {
	return unavailable("HiddenUpload")
}

func (b *Backend) HiddenDownload([]float32) error {
	return unavailable("HiddenDownload")
}

func (b *Backend) LogitsFromDevice(dtype.DType, string, []float32, string, []byte, []float32, []float32, int, int, float32) error {
	return unavailable("LogitsFromDevice")
}

func (b *Backend) AttentionScoresInto(_, _, _, _, _ []float32, _, _, _, _ int) error {
	return unavailable("AttentionScoresInto")
}

func (b *Backend) KVCacheInit(int, int, int, int, int) error {
	return unavailable("KVCacheInit")
}

func (b *Backend) KVCacheReset() {}

func (b *Backend) KVCacheAppend(int, int, []float32, []float32) error {
	return unavailable("KVCacheAppend")
}

func (b *Backend) KVCacheAppendN(int, int, []float32, []float32, int) error {
	return unavailable("KVCacheAppendN")
}

func (b *Backend) AttentionScoresKV(int, []float32, []float32, int, int, int, int) error {
	return unavailable("AttentionScoresKV")
}

func (b *Backend) Close() error {
	return nil
}
