# CUDA + WSL2 Migration Handoff Notes

This document is a practical handoff for completing Voxtral CUDA enablement on a Windows 11 host running WSL2 Ubuntu.

## 1) Current status (implemented in repo)

### Build and backend wiring
- `make cuda` target exists in `Makefile`.
- `CUDA_HOME` override is supported (`CUDA_HOME ?= /usr/local/cuda`).
- `cuda-check` preflight validates `cuda_runtime.h` before compile.
- Core GEMM call sites (`vox_matmul`, `vox_matmul_t`) are CUDA-aware and fall back to BLAS/CPU if CUDA path fails.

### CUDA runtime abstraction
- `voxtral_cuda.c/.h` provide:
  - runtime availability detection,
  - cuBLAS SGEMM wrappers,
  - persistent device buffers,
  - a CUDA stream,
  - async memcpy and synchronization,
  - device-name accessor for debug logs.

### Validation + ops tooling
- `scripts/validate_cuda.sh` for environment/build/smoke tests.
- `scripts/benchmark_backends.sh` for BLAS vs CUDA timing comparisons.
- `scripts/accuracy_regression.sh` for transcript mismatch checks under a tolerance threshold.

### Documentation
- `README.md` now includes:
  - WSL2 CUDA prerequisites,
  - troubleshooting notes,
  - version matrix baseline,
  - real-time mic pipeline recipe,
  - helper script usage.
- `MASTER_ISSUE_CUDA_WSL2.md` tracks milestone progress and remaining items.

---

## 2) What remains (true open work)

## A. Performance-critical CUDA work still open
1. **BF16 decoder hot-path acceleration**
   - Current CUDA path accelerates FP32 GEMM calls.
   - The largest remaining gain is to accelerate BF16-heavy decode paths directly on GPU.
   - Investigate cublasLt BF16 GEMM and/or fused kernels.

2. **Reduce host/device transfers further**
   - Persistent buffers are present, but compute still uses host-resident flow between calls.
   - Next step: keep more intermediate tensors on device across decode iterations.

3. **Kernel fusion opportunities**
   - Candidate fusions: linear + bias + activation, or batched projection paths.
   - Evaluate before/after latency and first-token time.

## B. On-target validation still required (cannot be completed in this container)
1. Build with actual CUDA toolkit installed in WSL2.
2. Validate runtime with `nvidia-smi` and end-to-end transcription.
3. Capture benchmark numbers on the target RTX 3080 Ti.
4. Run transcript quality comparisons BLAS vs CUDA and record mismatch metrics.

---

## 3) Why the previous run could not fully complete everything

This container does not include CUDA toolkit headers/libs (for example, `cuda_runtime.h` under `/usr/local/cuda/include`), and does not provide the target RTX device context for runtime profiling. Because of that, local completion was limited to:
- compile-time-safe integration,
- fallback-safe runtime behavior,
- scripts/docs that are ready to execute on your WSL2 host.

---

## 4) Exact next steps on your Windows 11 + WSL2 machine

## Step 1: Validate host + guest GPU stack
```bash
# In WSL2 Ubuntu
nvidia-smi
```
Expected: RTX 3080 Ti is visible.

## Step 2: Ensure CUDA toolkit is installed in Ubuntu
```bash
# Verify headers/libs
ls /usr/local/cuda/include/cuda_runtime.h
ls /usr/local/cuda/lib64/libcublas.so
```
If toolkit path differs, note it and pass `CUDA_HOME=/your/path` to make.

## Step 3: Build and smoke test CUDA backend
```bash
make cuda
./scripts/validate_cuda.sh voxtral-model samples/test_speech.wav
```

## Step 4: Baseline perf and quality
```bash
./scripts/benchmark_backends.sh voxtral-model samples/test_speech.wav
./scripts/accuracy_regression.sh voxtral-model samples/test_speech.wav 0.005
```

## Step 5: Real-time stdin pipeline sanity check
```bash
ffmpeg -f pulse -i default -f s16le -ar 16000 -ac 1 - 2>/dev/null | \
  ./voxtral -d voxtral-model --stdin
```

---

## 5) Suggested follow-up PR plan (for Codex agents)

## PR-A: BF16 acceleration foundation
- Add cublasLt path for BF16 GEMM where tensor shapes are stable.
- Add runtime capability checks for BF16/tensor-core support.
- Include fallback path and debug logging.

## PR-B: Decode-loop device residency
- Keep selected decode intermediates on device across tokens.
- Minimize H2D/D2H traffic in per-token generation.
- Add benchmarks focused on first-token latency + tokens/sec.

## PR-C: Regression and benchmark hardening
- Add machine-readable benchmark output (CSV/JSON).
- Add transcript diff report artifact from accuracy script.
- Gate PRs with tolerance thresholds for quality drift.

## PR-D: WSL2 operator completion
- Record known-good exact versions observed on your machine.
- Add troubleshooting outcomes from real runs (driver mismatches, OOM behavior, etc.).

---

## 6) Acceptance checklist for “migration complete”

- [ ] `make cuda` succeeds on WSL2 host.
- [ ] `nvidia-smi` shows RTX 3080 Ti in WSL2.
- [ ] Offline file transcription works on CUDA build.
- [ ] `--stdin` live pipeline works.
- [ ] BLAS vs CUDA accuracy mismatch is within tolerance.
- [ ] CUDA benchmark demonstrates meaningful speedup vs BLAS.
- [ ] Remaining master issue items closed (or re-scoped with explicit rationale).

---

## 7) Files you should review first when continuing work
- `MASTER_ISSUE_CUDA_WSL2.md`
- `Makefile`
- `voxtral_cuda.c`
- `voxtral_kernels.c`
- `scripts/validate_cuda.sh`
- `scripts/benchmark_backends.sh`
- `scripts/accuracy_regression.sh`
- `README.md`

