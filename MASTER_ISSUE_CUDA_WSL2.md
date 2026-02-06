# Master Issue: CUDA enablement for Voxtral on Windows 11 + WSL2 (Ubuntu)

## Objective
Enable production-ready NVIDIA CUDA acceleration for Voxtral in Ubuntu under WSL2 so an RTX 3080 Ti can run low-latency/live speech-to-text inference.

## Success Criteria
- `make cuda` builds on Ubuntu in WSL2.
- Runtime can detect and use CUDA on NVIDIA GPUs from WSL2.
- End-to-end transcription works with the CUDA backend.
- Performance and quality are measured against the BLAS backend.
- Documentation is complete for setup, troubleshooting, and validation.

## Architecture Milestones

### 1) Build & Toolchain Foundation
**Deliverables**
- CUDA build target in `Makefile`.
- Compiler/linker checks for CUDA toolkit and cuBLAS.
- CI or scripted build verification on Linux.

**Sub-issues**
- [x] Add `make cuda` target and flags.
- [x] Add `make info` reporting for CUDA backend availability.
- [x] Add build troubleshooting docs for toolkit/library path issues.

### 2) Runtime CUDA Backend
**Deliverables**
- CUDA runtime detection.
- cuBLAS GEMM wrappers for core matrix paths.
- Correct CPU fallback when CUDA is unavailable.

**Sub-issues**
- [x] Add `voxtral_cuda.{c,h}` abstraction.
- [x] Wire CUDA backend into `vox_matmul` and `vox_matmul_t`.
- [x] Add backend selection logging (`--debug`) showing CPU/BLAS/CUDA path.

### 3) BF16 and Decoder Hot Path Optimization
**Deliverables**
- Reduce host/device transfer overhead in decoder loop.
- Improve single-token generation path performance.

**Sub-issues**
- [ ] Persistent device buffers for repeated linear ops.
- [ ] Optional CUDA stream and async memcpy for overlap.
- [ ] Evaluate cublasLt / fused kernels for BF16 weight paths.

### 4) Validation & Benchmarks
**Deliverables**
- Functional validation scripts (same transcript quality as baseline).
- Throughput and latency benchmarks on RTX 3080 Ti.

**Sub-issues**
- [x] Add smoke test command list (`samples/test_speech.wav`, stdin).
- [x] Add perf harness documenting tokens/sec and first-token latency.
- [ ] Add accuracy regression check (token diff tolerance strategy).

### 5) WSL2 Productization
**Deliverables**
- Explicit Windows+WSL2 setup guide.
- Known limitations and troubleshooting matrix.

**Sub-issues**
- [ ] Document driver/toolkit version matrix known to work.
- [ ] Add troubleshooting section (`nvidia-smi`, missing `libcublas.so`, OOM).
- [ ] Add optional deployment recipe for real-time microphone pipeline.

## Parallel Work Plan (for multiple agents)
- **Agent A:** Build system + docs foundation (Milestone 1 + part of 5).
- **Agent B:** CUDA runtime abstraction + matmul integration (Milestone 2).
- **Agent C:** BF16/decode optimization work (Milestone 3).
- **Agent D:** Benchmarks + regression validation assets (Milestone 4).

## Suggested PR Sequence
1. PR-1: Add CUDA build target + initial docs.
2. PR-2: Add CUDA runtime backend and GEMM integration.
3. PR-3: Add BF16 decode-path optimization and persistent buffers.
4. PR-4: Add benchmark + validation scripts and baseline numbers.
5. PR-5: Final WSL2 operator guide and troubleshooting hardening.

## Verification Checklist
- [ ] `make cuda` succeeds.
- [ ] `nvidia-smi` works inside WSL2.
- [ ] `./voxtral -d voxtral-model -i samples/test_speech.wav` succeeds on CUDA build.
- [ ] `--stdin` pipeline with ffmpeg succeeds.
- [ ] Performance report added and reviewed.
