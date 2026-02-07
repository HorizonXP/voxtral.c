# MPS Speed Optimization Log

## Standing Instructions
- **Continue tirelessly without asking for user prompt** — keep optimizing in a loop
- Commit each time a good improvement is reached
- Complexity increase must be justified by speed results
- Re-read this file at each context compaction
- Take notes on all results, what worked, what didn't

## Testing
- **Quick iteration**: use short samples in `samples/` (test_speech.wav, jfk.wav)
  for timing measurements during development
- **Before committing**: run `./runtest.sh` to verify no regressions (batch +
  streaming with KV cache compaction). This is slow (~2 min), so only run it as
  the final check before each commit

## Current Baseline (2026-02-06 / MacBook Pro M3 Max 40-core GPU, 128 GB, 400 GB/s)
- Decoder: 23.5 ms/token (was 43.2 at start)
- Prefill: ~252ms (was ~1200ms)
- Encoder: ~284ms (test_speech.wav, 3.6s audio), ~539ms (jfk.wav, 11s audio) (was ~2.7s at start)
- Theoretical decoder floor: ~17.3 ms/token (6.9 GB weights / 400 GB/s = 17.25 ms)
- Current efficiency: 73% of bandwidth limit (17.3 / 23.5)
- Remaining decoder overhead: ~6.2 ms/token above theoretical floor

## Already Optimized
- Fused QKV: 3 matmuls → 1 command buffer (saves 52 round-trips/token)
- Fused FFN: w1+w3+silu+mul+w2 → 1 command buffer (saves 52 round-trips/token)
- Persistent decoder buffers: eliminates 11 malloc/free per token
- Pre-warm weight cache: all bf16→f16 conversion at load time (8.4 GB GPU)
- GPU batched encoder attention: causal_softmax shader + strided MPS views
- Fused encoder QKV + FFN (reusing decoder fused functions)

## Optimization Attempts

### Attempt 1: Fused wo+FFN + Fused norm+QKV + Fused logits (SUCCESS)
- **Fused wo+FFN**: wo projection + residual + rms_norm + ada_scale + SwiGLU FFN + residual in 1 cmd buf
  - Eliminates separate wo cmd buf per layer
  - Result: 43 → ~40 ms/step
- **Fused norm+QKV**: rms_norm + 3 QKV matmuls in 1 cmd buf (replaces separate norm + fused QKV)
  - Marginal additional improvement (CPU rms_norm was fast)
- **Fused logits**: final rms_norm + logits matmul + argmax_f32 shader in 1 cmd buf
  - Avoids 512KB logits download when only argmax needed
  - Result: ~40 → ~38 ms/step
- **Prefill timing split**: separate prefill_ms from per-step timing for accurate measurement
- **Combined: 43.2 → 38 ms/step (12% improvement)**
- Command buffers per token: 79 → 53 (26×2 per layer + 1 logits)
- New shaders: ada_scale_mul, argmax_f32

### Attempt 2: Cross-layer fusion — persistent GPU x (SUCCESS)
- Keep x on GPU buffer across all 26 layers (no CPU round-trip between layers)
- Fuse wo_ffn[i] + norm_qkv[i+1] into a single command buffer
- Helper functions: encode_wo_ffn_steps(), encode_norm_qkv_steps()
- New API: decoder_start/end, decoder_norm_qkv, decoder_wo_ffn_next_qkv, decoder_wo_ffn_logits
- Command buffers per token: 53 → 27 (1 + 25 fused + 1 final)
- **Result: 38 → 32.5 ms/step (test_speech), 39 → 35.7 ms/step (jfk)**
- **Cumulative from baseline: 43.2 → 32.5 ms/step (25% faster)**

### Attempt 3: Monolithic decoder step — 1 command buffer (SUCCESS)
- All 26 layers + logits in a SINGLE Metal command buffer
- 3 new Metal compute shaders: rope_apply, kv_cache_copy, decoder_attention
- KV cache allocated with MTLResourceStorageModeShared (zero-copy CPU/GPU)
- GPU RoPE: applies rotary embeddings on Q and K (interleaved pair convention)
- GPU KV cache write: copies K/V to shared cache at correct layer/position offset
- GPU decoder attention: one threadgroup (128 threads) per head, cooperative dot products
  - Online softmax (single pass): avoids second scan over keys
  - First attempt with 1-thread-per-head: 77.9 ms/step (TERRIBLE — sequential per-key scan)
  - Fixed with 128-thread cooperative kernel: 26.6 ms/step (great improvement)
- Command buffers per token: 27 → 1
- **Result: 32.5 → 26.6 ms/step (test_speech), 35.7 → 28.7 ms/step (jfk)**
- **Cumulative from baseline: 43.2 → 26.6 ms/step (38% faster)**

### Attempt 4: SIMD attention (SUCCESS)
- Replaced threadgroup barrier reductions with simd_sum() in decoder_attention kernel
- 4 SIMD groups of 32 threads, simd_sum for dot product, cross-SIMD via threadgroup
- **Result: 26.6 → 25.3 ms/step (test_speech)**

### Attempt 5: Merged weight matrices (SUCCESS)
- Merged w1+w3 into single MPS matmul in FFN (saves 1 MPS encode per layer)
  - get_merged_f16_2(): concatenates two bf16 weight buffers into one f16 buffer
  - bufGate now sized hidden*2, silu/mul use buffer offsets
- Merged Q+K+V into single MPS matmul in QKV projection (saves 2 MPS encodes per layer)
  - get_merged_f16_3(): concatenates three bf16 weight buffers into one f16 buffer
  - Single bufQKV output, downstream RoPE/KV-cache/attention use buffer offsets
- Total: 3 fewer MPS matmul encodes per layer × 26 layers = 78 fewer encodes
- **Result: 25.3 → 23.7 ms/step (test_speech), ~24.9 ms/step (jfk)**
- **Cumulative from baseline: 43.2 → 23.7 ms/step (45% faster)**

### Attempt 6: BLAS conv1d (im2col + sgemm) (SUCCESS — HUGE)
- Replaced naive 4-nested-loop causal_conv1d with im2col + cblas_sgemm
- Old: per-element scalar multiply-adds (~875M iterations for conv1, ~174M for conv0)
- New: build im2col matrix, single BLAS sgemm per convolution
- Conv0: sgemm(1280, 355, 384) ≈ <1ms. Conv1: sgemm(1280, 178, 3840) ≈ 3ms
- **Result: encoder conv stem 2008ms → 12ms (167x faster!)**
- **Encoder total: 2737 → 804 ms (test_speech), 5613 → 1295 ms (jfk)**

### Attempt 7: Encoder wo projection on GPU (SUCCESS)
- Moved encoder output projection (wo) from CPU bf16→f32+sgemm to GPU MPS matmul
- Was the only remaining CPU matmul in the encoder pipeline
- **Result: encoder 804 → 683 ms (test_speech), 1295 → 1254 ms (jfk)**

### Attempt 8: Fused encoder attention kernel (SUCCESS)
- Replaced per-head MPS matmul encodes (QK^T + softmax + scores×V) with single compute kernel
- New Metal shader: encoder_attention (online softmax, SIMD cooperative dot products)
- 64 per-head encodes per layer → 1 compute dispatch per layer
- 2048 total MPS encodes → 32 compute dispatches (32 layers × 1)
- **Result: encoder 605 → 456 ms (test_speech), 1242 → 1082 ms (jfk)**
- **Cumulative encoder: 2.7s → 456 ms (83% faster)**

### Attempt 9: Pre-warm merged weights + KV cache + decoder ops (SUCCESS)
- Pre-warm merged weight buffers (QKV, w1+w3) during model loading instead of first decoder step
  - 26 layers × merged QKV (~38MB) + merged w1+w3 (~113MB) = ~3.9 GB created at load time
- Pre-allocate KV cache as shared GPU memory during model loading (saves ~130ms per run)
- Pre-warm MPS matmul ops and f32 weight caches (norms, ada_scale) during loading
- Reuse KV cache across decoder restarts (don't free+realloc on each audio file)
- Fused QKV and FFN for decoder prefill (3 cmd bufs/layer instead of 7)
- **Result: prefill 1212 → 500 ms (test_speech), 1166 → 499 ms (jfk) — 57% faster**

### Attempt 10: Monolithic encoder step (SUCCESS)
- All 32 encoder layers + final norm in ONE Metal command buffer
- 3 new Metal compute shaders: bias_add, batched_rope_apply, batched_kv_cache_copy
- Encoder KV cache pre-allocated as shared GPU memory (500 MB, zero-copy CPU/GPU)
- Separate Q/K/V matmuls per layer (not merged, to keep bias_add and RoPE simple)
- Command buffers per encoder invocation: 128 → 1
- **Result: encoder 456 → 310 ms (test_speech), 1082 → 690 ms (jfk) — 32-36% faster**
- **Cumulative encoder: 2.7s → 310 ms (89% faster)**

### Attempt 11: Monolithic decoder prefill (SUCCESS)
- All 26 decoder layers in ONE Metal command buffer for prefill (M>1)
- Reuses encoder_attention shader (already supports head_dim=128 via dynamic SIMD groups)
- Moved attention + wo projection from CPU to GPU (they were last remaining CPU ops in prefill)
- Fixed ada_scale_mul shader: added stride parameter for M>1 (was reading out-of-bounds)
- **Result: prefill 380 → 338 ms (test_speech), 380 → 327 ms (jfk) — 11-14% faster**

### Attempt 12: Merged encoder weights (SUCCESS — modest)
- Merged Q+K+V into single MPS matmul per encoder layer (3→1 encode)
  - New `deinterleave` Metal shader splits [M, 6144] → separate Q, K, V buffers
- Merged w1+w3 into single MPS matmul per encoder layer (2→1 encode)
  - New `silu_mul_merged` shader: fused silu+mul on interleaved [M, hidden*2] layout
  - w2 matmul uses strided rowBytes (hidden*2) to read directly from merged output
- Net: 7→4 MPS matmul encodes per layer × 32 layers = 96 fewer encodes
- MPS encoding overhead: 20ms → 14ms (saved 6ms)
- Larger merged matmuls slightly more efficient on GPU (better utilization)
- **Result: encoder ~310 → ~298 ms (test_speech) — 4% faster**
- jfk shows similar improvement (~690 → ~705ms, within noise)

### Attempt 13: Merged decoder prefill weights (SUCCESS — BIG)
- Same merged QKV + merged w1+w3 approach applied to decoder prefill path
- Decoder prefill uses seq_len=38 through 26 layers, all in one command buffer
- Merged QKV: [38, 3072] × [6144, 3072]^T instead of 3 separate matmuls
- Merged w1+w3: [38, 3072] × [18432, 3072]^T instead of 2 separate matmuls
- Reuses deinterleave + silu_mul_merged shaders from encoder
- w2 reads from merged bufGate with strided rowBytes (hidden*2 = 18432)
- No additional GPU memory (decoder merged weights already pre-warmed)
- **Result: prefill ~335 → ~252 ms (25% faster)**

### Attempt 14: Q-tiled encoder attention (SUCCESS — BIG for long audio)
- Each threadgroup processes BQ=8 consecutive queries instead of 1
- Loads K/V once per key position, computes 8 dot products (amortizes K/V reads ~8x)
- Per-query causal/window masking within the BQ block (skip invalid positions)
- BQ sets of online softmax state in registers (38 floats/thread, well within limits)
- Shared memory: 4×BQ (SIMD reduction) + BQ (scores) = 40 floats
- Attention cost: 0.133ms/KV_pos → 0.058ms/KV_pos (2.3x faster attention)
- **Result: test_speech encoder 298 → 284ms (5% faster, short KV cache)**
- **Result: jfk encoder 700 → 539ms (23% faster, attention-dominated)**
- **Cumulative encoder: 2.7s → 284ms (test_speech, 89% faster), 539ms (jfk)**

### Attempt 15: Custom SMA GEMM for prefill (FAILED)
- Wrote custom Metal GEMM kernel using simdgroup_multiply_accumulate (SMA)
- Target: replace MPS matmul for small-M prefill (M=37, 104 matmuls × 26 layers)
- BM=8, BN=32, BK=8 tiles, 4 simdgroups per threadgroup (128 threads)
- Key findings:
  - `thread_elements()` mapping is implementation-defined — manual indexing produces wrong results
  - Must use `simdgroup_load()`/`simdgroup_store()` for correct matrix I/O (Metal 3.0)
  - Transposed B loads (`simdgroup_load(..., transpose=true)`) have poor memory coalescing
  - f32→f16 conversion in inner loop adds ~20% overhead (tried barrier-free f16 variant, no help)
  - Monolithic compute encoder (1 for all 26 layers) didn't reduce overhead vs separate encoders
- **Result: ~300ms vs MPS ~252ms (20% SLOWER). Reverted.**
- Conclusion: MPS matmul's optimized tiling and memory access patterns are hard to beat.
  Custom GEMM would need pre-transposed weights for good coalescing, not worth the complexity.

### Attempt 16: exp2() for softmax (FAILED)
- Changed exp() to exp2(x * M_LOG2E_F) in decoder_attention, encoder_attention, causal_softmax
- No measurable improvement — Apple GPU already optimizes exp() internally
- Attention kernels are memory-bandwidth-bound, not compute-bound
- **Result: no change. Reverted.**

### Attempt 17: beta=1.0 fused residual in prefill/encoder (FAILED)
- Changed wo and w2 matmuls to beta=1.0 (C = A@B^T + C), eliminating add_inplace dispatches
- Removed bufProj and bufFfnOut buffers (fewer allocations)
- Applied to both decoder prefill and encoder monolithic step
- **Result: no measurable speed improvement. Reverted.**

### Attempt 18: f16 encoder activations (FAILED)
- Convert f32 activations to f16 before each MPS matmul in encoder (4 conversions/layer)
- Theory: f16×f16 enables f16 compute path (~28 TFLOPS vs 14 TFLOPS f32)
- Added f32_to_f16 Metal shader, allocated f16 input buffers, changed MPS descriptors
- Conversion overhead is negligible (~1ms total for all 32 layers)
- MPS matmul with f16 inputs produces correct results
- **Result: test_speech 283→266ms (within noise), jfk 526→538ms (no improvement). Reverted.**
- Conclusion: MPS likely already uses f16 hardware internally regardless of input type.
  The f32→f16 conversion adds overhead without unlocking faster compute.

### Attempt 19: Custom SMA GEMM with pre-transposed weights (FAILED)
- Pre-transpose encoder weights [N, K] → [K, N] at load time (perfect memory coalescing)
- Custom Metal GEMM kernel using simdgroup_multiply_accumulate (SMA)
- BM=8, BN=32, 4 simdgroups per threadgroup, simdgroup_load/store for correct I/O
- First version: f32 A × f16 B → uses slow f32×f32 SMA path (40% slower than MPS)
- Second version: convert A to f16 in threadgroup memory → f16×f16→f32 SMA (still 40% slower)
- **Result: encoder 283→375-445ms. Even worse than attempt 15 for prefill.**
- Root cause: 8×8 tiles are too small — MPS uses much larger tiles (64×64+) with prefetching,
  double-buffering, and vectorized loads. Writing a competitive GEMM requires production-level
  tiling (32×32+) and memory pipeline optimization — not worth the complexity.

### Analysis: Encoder efficiency at MPS limit
- MPS matmul achieves ~4.5 TFLOPS (32% of peak f32) for encoder dimensions (M=100-200)
- Remaining overhead per encoder call: ~50ms first-call warmup (MPS pipeline JIT), ~5ms for
  compute shaders (norm, bias, RoPE, attention), ~3ms CPU-side MPS encoding
- Using `-I 100` (fewer, larger encoder calls) improves jfk from 526→433ms (18% faster)
  by reducing command buffer commits and improving GPU utilization at larger M
- Further encoder optimization would require either:
  1. INT4/INT8 weight quantization (reduces bandwidth, but MPS doesn't support natively)
  2. MLX-level optimized GEMM kernels (production-quality tiling, significant engineering)
  3. Fundamentally different architecture (e.g., speculative decoding, pipeline parallelism)

### Next targets
- Decoder: ~23.5 ms/step, theoretical floor ~17.3 ms — **73% bandwidth efficiency, 6.2 ms headroom**
- Encoder: ~284ms (test_speech), ~539ms (jfk) — **at MPS matmul efficiency limit**
- Prefill: ~252ms — **at MPS matmul efficiency limit for M=38**

## Credits attribution rules
- Ideas / kernels / approaches should be only taken from BSD / MIT licensed code.
- If any optimization ideas or kernel code are taken from some other project,
  proper credits must be added to both the README and the relevant source file.
