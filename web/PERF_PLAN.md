# Voxtral Web Server Performance Plan (Batch / Paste Mode)

Goal: make `POST /v1/audio/transcriptions` feel as fast as the hosted API by
minimizing per-request overhead and then attacking true inference bottlenecks.

## Current State (After "Hot Batch Worker" Work)

Batch endpoint uses a persistent `./voxtral --worker` subprocess pool so model
load is paid once on startup, not per request.

Batch pipeline (current):

1. Upload multipart body -> temp file `upload.bin`
2. Decode to `PCM16LE @16kHz mono`:
   - Fast-path: if upload is already WAV PCM16 mono 16kHz, decode in-process via `wave`
   - Otherwise: spawn `ffmpeg` to decode/resample to raw PCM
3. Send PCM bytes to a hot worker via `P\t<id>\t<nbytes>\n<pcm...>`
4. Worker runs inference and returns `R\t<id>\tOK\t<text>`

The response includes timing headers:

- `X-Voxtral-Upload-Ms`
- `X-Voxtral-Decode-Ms`
- `X-Voxtral-Infer-Ms`
- `X-Voxtral-Total-Ms`
- `X-Voxtral-Audio-Sec`
- `X-Voxtral-xRT` (audio seconds / infer seconds; higher is faster-than-real-time)

## Benchmarking Method

We benchmark two local clips:

- `samples/test_speech.wav`
- `samples/I_have_a_dream.ogg`

And optionally benchmark against the hosted API (Mistral) for the same clips to
establish a target.

Script: `scripts/benchmark_local_vs_mistral.sh`

## Next Performance Workstreams

Ordered roughly by "likely to improve perceived latency" first, then throughput.

1. Reduce/avoid `ffmpeg` process spawn cost for non-WAV inputs
   - Option A: keep `ffmpeg` but use a small persistent decode worker process (pool)
   - Option B: decode in-process (Python libs) for common formats, fall back to ffmpeg
   - Goal: cut `decode_ms` for OGG/MP3 inputs (currently dominated by ffmpeg startup)

2. Stream upload -> decode -> worker (avoid buffering large PCM in memory)
   - Today the server buffers the entire decoded PCM in memory before sending it
   - Implement chunked streaming to worker:
     - Protocol: `S\t<id>\n` + repeated `D\t<id>\t<nbytes>\n<pcm...>` + `F\t<id>\n`
     - Worker: incremental `vox_stream_feed` as chunks arrive, `vox_stream_finish` on `F`
   - Goal: reduce memory footprint and improve tail latency for long clips

3. GPU profiling + kernel/GEMM tuning (real inference bottleneck)
   - Use `nsys` / `ncu` on hot worker runs for both short and long clips
   - Identify dominant kernels:
     - Encoder conv stem / attention / GEMMs
     - Decoder prefill vs per-token step
   - Tuning knobs already present in `voxtral_cuda.c`:
     - cuBLASLt autotune, algo selection, attention variants, fused logits, graphs
   - Goal: reduce `infer_ms` (this is the main remaining cost)

4. Quantization (bigger speedup, higher risk)
   - INT8/FP8 for LM head / decoder blocks can reduce compute and memory bandwidth
   - Requires accuracy regression checks (`scripts/accuracy_regression.sh`) and careful gating

## Validation / Definition of Done

For each change:

1. Run `scripts/benchmark_local_vs_mistral.sh` and record:
   - local `infer_ms` and `X-Voxtral-xRT` for both clips
   - (optional) hosted API wall time for both clips
2. Ensure transcripts still look correct (spot check) and regressions are caught.
3. Keep changes gated and documented (env flags / CLI flags).

