# Voxtral Realtime 4B Pure C Implementation

This is a C implementation of the inference pipeline for the [Mistral AI's Voxtral Realtime 4B model](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602). It has zero external dependencies beyond the C standard library. The MPS inference is decently fast, while the BLAS acceleration is usable but slow (it continuously convert the bf16 weights to fp32).

Audio processing uses a chunked encoder with overlapping windows, bounding memory usage regardless of input length. Audio can also be piped from stdin (`--stdin`), making it easy to transcode and transcribe any format via ffmpeg. A streaming C API (`vox_stream_t`) lets you feed audio incrementally and receive token strings as they become available.

**More testing needed:** please note that this project was mostly tested against few samples, and likely requires some more work to be production quality. However the hard part, to understand the model inference and reproduce the inference pipeline, is here, so the rest likely can be done easily. Testing it against very long transcriptions, able to stress the KV cache circular buffer, will be a useful task.

![demo](samples/demo.gif)

## Motivations (and some rant)

**Thank you to Mistral** for releasing such a great model in an Open Weights fashion. However, the author of this project believes that limiting the inference to a partnership with vLLM, without providing a self-contained reference implementation in Python, limits the model's actual reach and the potential good effects it could have. For this reason, this project was created: it provides both a pure C inference engine and a simple, self-contained Python reference implementation (`python_simple_implementation.py`) that anyone can read and understand without digging through the vLLM codebase.

## Quick Start

```bash
# Build (choose your backend)
make mps       # Apple Silicon (fastest)
# or: make blas    # Intel Mac / Linux with OpenBLAS
# or: make cuda    # NVIDIA CUDA/cuBLAS (Linux/WSL2)

# Download the model (~8.9GB)
./download_model.sh

# Transcribe audio (tokens stream to stdout as generated)
./voxtral -d voxtral-model -i audio.wav

# Pipe any format via ffmpeg
ffmpeg -i audio.mp3 -f s16le -ar 16000 -ac 1 - 2>/dev/null | \
    ./voxtral -d voxtral-model --stdin
```

That's it. No Python runtime, no CUDA toolkit, no `mistral_common` or vLLM required at inference time.

### Python Reference Implementation

A self-contained Python implementation is also provided for reading and understanding the model:

```bash
pip install torch safetensors soundfile soxr
python python_simple_implementation.py voxtral-model audio.wav
```

This requires just PyTorch and a few standard libraries.

## Features

- **Zero dependencies**: Pure C implementation, works standalone for MPS. BLAS required for other targets (OpenBLAS on Linux).
- **Metal GPU acceleration**: Automatic on Apple Silicon Macs with fused GPU operations and batched attention.
- **Streaming output**: Tokens are printed to stdout as they are generated, word by word.
- **Streaming C API**: Feed audio incrementally, get token strings back as they become available.
- **Memory-mapped weights**: BF16 weights are mmap'd directly from safetensors, loading is near-instant.
- **WAV input**: Supports 16-bit PCM WAV files at any sample rate (auto-resampled to 16kHz).
- **Chunked encoder**: Processes audio in overlapping chunks, bounding memory regardless of length.
- **Rolling KV cache**: Decoder KV cache is automatically compacted when it exceeds the sliding window (8192 positions), capping memory usage and allowing unlimited-length audio.

## Usage

### Basic Transcription

```bash
./voxtral -d voxtral-model -i recording.wav
```

Tokens stream to stdout as they are generated. By default, timing info is printed to stderr. Use `--silent` or `--debug` to control verbosity:

```bash
./voxtral -d voxtral-model -i samples/test_speech.wav --silent    # no stderr output
./voxtral -d voxtral-model -i samples/test_speech.wav --debug     # per-layer/per-chunk details
```

### Reading Audio from Stdin

The **`--stdin` flag** reads audio from standard input instead of a file. The format is auto-detected: if the data starts with a RIFF header it is parsed as WAV, otherwise it is treated as **raw signed 16-bit little-endian, 16 kHz, mono** (`s16le`).

This makes it trivial to transcode any audio/video format on the fly with ffmpeg:

```bash
# Transcribe an MP3 file
ffmpeg -i podcast.mp3 -f s16le -ar 16000 -ac 1 - 2>/dev/null | \
    ./voxtral -d voxtral-model --stdin

# Pipe a WAV directly (auto-detected)
cat recording.wav | ./voxtral -d voxtral-model --stdin
```

`--stdin` and `-i` are mutually exclusive.

To convert files to WAV format, just use `ffmpeg`:

    ffmpeg -i input.ogg output.wav

The above command line works for many file types, not just for OGG files, of course.
There are two example wave files under the `samples` directory.

### C API

The library exposes a streaming API (`vox_stream_t`) that works for both offline and real-time use. You feed audio samples and retrieve decoded token strings as they become available.

**Offline transcription** — feed all audio, then collect results:

```c
#include "voxtral.h"

vox_ctx_t *ctx = vox_load("voxtral-model");

/* Load audio (your own code, or use vox_load_wav) */
int n_samples;
float *samples = vox_load_wav("audio.wav", &n_samples);

/* Transcribe */
vox_stream_t *s = vox_stream_init(ctx);
vox_stream_feed(s, samples, n_samples);
vox_stream_finish(s);

/* Collect token strings */
const char *tokens[64];
int n;
while ((n = vox_stream_get(s, tokens, 64)) > 0) {
    for (int i = 0; i < n; i++)
        printf("%s", tokens[i]);
}
printf("\n");

vox_stream_free(s);
free(samples);
vox_free(ctx);
```

**Real-time streaming** — feed audio incrementally, retrieve tokens as they arrive:

```c
vox_stream_t *s = vox_stream_init(ctx);

while (have_more_audio()) {
    float chunk[4096];
    int n_read = read_audio(chunk, 4096);
    vox_stream_feed(s, chunk, n_read);

    const char *tokens[16];
    int n;
    while ((n = vox_stream_get(s, tokens, 16)) > 0) {
        for (int i = 0; i < n; i++)
            printf("%s", tokens[i]);
        fflush(stdout);
    }
}

vox_stream_finish(s);
const char *tokens[16];
int n;
while ((n = vox_stream_get(s, tokens, 16)) > 0) {
    for (int i = 0; i < n; i++)
        printf("%s", tokens[i]);
}
printf("\n");

vox_stream_free(s);
```

`feed()` runs the mel spectrogram, encoder, and decoder on available data, queuing output tokens. `finish()` adds padding and processes remaining audio. `get()` retrieves pending tokens — call it after each `feed()` or whenever convenient. If `vox_stream_output` is set, tokens are also printed there as they are generated. Token string pointers returned by `vox_stream_get()` are valid until `vox_stream_free()`.

Use `vox_set_processing_interval(s, seconds)` to batch encoder/decoder work. When set, `feed()` accumulates audio but only runs the encoder/decoder after at least the specified duration of new audio has been fed (0 = process on every `feed()`, the default). This can improve efficiency when feeding audio in many small chunks.

There is also a one-shot convenience function if you don't need streaming:

```c
char *text = vox_transcribe(ctx, "audio.wav");
printf("%s\n", text);
free(text);
```

## Building

Choose a backend when building:

```bash
make            # Show available backends
make blas       # BLAS acceleration (Accelerate on macOS, OpenBLAS on Linux)
make cuda       # NVIDIA CUDA/cuBLAS acceleration (Linux/WSL2)
make mps        # Apple Silicon Metal GPU (fastest, macOS only)
```

**Recommended:**
- macOS Apple Silicon: `make mps`
- macOS Intel: `make blas`
- Linux with NVIDIA GPU (including WSL2): `make cuda`
- Linux CPU-only / OpenBLAS: `make blas`

For `make blas` on Linux, install OpenBLAS first:
```bash
# Ubuntu/Debian
sudo apt install libopenblas-dev

# Fedora
sudo dnf install openblas-devel
```

Other targets:
```bash
make clean      # Clean build artifacts
make info       # Show available backends for this platform
make inspect    # Build safetensors weight inspector
```

## Model Download

Download model weights (~8.9GB) from HuggingFace:

```bash
./download_model.sh
```

This downloads to `./voxtral-model/` containing:
- `consolidated.safetensors` — all weights, BF16 (~8.9GB)
- `tekken.json` — Tekken tokenizer vocabulary (~15MB)
- `params.json` — model configuration

The model is [Apache-2.0 licensed](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602).

## How Fast Is It?

Benchmarks on **Apple M3 Max** (128GB RAM):

| Backend | Encoder (3.6s audio) | Decoder (ms/token) | Total |
|---------|---------------------|--------------------|-------|
| MPS | 2.4s | 43 ms | ~5s |
| BLAS | ~8s | 335 ms | ~28s |

The MPS backend uses fused GPU operations (QKV, FFN, batched attention) to minimize command buffer overhead. The BLAS backend uses Accelerate's multi-threaded sgemm with on-the-fly BF16→F32 conversion.

Longer audio scales linearly with the encoder (O(n) with sliding window attention) and the decoder (one token per 80ms of audio).

## Model Architecture

Voxtral Realtime 4B is a streaming speech-to-text model with ~4B parameters:

**Pipeline:**
```
WAV → 16kHz → Mel Spectrogram → Conv Stem → Encoder → Downsample 4x → Adapter → Decoder → Tokens
```

| Component | Architecture |
|-----------|-------------|
| Audio Encoder | 32-layer causal transformer, 1280 dim, 32 heads, sliding window 750 |
| Adapter | Linear(5120→3072) → GELU → Linear(3072→3072) |
| LLM Decoder | 26-layer transformer (Ministral-3 based), 3072 dim, GQA (32 heads / 8 KV) |

| Parameter | Value |
|-----------|-------|
| Total parameters | ~4B (0.6B encoder + 3.4B decoder) |
| Weight format | BF16 |
| Vocab size | 131,072 (Tekken tokenizer) |
| Audio frame rate | 12.5 Hz (1 token = 80ms) |
| Max audio length | Unlimited (rolling KV cache) |
| Supported languages | EN, ES, FR, PT, HI, DE, NL, IT, AR, RU, ZH, JA, KO |

## Memory Requirements

| Component | Size |
|-----------|------|
| Model weights (mmap'd) | 8.9 GB on disk, mapped on-demand |
| MPS GPU weight cache | ~8.4 GB (BF16→F16 cached on GPU) |
| KV cache (decoder) | ~1.8 GB max (rolling, capped at sliding window) |
| Working buffers | ~200 MB |

## License

MIT

## CUDA on Windows 11 + WSL2 (Ubuntu)

This repository now includes a CUDA backend that uses **cuBLAS** for large matrix multiplies (`make cuda`).

### Prerequisites

1. Install the latest NVIDIA Windows driver with WSL CUDA support.
2. Install WSL2 + Ubuntu 22.04+.
3. Inside Ubuntu, install the CUDA toolkit (includes `nvcc`, `cublas`, and runtime libraries).

### Build

```bash
make cuda
```

### Verify CUDA visibility in WSL2

```bash
nvidia-smi
```

If your RTX 3080 Ti is visible there, `make cuda` should work and Voxtral will offload large GEMMs to the GPU.
