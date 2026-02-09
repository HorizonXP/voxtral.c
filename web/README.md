# Voxtral Web Server (MVP)

This folder contains a small self-hostable web server that exposes:

- `POST /v1/audio/transcriptions` (OpenAI-ish batch transcription)
- `GET  /v1/audio/transcriptions/realtime` (WebSocket realtime dictation)
- a minimal browser mic demo at `/`

The MVP shells out to the `voxtral` CLI for inference:

- Batch: a small pool of persistent `voxtral --worker` subprocesses (keeps the model loaded across requests).
- Realtime: one `voxtral` process per WebSocket session.

## Prereqs

- Build `voxtral` (CUDA recommended):
  - `make cuda` (or `make` for CPU-only)
- Download the model:
  - `./download_model.sh`
- Install `ffmpeg` (used for decoding arbitrary uploads to `16kHz mono WAV`).
- Python 3.12+.

## Setup

```bash
python3 -m venv web/.venv
source web/.venv/bin/activate
pip install -r web/requirements.txt
```

## Run

From the repo root:

```bash
# Example: enable the fast CUDA path for voxtral subprocesses.
env VOX_CUDA_FAST=1 python3 web/server.py
```

Batch worker knobs:

- `--batch-workers N` (default: 1)
- `--batch-timeout SECS` (default: 600)
- `--no-batch-warmup` (disable startup warmup that avoids first-request autotune/graph-capture latency)

Then open:

- Browser demo: `http://127.0.0.1:8000/`
- Health: `http://127.0.0.1:8000/health`

### Auth (optional)

If you set `VOXTRAL_API_KEY` (or pass `--api-key`), requests must include:

- HTTP: `Authorization: Bearer <token>`
- WebSocket (browser-friendly): `?access_token=<token>` (query param fallback)

## Batch API

Example:

```bash
curl -sS \
  -F file=@samples/test_speech.wav \
  -F model=voxtral \
  -F response_format=json \
  http://127.0.0.1:8000/v1/audio/transcriptions
```

Response (default):

```json
{"text":"..."}
```

## Realtime API

WebSocket endpoint:

- `ws://127.0.0.1:8000/v1/audio/transcriptions/realtime?interval=0.5`

The client sends binary frames containing raw `PCM16LE @ 16kHz mono` audio.
See `web/PROTOCOL.md` for the wire format and events.
