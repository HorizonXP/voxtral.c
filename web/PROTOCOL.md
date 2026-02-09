# Realtime Dictation Protocol (MVP)

This is the v1 wire protocol used by `web/server.py` for realtime dictation.

## Transport

- WebSocket upgrade:
  - `GET /v1/audio/transcriptions/realtime?interval=<seconds>[&access_token=<token>]`

## Audio Format (client -> server)

Binary WebSocket messages contain **raw audio bytes**:

- Encoding: signed 16-bit PCM (`s16le`)
- Sample rate: `16000`
- Channels: `1` (mono)
- Framing: arbitrary chunk sizes; the server simply concatenates bytes and feeds stdin of `voxtral --stdin`.

No container headers (WAV/OGG) are used in realtime mode.

## Control Messages (client -> server)

Text WebSocket messages are JSON objects.

Supported:

- `{"type":"stop"}`
  - Closes `voxtral` stdin, waits for the process to exit, emits `transcript.done`, then closes the WebSocket.

Unknown control messages are ignored.

## Events (server -> client)

The server sends JSON text messages:

- `{"type":"session.created"}`
  - Sent after the `voxtral` subprocess is started and output pumps are running.
- `{"type":"transcript.delta","text":"..."}`
  - Incremental transcript fragments (as produced by `voxtral` stdout).
- `{"type":"transcript.done","text":"..."}`
  - Final transcript (concatenation of all deltas), sent after `stop`.
- `{"type":"error","message":"...","stderr_tail":"..."}`
  - Best-effort error reporting if the subprocess fails.

## Notes / Limitations

- MVP implementation is **not** a full OpenAI Realtime protocol. It's a small WS protocol designed to be easy to embed.
- One realtime session maps to one `voxtral` subprocess. Concurrency/scaling is tracked separately in Issue `#41`.
- For accurate resampling and bandwidth savings, a future iteration may accept Opus/WebRTC (`#40`).

