#!/usr/bin/env python3
"""
voxtral web server (local/self-hosted)

Goals:
- Provide an OpenAI-compatible batch transcription endpoint:
    POST /v1/audio/transcriptions  (multipart/form-data: file=..., model=...)
- Provide a simple WebSocket realtime endpoint for dictation:
    WS /v1/audio/transcriptions/realtime

Implementation note:
- This server shells out to the existing `./voxtral` CLI for inference.
- Batch requests use a persistent `./voxtral --worker` subprocess pool so the
  model stays loaded across requests (improves latency for paste mode).
- Realtime WS sessions still spawn one `voxtral` process per session.
"""

from __future__ import annotations

import argparse
import asyncio
import codecs
import json
import os
import pathlib
import shutil
import tempfile
import time
import wave
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from aiohttp import WSMsgType, web


@dataclass(frozen=True)
class ServerConfig:
    host: str
    port: int
    voxtral_bin: str
    model_dir: str
    default_interval_s: float
    max_sessions: int
    batch_workers: int
    batch_timeout_s: float
    batch_startup_timeout_s: float
    batch_warmup: bool
    api_key: Optional[str]
    extra_env: Dict[str, str]


def _now_ms() -> int:
    return int(time.time() * 1000)


def _require_auth(cfg: ServerConfig, request: web.Request) -> None:
    if not cfg.api_key:
        return
    token = ""
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        token = auth[len("Bearer ") :].strip()
    # Browser WebSockets cannot set custom headers; allow query param for demos.
    if not token:
        token = request.query.get("access_token", "").strip()
    if token == cfg.api_key:
        return
    raise web.HTTPUnauthorized(
        text="Unauthorized (set Authorization: Bearer <token>)\n",
        content_type="text/plain",
    )


def _cors_headers() -> Dict[str, str]:
    # Demo-friendly. In real deployments, restrict origin(s).
    return {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
        "Access-Control-Allow-Headers": "Authorization,Content-Type",
    }


@web.middleware
async def cors_middleware(request: web.Request, handler):
    if request.method == "OPTIONS":
        return web.Response(status=204, headers=_cors_headers())
    resp = await handler(request)
    for k, v in _cors_headers().items():
        resp.headers.setdefault(k, v)
    return resp


async def _ws_close_with_timeout(ws: web.WebSocketResponse, timeout_s: float = 2.0) -> None:
    # A misbehaving client can stall the close handshake; don't let this handler hang.
    try:
        await asyncio.wait_for(ws.close(), timeout=timeout_s)
    except asyncio.TimeoutError:
        try:
            ws.force_close()
        except Exception:
            pass
    except Exception:
        try:
            ws.force_close()
        except Exception:
            pass


async def _run_ffmpeg_to_wav(in_path: str, out_path: str) -> Tuple[int, bytes]:
    # Decode any input format ffmpeg supports to 16kHz mono WAV.
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        in_path,
        "-ac",
        "1",
        "-ar",
        "16000",
        out_path,
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    _out, err = await proc.communicate()
    return proc.returncode, err or b""


async def _run_ffmpeg_to_pcm16le_16k_mono(in_path: str) -> Tuple[int, bytes, bytes]:
    # Decode any input format ffmpeg supports to raw PCM16LE @16kHz mono on stdout.
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        in_path,
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "s16le",
        "-",
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    out, err = await proc.communicate()
    return proc.returncode, out or b"", err or b""

def _write_silence_wav_16k_mono(path: str, *, seconds: float = 1.0) -> None:
    n = int(max(0.0, seconds) * 16000)
    # 16-bit PCM little-endian, mono, 16kHz.
    pcm = b"\x00\x00" * n
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(pcm)


def _try_read_wav_pcm16le_16k_mono(path: str) -> Optional[bytes]:
    # Fast-path: avoid spawning ffmpeg if the upload is already the right WAV format.
    try:
        with wave.open(path, "rb") as wf:
            if wf.getnchannels() != 1:
                return None
            if wf.getsampwidth() != 2:
                return None
            if wf.getframerate() != 16000:
                return None
            return wf.readframes(wf.getnframes())
    except Exception:
        return None


def _format_transcription_response(
    *,
    text: str,
    response_format: str,
    timings_ms: Optional[Dict[str, float]] = None,
) -> web.StreamResponse:
    text = (text or "").strip()

    if response_format == "text":
        return web.Response(text=text + "\n", content_type="text/plain")
    if response_format in ("json", "", None):
        return web.json_response({"text": text})
    if response_format == "verbose_json":
        # Minimal "verbose" response for compatibility.
        payload = {
            "task": "transcribe",
            "text": text,
            "segments": [],
        }
        if timings_ms:
            payload["timings_ms"] = timings_ms
        return web.json_response(payload)

    raise web.HTTPBadRequest(
        text="Unsupported response_format (use: json|text|verbose_json)\n",
        content_type="text/plain",
    )


class BatchWorkerError(RuntimeError):
    pass


class BatchWorker:
    def __init__(self, *, cfg: ServerConfig, idx: int):
        self._cfg = cfg
        self._idx = idx
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._stderr_task: Optional[asyncio.Task] = None
        self._stderr_tail = bytearray()
        self._stderr_tail_cap = 64 * 1024
        self._next_id = 0

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    def stderr_tail_text(self) -> str:
        return bytes(self._stderr_tail).decode("utf-8", errors="replace")

    async def start(self) -> None:
        env = os.environ.copy()
        env.update(self._cfg.extra_env)
        cmd = [
            self._cfg.voxtral_bin,
            "-d",
            self._cfg.model_dir,
            "--silent",
            "--worker",
        ]
        self._proc = await asyncio.create_subprocess_exec(
            *cmd,
            env=env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        async def pump_stderr() -> None:
            assert self._proc is not None
            try:
                while True:
                    data = await self._proc.stderr.read(4096)  # type: ignore[union-attr]
                    if not data:
                        break
                    self._stderr_tail.extend(data)
                    if len(self._stderr_tail) > self._stderr_tail_cap:
                        del self._stderr_tail[: len(self._stderr_tail) - self._stderr_tail_cap]
            except Exception:
                pass

        self._stderr_task = asyncio.create_task(pump_stderr())

        # Wait for READY (protocol handshake) so /health can report true readiness.
        assert self._proc.stdout is not None
        deadline_s = self._cfg.batch_startup_timeout_s
        try:
            while True:
                line = await asyncio.wait_for(
                    self._proc.stdout.readline(), timeout=deadline_s  # type: ignore[union-attr]
                )
                if not line:
                    raise BatchWorkerError("worker exited during startup")
                s = line.decode("utf-8", errors="replace").strip()
                if s == "READY":
                    return
                # Ignore any stray stdout lines; stderr is captured separately.
        except asyncio.TimeoutError as e:
            raise BatchWorkerError("timeout waiting for READY from worker") from e

    async def close(self) -> None:
        if self._proc is None:
            return
        proc = self._proc
        self._proc = None
        try:
            if proc.stdin:
                try:
                    proc.stdin.write(b"Q\n")
                    await proc.stdin.drain()
                except Exception:
                    pass
                try:
                    proc.stdin.close()
                except Exception:
                    pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
        finally:
            if self._stderr_task and not self._stderr_task.done():
                self._stderr_task.cancel()
            self._stderr_task = None

    async def _read_reply(self, *, req_id: int, timeout_s: float) -> str:
        if self._proc is None or self._proc.returncode is not None:
            raise BatchWorkerError("worker not running")
        if self._proc.stdout is None:
            raise BatchWorkerError("worker pipes unavailable")

        try:
            while True:
                line = await asyncio.wait_for(
                    self._proc.stdout.readline(), timeout=timeout_s  # type: ignore[union-attr]
                )
                if not line:
                    raise BatchWorkerError("worker exited while waiting for response")
                s = line.decode("utf-8", errors="replace").rstrip("\r\n")
                if s == "READY":
                    continue
                if not s.startswith("R\t"):
                    continue
                parts = s.split("\t", 3)
                if len(parts) < 4:
                    continue
                _tag, rid_s, status, payload = parts
                try:
                    rid = int(rid_s)
                except Exception:
                    continue
                if rid != req_id:
                    # Shouldn't happen (we serialize per worker), but be defensive.
                    continue
                if status == "OK":
                    return payload.strip()
                if status == "ERR":
                    raise BatchWorkerError(payload.strip() or "transcription failed")
                raise BatchWorkerError(f"unknown status from worker: {status}")
        except asyncio.TimeoutError as e:
            raise BatchWorkerError(
                f"timeout waiting for worker response ({timeout_s}s)"
            ) from e

    async def transcribe(self, *, wav_path: str, timeout_s: float) -> str:
        if self._proc is None or self._proc.returncode is not None:
            raise BatchWorkerError("worker not running")
        if self._proc.stdin is None:
            raise BatchWorkerError("worker pipes unavailable")

        self._next_id += 1
        req_id = self._next_id
        cmd = f"T\t{req_id}\t{wav_path}\n".encode("utf-8", errors="strict")
        try:
            self._proc.stdin.write(cmd)
            await self._proc.stdin.drain()
        except Exception as e:
            raise BatchWorkerError(f"failed to write request to worker: {e}") from e

        return await self._read_reply(req_id=req_id, timeout_s=timeout_s)

    async def transcribe_pcm16le_16k_mono(self, *, pcm: bytes, timeout_s: float) -> str:
        if self._proc is None or self._proc.returncode is not None:
            raise BatchWorkerError("worker not running")
        if self._proc.stdin is None:
            raise BatchWorkerError("worker pipes unavailable")

        if (len(pcm) & 1) != 0:
            raise BatchWorkerError("pcm payload must be an even number of bytes")

        self._next_id += 1
        req_id = self._next_id
        header = f"P\t{req_id}\t{len(pcm)}\n".encode("utf-8", errors="strict")
        try:
            self._proc.stdin.write(header)
            if pcm:
                self._proc.stdin.write(pcm)
            await self._proc.stdin.drain()
        except Exception as e:
            raise BatchWorkerError(f"failed to write request to worker: {e}") from e

        return await self._read_reply(req_id=req_id, timeout_s=timeout_s)


class BatchWorkerPool:
    def __init__(self, *, cfg: ServerConfig):
        self._cfg = cfg
        self._q: asyncio.Queue[BatchWorker] = asyncio.Queue()
        self._workers: list[BatchWorker] = []

    async def start(self) -> None:
        n = max(0, int(self._cfg.batch_workers))
        for i in range(n):
            w = BatchWorker(cfg=self._cfg, idx=i)
            await w.start()
            self._workers.append(w)
            self._q.put_nowait(w)

    async def close(self) -> None:
        for w in self._workers:
            try:
                await w.close()
            except Exception:
                pass
        self._workers.clear()
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except Exception:
                break

    def ready_count(self) -> int:
        return sum(1 for w in self._workers if w.alive)

    async def transcribe(self, *, wav_path: str) -> str:
        if not self._workers:
            raise BatchWorkerError("batch workers disabled")
        w = await self._q.get()
        try:
            if not w.alive:
                await w.close()
                await w.start()
            return await w.transcribe(wav_path=wav_path, timeout_s=self._cfg.batch_timeout_s)
        finally:
            self._q.put_nowait(w)

    async def transcribe_pcm16le_16k_mono(self, *, pcm: bytes) -> str:
        if not self._workers:
            raise BatchWorkerError("batch workers disabled")
        w = await self._q.get()
        try:
            if not w.alive:
                await w.close()
                await w.start()
            return await w.transcribe_pcm16le_16k_mono(pcm=pcm, timeout_s=self._cfg.batch_timeout_s)
        finally:
            self._q.put_nowait(w)


async def _run_voxtral_file(
    *,
    cfg: ServerConfig,
    wav_path: str,
    timeout_s: float,
) -> str:
    env = os.environ.copy()
    env.update(cfg.extra_env)

    cmd = [
        cfg.voxtral_bin,
        "-d",
        cfg.model_dir,
        "--silent",
        "-i",
        wav_path,
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except asyncio.TimeoutError:
        proc.kill()
        out, err = await proc.communicate()
        raise web.HTTPGatewayTimeout(
            text="voxtral timed out\n",
            content_type="text/plain",
        )
    if proc.returncode != 0:
        raise web.HTTPInternalServerError(
            text=f"voxtral failed (exit={proc.returncode}):\n{(err or b'').decode('utf-8', errors='replace')}\n",
            content_type="text/plain",
        )

    return (out or b"").decode("utf-8", errors="replace").strip()


async def handle_health(request: web.Request) -> web.Response:
    cfg: ServerConfig = request.app["cfg"]
    pool: Optional[BatchWorkerPool] = request.app.get("batch_pool")
    return web.json_response(
        {
            "ok": True,
            "ts_ms": _now_ms(),
            "batch_workers": cfg.batch_workers,
            "batch_ready": (pool.ready_count() if pool else 0),
        }
    )


async def handle_index(request: web.Request) -> web.FileResponse:
    static_dir: pathlib.Path = request.app["static_dir"]
    return web.FileResponse(static_dir / "index.html")


async def handle_transcriptions(request: web.Request) -> web.StreamResponse:
    cfg: ServerConfig = request.app["cfg"]
    _require_auth(cfg, request)

    if not request.content_type.startswith("multipart/"):
        raise web.HTTPBadRequest(
            text="Expected multipart/form-data\n", content_type="text/plain"
        )

    reader = await request.multipart()
    uploaded_path = None
    response_format = "json"

    with tempfile.TemporaryDirectory(prefix="voxtral_upload_") as td:
        t0 = time.perf_counter()
        td_path = pathlib.Path(td)
        async for part in reader:
            if part.name == "file":
                # Stream upload to disk.
                # Do not trust the client-provided filename (path traversal / overwrite).
                uploaded_path = str(td_path / "upload.bin")
                with open(uploaded_path, "wb") as f:
                    while True:
                        chunk = await part.read_chunk()
                        if not chunk:
                            break
                        f.write(chunk)
            elif part.name == "response_format":
                response_format = (await part.text()).strip()
            else:
                # Ignore other OpenAI params for now (model, language, prompt, temperature...).
                await part.read()

        if not uploaded_path:
            raise web.HTTPBadRequest(
                text="Missing form field: file\n", content_type="text/plain"
            )

        t_upload_done = time.perf_counter()

        # Batch path: prefer the hot worker pool (keeps model loaded across requests).
        pool: Optional[BatchWorkerPool] = request.app.get("batch_pool")
        audio_sec = 0.0
        if pool is not None and cfg.batch_workers > 0:
            # Decode to raw PCM16LE @16kHz mono, then send to the persistent worker.
            pcm = _try_read_wav_pcm16le_16k_mono(uploaded_path)
            if pcm is None:
                rc, pcm, err = await _run_ffmpeg_to_pcm16le_16k_mono(uploaded_path)
                if rc != 0:
                    raise web.HTTPBadRequest(
                        text=f"ffmpeg decode failed:\n{err.decode('utf-8', errors='replace')}\n",
                        content_type="text/plain",
                    )

            t_decode_done = time.perf_counter()
            audio_sec = len(pcm) / (2.0 * 16000.0) if pcm else 0.0

            try:
                text = await pool.transcribe_pcm16le_16k_mono(pcm=pcm)
            except BatchWorkerError as e:
                raise web.HTTPInternalServerError(
                    text=f"batch worker failed: {e}\n",
                    content_type="text/plain",
                )
        else:
            # Fallback (no persistent workers): decode to WAV then spawn `./voxtral -i`.
            wav_path = str(td_path / "audio_16k_mono.wav")
            rc, err = await _run_ffmpeg_to_wav(uploaded_path, wav_path)
            if rc != 0:
                raise web.HTTPBadRequest(
                    text=f"ffmpeg decode failed:\n{err.decode('utf-8', errors='replace')}\n",
                    content_type="text/plain",
                )
            t_decode_done = time.perf_counter()
            try:
                with wave.open(wav_path, "rb") as wf:
                    audio_sec = wf.getnframes() / float(wf.getframerate() or 1)
            except Exception:
                audio_sec = 0.0

            text = await _run_voxtral_file(
                cfg=cfg, wav_path=wav_path, timeout_s=cfg.batch_timeout_s
            )

        t_infer_done = time.perf_counter()

        timings_ms = {
            "upload_ms": (t_upload_done - t0) * 1000.0,
            "decode_ms": (t_decode_done - t_upload_done) * 1000.0,
            "infer_ms": (t_infer_done - t_decode_done) * 1000.0,
            "total_ms": (t_infer_done - t0) * 1000.0,
        }

        resp = _format_transcription_response(
            text=text,
            response_format=response_format,
            timings_ms=timings_ms,
        )
        # Non-invasive timing info for scripts.
        resp.headers["X-Voxtral-Upload-Ms"] = f"{timings_ms['upload_ms']:.3f}"
        resp.headers["X-Voxtral-Decode-Ms"] = f"{timings_ms['decode_ms']:.3f}"
        resp.headers["X-Voxtral-Infer-Ms"] = f"{timings_ms['infer_ms']:.3f}"
        resp.headers["X-Voxtral-Total-Ms"] = f"{timings_ms['total_ms']:.3f}"
        resp.headers["X-Voxtral-Audio-Sec"] = f"{audio_sec:.6f}"
        if audio_sec > 0.0 and timings_ms["infer_ms"] > 0.0:
            xrt = audio_sec / (timings_ms["infer_ms"] / 1000.0)
            resp.headers["X-Voxtral-xRT"] = f"{xrt:.3f}"
        return resp


async def handle_ws_realtime(request: web.Request) -> web.StreamResponse:
    cfg: ServerConfig = request.app["cfg"]
    _require_auth(cfg, request)

    sem: asyncio.Semaphore = request.app["session_sem"]
    acquired = False
    ws = web.WebSocketResponse(heartbeat=20.0, max_msg_size=16 * 1024 * 1024)
    proc: Optional[asyncio.subprocess.Process] = None
    stdout_task: Optional[asyncio.Task] = None
    stderr_task: Optional[asyncio.Task] = None

    await sem.acquire()
    acquired = True
    try:
        await ws.prepare(request)

        env = os.environ.copy()
        env.update(cfg.extra_env)

        # aiohttp WebSocketResponse isn't safe for concurrent sends from multiple tasks.
        send_lock = asyncio.Lock()

        interval = cfg.default_interval_s
        try:
            if "interval" in request.query:
                interval = float(request.query["interval"])
        except Exception:
            interval = cfg.default_interval_s

        cmd = [
            cfg.voxtral_bin,
            "-d",
            cfg.model_dir,
            "--silent",
            "--stdin",
            "-I",
            f"{interval:.3f}",
        ]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                env=env,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except Exception as e:
            async with send_lock:
                await ws.send_str(
                    json.dumps(
                        {
                            "type": "error",
                            "message": f"failed to spawn voxtral subprocess: {e}",
                        }
                    )
                )
            await ws.close()
            return ws

        full_text_parts = []
        stdout_decoder = codecs.getincrementaldecoder("utf-8")()
        stderr_tail = bytearray()
        stderr_tail_cap = 32 * 1024

        async def pump_stdout() -> None:
            assert proc is not None
            try:
                while True:
                    data = await proc.stdout.read(4096)  # type: ignore[union-attr]
                    if not data:
                        break
                    s = stdout_decoder.decode(data)
                    if not s:
                        continue
                    full_text_parts.append(s)
                    try:
                        async with send_lock:
                            await ws.send_str(
                                json.dumps({"type": "transcript.delta", "text": s})
                            )
                    except Exception:
                        break
            finally:
                try:
                    rest = stdout_decoder.decode(b"", final=True)
                    if rest:
                        full_text_parts.append(rest)
                except Exception:
                    pass

        async def pump_stderr() -> None:
            assert proc is not None
            try:
                while True:
                    data = await proc.stderr.read(4096)  # type: ignore[union-attr]
                    if not data:
                        break
                    stderr_tail.extend(data)
                    if len(stderr_tail) > stderr_tail_cap:
                        del stderr_tail[: len(stderr_tail) - stderr_tail_cap]
            except Exception:
                pass

        stdout_task = asyncio.create_task(pump_stdout())
        stderr_task = asyncio.create_task(pump_stderr())

        async with send_lock:
            await ws.send_str(json.dumps({"type": "session.created"}))

        async def finish_and_close() -> None:
            assert proc is not None
            try:
                if proc.stdin:
                    try:
                        proc.stdin.close()
                    except Exception:
                        pass
                try:
                    await asyncio.wait_for(proc.wait(), timeout=10.0)
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()

                if stdout_task:
                    await stdout_task
                if stderr_task:
                    await stderr_task

                full_text = "".join(full_text_parts).strip()
                async with send_lock:
                    await ws.send_str(
                        json.dumps({"type": "transcript.done", "text": full_text})
                    )
            finally:
                await _ws_close_with_timeout(ws)

        try:
            async for msg in ws:
                if msg.type == WSMsgType.BINARY:
                    if proc.stdin is None:
                        continue
                    try:
                        proc.stdin.write(msg.data)
                        await proc.stdin.drain()
                    except Exception:
                        break
                elif msg.type == WSMsgType.TEXT:
                    try:
                        payload = json.loads(msg.data)
                    except Exception:
                        payload = {"type": msg.data}
                    mtype = payload.get("type")
                    if mtype == "stop":
                        await finish_and_close()
                        return ws
                    # Ignore unknown control messages for now.
                elif msg.type in (WSMsgType.ERROR, WSMsgType.CLOSE):
                    break
        finally:
            if proc is not None:
                try:
                    if proc.stdin:
                        proc.stdin.close()
                except Exception:
                    pass
                try:
                    await asyncio.wait_for(proc.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()

            # Try to drain output tasks (best effort).
            if stdout_task:
                try:
                    await asyncio.wait_for(stdout_task, timeout=1.0)
                except Exception:
                    pass
            if stderr_task:
                try:
                    await asyncio.wait_for(stderr_task, timeout=1.0)
                except Exception:
                    pass

            # If the process failed, surface the error to the client if still open.
            if proc is not None and proc.returncode and proc.returncode != 0 and not ws.closed:
                tail = bytes(stderr_tail).decode("utf-8", errors="replace")
                async with send_lock:
                    await ws.send_str(
                        json.dumps(
                            {
                                "type": "error",
                                "message": f"voxtral exited with {proc.returncode}",
                                "stderr_tail": tail,
                            }
                        )
                    )
                await _ws_close_with_timeout(ws)

            # If the client ended the WS without "stop", ensure we close our side too.
            if not ws.closed:
                await _ws_close_with_timeout(ws)

        return ws
    finally:
        # Ensure we don't leave dangling tasks on timeout paths.
        for t in (stdout_task, stderr_task):
            if t and not t.done():
                t.cancel()
        if acquired:
            sem.release()


def _parse_env_kv(kvs) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for kv in kvs or []:
        if "=" not in kv:
            raise ValueError(f"--env expects KEY=VALUE (got: {kv})")
        k, v = kv.split("=", 1)
        out[k] = v
    return out


def main() -> int:
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description="voxtral OpenAI-compatible web server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument(
        "--voxtral-bin",
        default=str(repo_root / "voxtral"),
        help="Path to voxtral binary",
    )
    ap.add_argument(
        "--model-dir",
        default=str(repo_root / "voxtral-model"),
        help="Path to voxtral model directory",
    )
    ap.add_argument(
        "--interval",
        type=float,
        default=0.5,
        help="Default processing interval for realtime WS sessions (-I).",
    )
    ap.add_argument(
        "--max-sessions",
        type=int,
        default=1,
        help="Max concurrent realtime WS sessions (MVP default: 1).",
    )
    ap.add_argument(
        "--batch-workers",
        type=int,
        default=1,
        help="Number of persistent batch workers for /v1/audio/transcriptions (default: 1).",
    )
    ap.add_argument(
        "--batch-timeout",
        type=float,
        default=600.0,
        help="Max seconds to wait for a batch transcription (default: 600).",
    )
    ap.add_argument(
        "--batch-startup-timeout",
        type=float,
        default=60.0,
        help="Max seconds to wait for batch worker startup (default: 60).",
    )
    ap.add_argument(
        "--batch-warmup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run a short warmup transcription on startup to avoid first-request CUDA autotune/graph-capture latency.",
    )
    ap.add_argument(
        "--api-key",
        default=os.environ.get("VOXTRAL_API_KEY", ""),
        help="If set, require Authorization: Bearer <token> (OpenAI style).",
    )
    ap.add_argument(
        "--env",
        action="append",
        default=[],
        help="Extra env var for voxtral subprocess (KEY=VALUE). Repeatable.",
    )
    args = ap.parse_args()

    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg not found in PATH (install ffmpeg)\n")
    voxtral_bin_path = pathlib.Path(args.voxtral_bin)
    if not voxtral_bin_path.is_file():
        raise SystemExit(
            f"voxtral binary not found: {voxtral_bin_path}\n"
            "Build it first (e.g. `make cuda`) and/or set --voxtral-bin.\n"
        )
    if not os.access(str(voxtral_bin_path), os.X_OK):
        raise SystemExit(f"voxtral binary is not executable: {voxtral_bin_path}\n")
    model_dir_path = pathlib.Path(args.model_dir)
    if not model_dir_path.is_dir():
        raise SystemExit(
            f"model dir not found: {model_dir_path}\n"
            "Download it first (e.g. `./download_model.sh`) and/or set --model-dir.\n"
        )

    cfg = ServerConfig(
        host=args.host,
        port=args.port,
        voxtral_bin=str(voxtral_bin_path),
        model_dir=str(model_dir_path),
        default_interval_s=args.interval,
        max_sessions=max(1, int(args.max_sessions)),
        batch_workers=max(0, int(args.batch_workers)),
        batch_timeout_s=float(args.batch_timeout),
        batch_startup_timeout_s=float(args.batch_startup_timeout),
        batch_warmup=bool(args.batch_warmup),
        api_key=(args.api_key.strip() or None),
        extra_env=_parse_env_kv(args.env),
    )

    static_dir = pathlib.Path(__file__).resolve().parent / "static"
    if not static_dir.is_dir():
        raise SystemExit(f"static dir not found: {static_dir}")

    # Raise default upload limit (aiohttp default is 1 MiB).
    app = web.Application(
        middlewares=[cors_middleware],
        client_max_size=200 * 1024 * 1024,
    )
    app["cfg"] = cfg
    app["static_dir"] = static_dir
    app["session_sem"] = asyncio.Semaphore(cfg.max_sessions)
    app["batch_pool"] = BatchWorkerPool(cfg=cfg)

    async def _on_startup(app: web.Application) -> None:
        pool: BatchWorkerPool = app["batch_pool"]
        if cfg.batch_workers > 0:
            await pool.start()
            if cfg.batch_warmup:
                print(f"warming up {cfg.batch_workers} batch worker(s)...")
                pcm = b"\x00\x00" * 16000  # 1 second of silence (PCM16LE @16kHz mono)
                for _ in range(cfg.batch_workers):
                    try:
                        await pool.transcribe_pcm16le_16k_mono(pcm=pcm)
                    except Exception as e:
                        print(f"warmup failed: {e}")
                        break

    async def _on_cleanup(app: web.Application) -> None:
        pool: BatchWorkerPool = app["batch_pool"]
        await pool.close()

    app.on_startup.append(_on_startup)
    app.on_cleanup.append(_on_cleanup)

    app.router.add_get("/", handle_index)
    app.router.add_get("/health", handle_health)
    app.router.add_post("/v1/audio/transcriptions", handle_transcriptions)
    app.router.add_get("/v1/audio/transcriptions/realtime", handle_ws_realtime)
    app.router.add_static("/static/", path=str(static_dir), show_index=False)

    print(
        f"voxtral server listening on http://{cfg.host}:{cfg.port} "
        f"(max_sessions={cfg.max_sessions}, batch_workers={cfg.batch_workers})"
    )
    if cfg.api_key:
        print("auth enabled: set Authorization: Bearer <token>")
    if cfg.extra_env:
        print(f"voxtral subprocess env overrides: {cfg.extra_env}")
    return web.run_app(app, host=cfg.host, port=cfg.port, print=None)  # type: ignore[arg-type]


if __name__ == "__main__":
    raise SystemExit(main())
