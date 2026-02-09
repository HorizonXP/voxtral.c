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
  This keeps the MVP simple. Performance and multi-session concurrency work
  is tracked separately (see PRD issue #32 / child issues).
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


async def _run_voxtral_file(
    *,
    cfg: ServerConfig,
    wav_path: str,
    response_format: str,
) -> web.StreamResponse:
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
    out, err = await proc.communicate()
    if proc.returncode != 0:
        raise web.HTTPInternalServerError(
            text=f"voxtral failed (exit={proc.returncode}):\n{(err or b'').decode('utf-8', errors='replace')}\n",
            content_type="text/plain",
        )

    text = (out or b"").decode("utf-8", errors="replace").strip()

    if response_format == "text":
        return web.Response(text=text + "\n", content_type="text/plain")
    if response_format in ("json", "", None):
        return web.json_response({"text": text})
    if response_format == "verbose_json":
        # Minimal "verbose" response for compatibility.
        return web.json_response(
            {
                "task": "transcribe",
                "text": text,
                "segments": [],
            }
        )

    raise web.HTTPBadRequest(
        text="Unsupported response_format (use: json|text|verbose_json)\n",
        content_type="text/plain",
    )


async def handle_health(request: web.Request) -> web.Response:
    return web.json_response({"ok": True, "ts_ms": _now_ms()})


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
        td_path = pathlib.Path(td)
        async for part in reader:
            if part.name == "file":
                # Stream upload to disk.
                fn = part.filename or "audio"
                uploaded_path = str(td_path / fn)
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

        wav_path = str(td_path / "audio_16k_mono.wav")
        rc, err = await _run_ffmpeg_to_wav(uploaded_path, wav_path)
        if rc != 0:
            raise web.HTTPBadRequest(
                text=f"ffmpeg decode failed:\n{err.decode('utf-8', errors='replace')}\n",
                content_type="text/plain",
            )

        return await _run_voxtral_file(
            cfg=cfg, wav_path=wav_path, response_format=response_format
        )


async def handle_ws_realtime(request: web.Request) -> web.StreamResponse:
    cfg: ServerConfig = request.app["cfg"]
    _require_auth(cfg, request)

    sem: asyncio.Semaphore = request.app["session_sem"]
    acquired = False
    ws = web.WebSocketResponse(heartbeat=20.0, max_msg_size=16 * 1024 * 1024)
    proc: Optional[asyncio.subprocess.Process] = None
    stdout_task: Optional[asyncio.Task] = None
    stderr_task: Optional[asyncio.Task] = None

    try:
        await asyncio.wait_for(sem.acquire(), timeout=0.05)
        acquired = True
    except asyncio.TimeoutError:
        raise web.HTTPTooManyRequests(
            text="Too many active sessions\n", content_type="text/plain"
        )
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
                await ws.close()

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
                await ws.close()

            # If the client ended the WS without "stop", ensure we close our side too.
            if not ws.closed:
                await ws.close()

        return ws
    finally:
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
        api_key=(args.api_key.strip() or None),
        extra_env=_parse_env_kv(args.env),
    )

    static_dir = pathlib.Path(__file__).resolve().parent / "static"
    if not static_dir.is_dir():
        raise SystemExit(f"static dir not found: {static_dir}")

    app = web.Application(middlewares=[cors_middleware])
    app["cfg"] = cfg
    app["static_dir"] = static_dir
    app["session_sem"] = asyncio.Semaphore(cfg.max_sessions)

    app.router.add_get("/", handle_index)
    app.router.add_get("/health", handle_health)
    app.router.add_post("/v1/audio/transcriptions", handle_transcriptions)
    app.router.add_get("/v1/audio/transcriptions/realtime", handle_ws_realtime)
    app.router.add_static("/static/", path=str(static_dir), show_index=False)

    print(
        f"voxtral server listening on http://{cfg.host}:{cfg.port} (max_sessions={cfg.max_sessions})"
    )
    if cfg.api_key:
        print("auth enabled: set Authorization: Bearer <token>")
    if cfg.extra_env:
        print(f"voxtral subprocess env overrides: {cfg.extra_env}")
    return web.run_app(app, host=cfg.host, port=cfg.port, print=None)  # type: ignore[arg-type]


if __name__ == "__main__":
    raise SystemExit(main())
