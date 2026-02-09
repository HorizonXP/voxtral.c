#!/usr/bin/env python3
"""
Tiny smoke client for the batch transcription endpoint.

Usage:
  python3 web/smoke_client.py --file samples/test_speech.wav
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from typing import Optional

import aiohttp


async def _run(url: str, file_path: str, api_key: Optional[str]) -> int:
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    form = aiohttp.FormData()
    form.add_field("model", "voxtral")
    form.add_field("response_format", "json")
    form.add_field(
        "file",
        open(file_path, "rb"),
        filename=os.path.basename(file_path),
        content_type="application/octet-stream",
    )

    async with aiohttp.ClientSession(headers=headers) as session:
        async with session.post(url, data=form) as resp:
            body = await resp.text()
            if resp.status != 200:
                print(body, end="")
                return 1
            try:
                obj = json.loads(body)
                print(obj.get("text", "").strip())
            except Exception:
                print(body, end="")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--url",
        default="http://127.0.0.1:8000/v1/audio/transcriptions",
        help="Batch transcription endpoint URL",
    )
    ap.add_argument("--file", default="samples/test_speech.wav")
    ap.add_argument(
        "--api-key",
        default=os.environ.get("VOXTRAL_API_KEY", ""),
        help="Optional bearer token (also via VOXTRAL_API_KEY)",
    )
    args = ap.parse_args()
    return asyncio.run(_run(args.url, args.file, args.api_key.strip() or None))


if __name__ == "__main__":
    raise SystemExit(main())

