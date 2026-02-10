#!/usr/bin/env bash
set -euo pipefail

# Benchmarks local Voxtral web server vs Mistral hosted API (optional).
#
# Local server should be running:
#   env VOX_CUDA_FAST=1 python3 web/server.py
#
# Optional API benchmark:
#   export MISTRAL_API_KEY=...
#
# Tunables:
#   LOCAL_URL=http://127.0.0.1:8000
#   LOCAL_API_KEY=...                 (if VOXTRAL_API_KEY is enabled on the server)
#   MISTRAL_URL=https://api.mistral.ai/v1/audio/transcriptions
#   MISTRAL_MODEL=voxtral-2

LOCAL_URL="${LOCAL_URL:-http://127.0.0.1:8000}"
LOCAL_API_KEY="${LOCAL_API_KEY:-}"

MISTRAL_URL="${MISTRAL_URL:-https://api.mistral.ai/v1/audio/transcriptions}"
MISTRAL_MODEL="${MISTRAL_MODEL:-voxtral-2}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

need() {
  command -v "$1" >/dev/null 2>&1 || { echo "missing dependency: $1" >&2; exit 1; }
}

need curl
need ffprobe
need python3

duration_sec() {
  ffprobe -v error -show_entries format=duration \
    -of default=noprint_wrappers=1:nokey=1 "$1"
}

header_value() {
  local hdr="$1"
  local name="$2"
  python3 - <<PY
import pathlib, re
hdr = pathlib.Path("${hdr}").read_text(errors="replace").splitlines()
pat = re.compile(rf"^{re.escape('${name}')}:\\s*(.*)$", re.I)
for line in hdr:
    m = pat.match(line)
    if m:
        print(m.group(1).strip())
        raise SystemExit(0)
print("")
PY
}

json_text_len() {
  local body="$1"
  python3 - <<PY
import json, pathlib
p = pathlib.Path("${body}")
try:
    obj = json.loads(p.read_text(errors="replace"))
except Exception:
    print("")
    raise SystemExit(0)
if isinstance(obj, dict) and isinstance(obj.get("text"), str):
    print(len(obj["text"]))
else:
    print("")
PY
}

local_health_check() {
  local health
  health="$(curl -sS --max-time 2 "${LOCAL_URL}/health" || true)"
  if [[ -z "${health}" ]]; then
    echo "[error] local server not reachable at ${LOCAL_URL}" >&2
    echo "Start it from repo root: env VOX_CUDA_FAST=1 python3 web/server.py" >&2
    exit 1
  fi
  python3 - <<PY
import json, sys
s = """${health}"""
try:
    obj = json.loads(s)
except Exception:
    print("[warn] /health returned non-JSON; continuing.", file=sys.stderr)
    raise SystemExit(0)
if isinstance(obj, dict) and "batch_ready" not in obj:
    print("[warn] /health is missing batch_ready; you may be running an older server process. Restart the server to get timing headers.", file=sys.stderr)
PY
}

bench_local_one() {
  local file="$1"
  local dur
  dur="$(duration_sec "${file}")"

  local hdr body
  hdr="$(mktemp)"
  body="$(mktemp)"
  local auth=()
  if [[ -n "${LOCAL_API_KEY}" ]]; then
    auth=(-H "Authorization: Bearer ${LOCAL_API_KEY}")
  fi

  local metrics time_total_s http_code
  # shellcheck disable=SC2068
  metrics="$(
    curl -sS "${auth[@]}" \
      -D "${hdr}" \
      -o "${body}" \
      -w '%{time_total}\t%{http_code}' \
      -F "file=@${file}" \
      -F "model=voxtral" \
      -F "response_format=json" \
      "${LOCAL_URL}/v1/audio/transcriptions"
  )"

  time_total_s="${metrics%%$'\t'*}"
  http_code="${metrics##*$'\t'}"

  if [[ "${http_code}" != "200" ]]; then
    echo "[error] local server returned HTTP ${http_code} for ${file}" >&2
    head -c 4000 "${body}" >&2 || true
    rm -f "${hdr}" "${body}"
    exit 1
  fi

  local audio_s upload_ms decode_ms infer_ms total_ms xrt_infer
  audio_s="$(header_value "${hdr}" "X-Voxtral-Audio-Sec")"
  upload_ms="$(header_value "${hdr}" "X-Voxtral-Upload-Ms")"
  decode_ms="$(header_value "${hdr}" "X-Voxtral-Decode-Ms")"
  infer_ms="$(header_value "${hdr}" "X-Voxtral-Infer-Ms")"
  total_ms="$(header_value "${hdr}" "X-Voxtral-Total-Ms")"
  xrt_infer="$(header_value "${hdr}" "X-Voxtral-xRT")"

  local wall_ms xrt_wall text_chars note
  wall_ms="$(python3 - <<PY
t = float("${time_total_s}")
print(f"{t*1000.0:.3f}")
PY
)"
  xrt_wall="$(python3 - <<PY
dur = float("${dur}")
t = float("${time_total_s}")
print(f"{(dur/t) if t>0 else 0.0:.3f}")
PY
)"
  text_chars="$(json_text_len "${body}")"
  note=""
  if [[ -z "${audio_s}" ]]; then
    audio_s="${dur}"
  fi
  if [[ -z "${upload_ms}${decode_ms}${infer_ms}${total_ms}" ]]; then
    note="missing_x_voxtral_headers(restart_server?)"
  fi

  echo -e "local\t${file}\t${audio_s}\t${http_code}\t${upload_ms}\t${decode_ms}\t${infer_ms}\t${total_ms}\t${xrt_infer}\t${wall_ms}\t${xrt_wall}\t${text_chars}\t${note}"

  rm -f "${hdr}" "${body}"
}

bench_mistral_one() {
  local file="$1"
  local dur
  dur="$(duration_sec "${file}")"

  if [[ -z "${MISTRAL_API_KEY:-}" ]]; then
    echo -e "mistral\t${file}\t${dur}\t\t\t\t\t\t\t\t\t\tSKIP(no MISTRAL_API_KEY)"
    return 0
  fi

  local hdr body
  hdr="$(mktemp)"
  body="$(mktemp)"

  local metrics time_total_s http_code
  metrics="$(
    curl -sS \
      -D "${hdr}" \
      -o "${body}" \
      -w '%{time_total}\t%{http_code}' \
      -H "Authorization: Bearer ${MISTRAL_API_KEY}" \
      -F "file=@${file}" \
      -F "model=${MISTRAL_MODEL}" \
      -F "response_format=json" \
      "${MISTRAL_URL}"
  )"

  time_total_s="${metrics%%$'\t'*}"
  http_code="${metrics##*$'\t'}"

  if [[ "${http_code}" != "200" ]]; then
    echo "[error] mistral returned HTTP ${http_code} for ${file}" >&2
    head -c 4000 "${body}" >&2 || true
    rm -f "${hdr}" "${body}"
    exit 1
  fi

  local wall_ms xrt_wall text_chars note
  wall_ms="$(python3 - <<PY
t = float("${time_total_s}")
print(f"{t*1000.0:.3f}")
PY
)"
  xrt_wall="$(python3 - <<PY
dur = float("${dur}")
t = float("${time_total_s}")
print(f"{(dur/t) if t>0 else 0.0:.3f}")
PY
)"
  text_chars="$(json_text_len "${body}")"
  note=""
  if [[ -z "${text_chars}" ]]; then
    note="no_text_field_or_invalid_json"
  fi

  echo -e "mistral\t${file}\t${dur}\t${http_code}\t\t\t\t\t\t${wall_ms}\t${xrt_wall}\t${text_chars}\t${note}"

  rm -f "${hdr}" "${body}"
}

main() {
  cd "${ROOT}"

  local files=(
    "samples/test_speech.wav"
    "samples/I_have_a_dream.ogg"
  )

  local_health_check

  echo -e "backend\tfile\taudio_s\thttp_code\tupload_ms\tdecode_ms\tinfer_ms\ttotal_ms\txRT_infer\twall_ms\txRT_wall\ttext_chars\tnote"
  for f in "${files[@]}"; do
    bench_local_one "${f}"
  done

  echo ""
  for f in "${files[@]}"; do
    bench_mistral_one "${f}"
  done
}

main "$@"

