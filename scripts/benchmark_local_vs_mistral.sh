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

bench_local_one() {
  local file="$1"
  local hdr body
  hdr="$(mktemp)"
  body="$(mktemp)"
  local auth=()
  if [[ -n "${LOCAL_API_KEY}" ]]; then
    auth=(-H "Authorization: Bearer ${LOCAL_API_KEY}")
  fi

  # shellcheck disable=SC2068
  curl -sS "${auth[@]}" \
    -D "${hdr}" \
    -o "${body}" \
    -F "file=@${file}" \
    -F "model=voxtral" \
    -F "response_format=json" \
    "${LOCAL_URL}/v1/audio/transcriptions" >/dev/null

  python3 - <<PY
import pathlib, re, sys
hdr = pathlib.Path("${hdr}").read_text(errors="replace").splitlines()
def h(name):
    pat = re.compile(rf"^{re.escape(name)}:\\s*(.*)$", re.I)
    for line in hdr:
        m = pat.match(line)
        if m:
            return m.group(1).strip()
    return ""
print("\\t".join([
    "local",
    "${file}",
    h("X-Voxtral-Audio-Sec"),
    h("X-Voxtral-Upload-Ms"),
    h("X-Voxtral-Decode-Ms"),
    h("X-Voxtral-Infer-Ms"),
    h("X-Voxtral-Total-Ms"),
    h("X-Voxtral-xRT"),
]))
PY

  rm -f "${hdr}" "${body}"
}

bench_mistral_one() {
  local file="$1"
  local hdr body
  hdr="$(mktemp)"
  body="$(mktemp)"

  if [[ -z "${MISTRAL_API_KEY:-}" ]]; then
    echo -e "mistral\t${file}\tSKIP(no MISTRAL_API_KEY)"
    rm -f "${hdr}" "${body}"
    return 0
  fi

  local t_total
  t_total="$(
    curl -sS \
      -D "${hdr}" \
      -o "${body}" \
      -w '%{time_total}' \
      -H "Authorization: Bearer ${MISTRAL_API_KEY}" \
      -F "file=@${file}" \
      -F "model=${MISTRAL_MODEL}" \
      -F "response_format=json" \
      "${MISTRAL_URL}"
  )"

  local dur
  dur="$(duration_sec "${file}")"

  python3 - <<PY
dur = float("${dur}")
t = float("${t_total}")
xrt = (dur / t) if t > 0 else 0.0
print("\\t".join([
    "mistral",
    "${file}",
    f"{dur:.6f}",
    f"{t*1000.0:.3f}",
    f"{xrt:.3f}",
]))
PY

  rm -f "${hdr}" "${body}"
}

main() {
  cd "${ROOT}"
  local files=(
    "samples/test_speech.wav"
    "samples/I_have_a_dream.ogg"
  )

  echo -e "backend\tfile\taudio_s\tupload_ms\tdecode_ms\tinfer_ms\ttotal_ms\txRT"
  for f in "${files[@]}"; do
    bench_local_one "${f}"
  done

  echo ""
  echo -e "backend\tfile\taudio_s\twall_ms\txRT"
  for f in "${files[@]}"; do
    bench_mistral_one "${f}"
  done
}

main "$@"

