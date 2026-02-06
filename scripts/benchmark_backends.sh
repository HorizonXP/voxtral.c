#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${1:-voxtral-model}"
SAMPLE_FILE="${2:-samples/test_speech.wav}"

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "model dir '$MODEL_DIR' missing"
  exit 1
fi

run_case() {
  local backend="$1"
  echo "== backend: $backend =="
  make "$backend"
  /usr/bin/time -f "elapsed=%E cpu=%P maxrss_kb=%M" ./voxtral -d "$MODEL_DIR" -i "$SAMPLE_FILE" --silent >/tmp/voxtral_${backend}.txt
  echo "output_bytes=$(wc -c </tmp/voxtral_${backend}.txt)"
  echo
}

run_case blas
run_case cuda

echo "Done. Compare /tmp/voxtral_blas.txt and /tmp/voxtral_cuda.txt for transcript diffs."
