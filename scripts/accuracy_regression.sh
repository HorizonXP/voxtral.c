#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${1:-voxtral-model}"
SAMPLE_FILE="${2:-samples/test_speech.wav}"
TOLERANCE_RATIO="${3:-0.005}"   # 0.5% token mismatch tolerance

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "model dir '$MODEL_DIR' missing"
  exit 1
fi

normalize() {
  tr '[:upper:]' '[:lower:]' | tr -s '[:space:]' ' '
}

echo "[1/4] build BLAS"
make blas >/dev/null

echo "[2/4] run BLAS transcript"
./voxtral -d "$MODEL_DIR" -i "$SAMPLE_FILE" --silent | normalize > /tmp/voxtral_ref_blas.txt

echo "[3/4] build CUDA"
make cuda >/dev/null

echo "[4/4] run CUDA transcript"
./voxtral -d "$MODEL_DIR" -i "$SAMPLE_FILE" --silent | normalize > /tmp/voxtral_ref_cuda.txt

python3 - <<PY
from pathlib import Path
import difflib

ref = Path('/tmp/voxtral_ref_blas.txt').read_text().split()
tst = Path('/tmp/voxtral_ref_cuda.txt').read_text().split()
sm = difflib.SequenceMatcher(a=ref, b=tst)
ratio = 1.0 - sm.ratio()
print(f"token_mismatch_ratio={ratio:.6f}")
print(f"ref_tokens={len(ref)} cuda_tokens={len(tst)}")

tol = float('${TOLERANCE_RATIO}')
if ratio > tol:
    raise SystemExit(f"FAIL: mismatch ratio {ratio:.6f} exceeds tolerance {tol:.6f}")
print("PASS")
PY
