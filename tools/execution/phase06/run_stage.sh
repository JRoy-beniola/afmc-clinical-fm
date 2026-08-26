#!/usr/bin/env bash
set -Eeuo pipefail

STAGE="${1:-}"
MODE="${2:-fresh}"

if [[ "$STAGE" != "d1" && "$STAGE" != "d2a" ]]; then
  echo "Usage: $0 {d1|d2a} [fresh|resume]" >&2
  exit 2
fi
if [[ "$MODE" != "fresh" && "$MODE" != "resume" ]]; then
  echo "Usage: $0 {d1|d2a} [fresh|resume]" >&2
  exit 2
fi

REPO="${REPO:-$HOME/afmc-clinical-fm}"
CONTROL="${CONTROL:-$HOME/phase06-control}"
VALIDATION_RECORD="$REPO/docs/superpowers/validation/2026-08-26-phase0-6-core-validation.md"
PHASE06_CONFIG="$REPO/configs/experiments/phase06.yaml"
PHASE06_SPEC="$REPO/docs/superpowers/specs/2026-08-26-phase0-6-diagnostics-design.md"
PHASE05_CONFIG="$REPO/configs/experiments/phase05.yaml"
PHASE05_PROTOCOL="$REPO/docs/results/phase05/raw/official_output/protocol_lock.json"
PY="$REPO/.venv/bin/python"

LOG_DIR="$CONTROL/logs"
LOG="$LOG_DIR/phase06_${STAGE}.log"
STATUS="$CONTROL/${STAGE}.status"
PIDFILE="$CONTROL/${STAGE}.pid"
STARTFILE="$CONTROL/${STAGE}.started"
EXITFILE="$CONTROL/${STAGE}.exit"
MANIFEST="$CONTROL/${STAGE}_launch_manifest.txt"

mkdir -p "$CONTROL" "$LOG_DIR"
if [[ "$MODE" == "fresh" ]]; then
  : > "$LOG"
  rm -f "$EXITFILE"
fi

exec > >(tee -a "$LOG") 2>&1

write_status() {
  printf '%s\n' "$1" > "$STATUS"
}

cleanup() {
  rc=$?
  printf '%s\n' "$rc" > "$EXITFILE"
  if [[ $rc -ne 0 ]]; then
    write_status "FAILED"
    echo
    echo "PHASE 0.6 $STAGE launcher failed with exit code $rc."
    echo "Preserve the output and log. Diagnose the cause before using resume."
  fi
}
trap cleanup EXIT

echo "$$" > "$PIDFILE"
date +%s > "$STARTFILE"

cd "$REPO"
export TMPDIR=/tmp
export PYTHONUNBUFFERED=1

write_status "PREFLIGHT"

echo "============================================================"
echo " AFMC PHASE 0.6 ${STAGE^^} DIAGNOSTIC EXECUTION"
echo "============================================================"
echo "Mode:       $MODE"
echo "Repository: $REPO"
echo

[[ -x "$PY" ]] || { echo "ERROR: project interpreter missing: $PY"; exit 1; }
[[ -f "$VALIDATION_RECORD" ]] || {
  echo "ERROR: Phase 0.6 validation record is missing: $VALIDATION_RECORD"
  exit 1
}
[[ -f "$PHASE06_CONFIG" ]] || { echo "ERROR: missing $PHASE06_CONFIG"; exit 1; }
[[ -f "$PHASE06_SPEC" ]] || { echo "ERROR: missing $PHASE06_SPEC"; exit 1; }
[[ -f "$PHASE05_CONFIG" ]] || { echo "ERROR: missing $PHASE05_CONFIG"; exit 1; }
[[ -f "$PHASE05_PROTOCOL" ]] || { echo "ERROR: missing $PHASE05_PROTOCOL"; exit 1; }

grep -Eqi 'implementation status:.*READY FOR D1 EXECUTION' "$VALIDATION_RECORD" || {
  echo "ERROR: validation record does not authorize D1 execution."
  echo "Expected an implementation-status line containing: READY FOR D1 EXECUTION"
  exit 1
}

VALIDATED_SHA="$(
  grep -Ei 'Validated implementation SHA|branch/head SHA' "$VALIDATION_RECORD" \
    | grep -Eo '[0-9a-fA-F]{40}' \
    | head -n 1 \
    | tr 'A-F' 'a-f' \
    || true
)"
[[ "$VALIDATED_SHA" =~ ^[0-9a-f]{40}$ ]] || {
  echo "ERROR: validation record does not expose a 40-character validated SHA."
  echo "Expected a line containing: Validated implementation SHA or branch/head SHA"
  exit 1
}

HEAD="$(git rev-parse HEAD | tr 'A-F' 'a-f')"
OUTPUT="${PHASE06_OUTPUT:-$REPO/outputs/phase06_${HEAD}}"

echo "Current HEAD:   $HEAD"
echo "Validated SHA:  $VALIDATED_SHA"
echo "Output:         $OUTPUT"
echo

# Task 12 validates execution-critical code at VALIDATED_SHA. A later
# validation-document or monitoring-only commit may move HEAD, but scientific
# source/config must remain byte-identical across that boundary.
if ! git diff --quiet "$VALIDATED_SHA" "$HEAD" -- src configs pyproject.toml; then
  echo "ERROR: execution-critical source/config differs from the Task-12 validated SHA."
  git diff --stat "$VALIDATED_SHA" "$HEAD" -- src configs pyproject.toml
  exit 1
fi

git diff --quiet || {
  echo "ERROR: tracked working-tree modifications exist."
  git status --short
  exit 1
}
git diff --cached --quiet || {
  echo "ERROR: staged modifications exist."
  git status --short
  exit 1
}

UNTRACKED="$(git ls-files --others --exclude-standard || true)"
if [[ -n "$UNTRACKED" ]]; then
  echo "Note: untracked files exist; they are outside the execution identity:"
  printf '%s\n' "$UNTRACKED"
fi

command -v nvidia-smi >/dev/null 2>&1 || {
  echo "ERROR: nvidia-smi is unavailable. Phase 0.6 official diagnostics require CUDA."
  exit 1
}

"$PY" - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    print("ERROR: torch.cuda.is_available() is False", file=sys.stderr)
    raise SystemExit(1)
print("CUDA preflight: PASS")
print("torch=" + torch.__version__)
print("torch_cuda=" + str(torch.version.cuda))
print("gpu=" + torch.cuda.get_device_name(0))
PY
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

if [[ "$STAGE" == "d1" ]]; then
  if [[ "$MODE" == "fresh" ]]; then
    if [[ -e "$OUTPUT" ]] && [[ -n "$(find "$OUTPUT" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
      echo "ERROR: D1 fresh output already contains data: $OUTPUT"
      echo "Use resume only for the intended interrupted run."
      exit 1
    fi
  else
    [[ -f "$OUTPUT/protocol_lock.json" ]] || {
      echo "ERROR: D1 resume requires $OUTPUT/protocol_lock.json"
      exit 1
    }
  fi
else
  [[ -f "$OUTPUT/protocol_lock.json" ]] || {
    echo "ERROR: D2-A requires an existing D1-bound Phase 0.6 output root."
    exit 1
  }
  [[ -f "$OUTPUT/stages/d1/COMPLETE" ]] || {
    echo "ERROR: D2-A is locked until D1 has an exact COMPLETE marker."
    exit 1
  }
  [[ -f "$OUTPUT/analysis/phase06_d1_classification.json" ]] || {
    echo "ERROR: D2-A is locked until D1 analysis exists:"
    echo "  $OUTPUT/analysis/phase06_d1_classification.json"
    exit 1
  }
  if [[ "$MODE" == "fresh" ]] && [[ -d "$OUTPUT/stages/d2a" ]] \
      && [[ -n "$(find "$OUTPUT/stages/d2a" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "ERROR: D2-A fresh requested but D2-A artifacts already exist."
    echo "Use resume only after diagnosing a genuine interruption."
    exit 1
  fi
fi

# Any pre-existing protocol lock must be bound to this exact execution HEAD.
if [[ -f "$OUTPUT/protocol_lock.json" ]]; then
  "$PY" - "$OUTPUT/protocol_lock.json" "$HEAD" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
head = sys.argv[2]
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
    raise SystemExit(f"ERROR: invalid existing protocol lock: {error}")
if payload.get("execution_commit") != head:
    raise SystemExit(
        "ERROR: existing Phase 0.6 protocol lock is bound to a different execution commit: "
        + str(payload.get("execution_commit"))
    )
PY
fi

{
  echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "stage=$STAGE"
  echo "mode=$MODE"
  echo "repo=$REPO"
  echo "execution_head=$HEAD"
  echo "validated_sha=$VALIDATED_SHA"
  echo "validation_record=$VALIDATION_RECORD"
  echo "validation_record_sha256=$(sha256sum "$VALIDATION_RECORD" | awk '{print $1}')"
  echo "phase06_config=$PHASE06_CONFIG"
  echo "phase06_config_sha256=$(sha256sum "$PHASE06_CONFIG" | awk '{print $1}')"
  echo "phase06_spec=$PHASE06_SPEC"
  echo "phase06_spec_sha256=$(sha256sum "$PHASE06_SPEC" | awk '{print $1}')"
  echo "phase05_config=$PHASE05_CONFIG"
  echo "phase05_config_sha256=$(sha256sum "$PHASE05_CONFIG" | awk '{print $1}')"
  echo "phase05_protocol=$PHASE05_PROTOCOL"
  echo "phase05_protocol_sha256=$(sha256sum "$PHASE05_PROTOCOL" | awk '{print $1}')"
  echo "output=$OUTPUT"
  "$PY" --version 2>&1
  "$PY" - <<'PY'
import torch
print("torch=" + torch.__version__)
print("torch_cuda=" + str(torch.version.cuda))
print("cuda_available=" + str(torch.cuda.is_available()))
if torch.cuda.is_available():
    print("gpu=" + torch.cuda.get_device_name(0))
PY
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader \
    | sed 's/^/nvidia_smi=/'
} > "$MANIFEST"

echo
echo "Preflight passed."
write_status "RUNNING"

ARGS=(
  --config "$PHASE06_CONFIG"
  --output "$OUTPUT"
  --device cuda
)
if [[ "$MODE" == "resume" ]]; then
  ARGS+=(--resume)
fi

echo
echo ">>> PHASE 0.6 ${STAGE^^}: starting one-worker CUDA diagnostic stage"
"$PY" -m afmc_fm.phase06.cli "$STAGE" "${ARGS[@]}"

[[ -f "$OUTPUT/stages/$STAGE/COMPLETE" ]] || {
  echo "ERROR: stage command returned successfully but COMPLETE marker is missing."
  exit 1
}

if [[ "$STAGE" == "d1" ]]; then
  [[ -f "$OUTPUT/analysis/phase06_d1_classification.json" ]] || {
    echo "ERROR: D1 command returned successfully but D1 classification is missing."
    exit 1
  }
else
  [[ -f "$OUTPUT/analysis/phase06_d2_n_shift_summary.json" ]] || {
    echo "ERROR: D2-A command returned successfully but D2-A analysis is missing."
    exit 1
  }
fi

write_status "COMPLETE"
echo
echo "============================================================"
echo " PHASE 0.6 ${STAGE^^} COMPLETE"
echo "============================================================"
echo "Output: $OUTPUT"
if [[ "$STAGE" == "d1" ]]; then
  echo "HARD STOP: inspect and audit D1 before starting D2-A."
else
  echo "HARD STOP: adjudication has NOT been run automatically."
  echo "Return D1/D2-A artifacts for audit before any D2-B or D4 addendum."
fi
