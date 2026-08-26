#!/usr/bin/env bash
set -Eeuo pipefail

MODE="${1:-fresh}"
if [[ "$MODE" != "fresh" && "$MODE" != "resume" ]]; then
  echo "Usage: $0 [fresh|resume]" >&2
  exit 2
fi

REPO="${REPO:-$HOME/afmc-clinical-fm}"
CONTROL="${CONTROL:-$HOME/phase05-control}"

IMPLEMENTATION_SHA="50a94c06bc1c419ca55738f15f074cc06ccc3f36"
VALIDATED_CODE_SHA="3fb62ff71e3d9f9750b6dbc1ebf5525db3f71e71"
PHASE0_SHA="d6f105eee73fcb8e9cc5987d292b1bb98a687382"

EXP_CONFIG="$REPO/configs/experiments/phase05.yaml"
SIM_CONFIG="$REPO/configs/simulator/full.yaml"
VALIDATION_RECORD="$REPO/docs/superpowers/validation/2026-08-24-phase0-5-core-validation.md"

PHASE0_OUT="$REPO/outputs/phase0_full_cuda_${PHASE0_SHA}"
PHASE0_METRICS="${PHASE0_METRICS:-$PHASE0_OUT/metrics.csv}"

OUTPUT="${PHASE05_OUTPUT:-$REPO/outputs/phase05_official_${IMPLEMENTATION_SHA}}"

PY="$REPO/.venv/bin/python"
LOG_DIR="$CONTROL/logs"
LOG="$LOG_DIR/phase05_stage1_official.log"
STATUS="$CONTROL/stage1.status"
PIDFILE="$CONTROL/stage1.pid"
STARTFILE="$CONTROL/stage1.started"
EXITFILE="$CONTROL/stage1.exit"
MANIFEST="$CONTROL/stage1_launch_manifest.txt"

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
    echo "PHASE-0.5 STAGE-I launcher failed with exit code $rc."
    echo "Do NOT delete the output directory. Inspect the log and resume only after the cause is understood."
  fi
}
trap cleanup EXIT

echo "$$" > "$PIDFILE"
date +%s > "$STARTFILE"

cd "$REPO"
export TMPDIR=/tmp
export PYTHONUNBUFFERED=1

echo "============================================================"
echo " AFMC PHASE-0.5 OFFICIAL STAGE-I + STAGE-II FREEZE"
echo "============================================================"
echo "Mode:              $MODE"
echo "Repository:        $REPO"
echo "Required HEAD:     $IMPLEMENTATION_SHA"
echo "Phase-0 metrics:   $PHASE0_METRICS"
echo "Simulator config:  $SIM_CONFIG"
echo "Phase-0.5 config:  $EXP_CONFIG"
echo "Output:            $OUTPUT"
echo

write_status "PREFLIGHT"

[[ -x "$PY" ]] || { echo "ERROR: project interpreter missing: $PY"; exit 1; }
[[ -f "$EXP_CONFIG" ]] || { echo "ERROR: missing $EXP_CONFIG"; exit 1; }
[[ -f "$SIM_CONFIG" ]] || { echo "ERROR: missing $SIM_CONFIG"; exit 1; }
[[ -f "$VALIDATION_RECORD" ]] || { echo "ERROR: missing validation record"; exit 1; }
grep -q '^READY FOR OFFICIAL STAGE-I: YES$' "$VALIDATION_RECORD" || {
  echo "ERROR: validation record does not authorize official Stage I."
  exit 1
}

HEAD="$(git rev-parse HEAD)"
[[ "$HEAD" == "$IMPLEMENTATION_SHA" ]] || {
  echo "ERROR: wrong repository HEAD."
  echo "Expected: $IMPLEMENTATION_SHA"
  echo "Actual:   $HEAD"
  exit 1
}

# The validation record names VALIDATED_CODE_SHA as the locally validated code.
# The current PR head contains only post-validation tests/docs/monitoring changes.
# Refuse launch if execution-critical source/config changed after that validation.
if ! git diff --quiet "$VALIDATED_CODE_SHA" "$IMPLEMENTATION_SHA" -- src configs pyproject.toml; then
  echo "ERROR: execution-critical source/config differs from the validated code commit."
  git diff --stat "$VALIDATED_CODE_SHA" "$IMPLEMENTATION_SHA" -- src configs pyproject.toml
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
  echo "Note: untracked files exist; they are not part of the execution identity:"
  printf '%s\n' "$UNTRACKED"
fi

[[ -f "$PHASE0_METRICS" ]] || {
  echo "ERROR: official Phase-0 metrics were not found at:"
  echo "  $PHASE0_METRICS"
  echo
  echo "Candidate metrics files under outputs/:"
  find "$REPO/outputs" -maxdepth 3 -type f -name metrics.csv -print 2>/dev/null || true
  exit 1
}

if [[ "$MODE" == "fresh" ]]; then
  if [[ -e "$OUTPUT" ]] && [[ -n "$(find "$OUTPUT" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "ERROR: official output directory already contains data:"
    echo "  $OUTPUT"
    echo "Use '$0 resume' only if this is the intended interrupted official run."
    exit 1
  fi
else
  [[ -f "$OUTPUT/protocol_lock.json" ]] || {
    echo "ERROR: resume requested but protocol_lock.json is missing from $OUTPUT"
    exit 1
  }
fi

echo
echo "Runtime diagnostics:"
"$PY" -m afmc_fm.cli diagnostics --device cuda --workers 1
echo
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

{
  echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "repo=$REPO"
  echo "implementation_sha=$HEAD"
  echo "validated_code_sha=$VALIDATED_CODE_SHA"
  echo "phase0_execution_sha=$PHASE0_SHA"
  echo "phase0_metrics=$PHASE0_METRICS"
  echo "phase0_metrics_sha256=$(sha256sum "$PHASE0_METRICS" | awk '{print $1}')"
  echo "phase05_config=$EXP_CONFIG"
  echo "phase05_config_sha256=$(sha256sum "$EXP_CONFIG" | awk '{print $1}')"
  echo "simulator_config=$SIM_CONFIG"
  echo "simulator_config_sha256=$(sha256sum "$SIM_CONFIG" | awk '{print $1}')"
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
} > "$MANIFEST"

echo
echo "Preflight passed."

if [[ "$MODE" == "fresh" ]]; then
  write_status "CALIBRATING"
  echo
  echo ">>> CALIBRATE: creating immutable protocol lock"
  "$PY" -m afmc_fm.cli phase05 calibrate \
    --phase0-metrics "$PHASE0_METRICS" \
    --exp-config "$EXP_CONFIG" \
    --output "$OUTPUT"
else
  echo
  echo ">>> RESUME: retaining existing protocol lock"
fi

write_status "DEVELOPING"
echo
echo ">>> STAGE I: flow -> jump -> uncertainty -> timing audit"
DEVELOP_ARGS=(
  --sim-config "$SIM_CONFIG"
  --exp-config "$EXP_CONFIG"
  --output "$OUTPUT"
  --device cuda
  --workers 1
  --fail-fast
)
if [[ "$MODE" == "resume" ]]; then
  DEVELOP_ARGS+=(--resume)
fi

"$PY" -m afmc_fm.cli phase05 develop "${DEVELOP_ARGS[@]}"

write_status "FREEZING"
echo
echo ">>> STAGE II: freezing selected candidate"
"$PY" -m afmc_fm.cli phase05 freeze \
  --exp-config "$EXP_CONFIG" \
  --output "$OUTPUT"

[[ -f "$OUTPUT/frozen_candidate.json" ]] || {
  echo "ERROR: freeze command returned successfully but frozen_candidate.json is missing."
  exit 1
}

write_status "FROZEN_STOP"
echo
echo "============================================================"
echo " STAGE I COMPLETE + STAGE II FROZEN"
echo "============================================================"
echo "Frozen candidate: $OUTPUT/frozen_candidate.json"
echo
echo "HARD STOP: confirmation has NOT been started."
echo "Do not run 'phase05 confirm' until the Stage-I artifacts are independently audited."
