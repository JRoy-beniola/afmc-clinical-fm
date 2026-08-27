#!/usr/bin/env bash
set -Eeuo pipefail

STAGE="${1:-}"
MODE="${2:-fresh}"

if [[ "$STAGE" != "d1" && "$STAGE" != "d2a" && "$STAGE" != "d2b" && "$STAGE" != "d4b" ]]; then
  echo "Usage: $0 {d1|d2a|d2b|d4b} [fresh|resume]" >&2
  exit 2
fi
if [[ "$MODE" != "fresh" && "$MODE" != "resume" ]]; then
  echo "Usage: $0 {d1|d2a|d2b|d4b} [fresh|resume]" >&2
  exit 2
fi

REPO="${REPO:-$HOME/afmc-clinical-fm}"
CONTROL="${CONTROL:-$HOME/phase06-control}"
CORE_VALIDATION_RECORD="$REPO/docs/superpowers/validation/2026-08-26-phase0-6-core-validation.md"
D2B_VALIDATION_RECORD="$REPO/docs/superpowers/validation/2026-08-26-phase0-6-d2b-validation.md"
D4B_VALIDATION_RECORD="$REPO/docs/superpowers/validation/2026-08-27-phase0-6-d4b-validation.md"
PHASE06_CONFIG="$REPO/configs/experiments/phase06.yaml"
PHASE06_SPEC="$REPO/docs/superpowers/specs/2026-08-26-phase0-6-diagnostics-design.md"
D2B_ADDENDUM="$REPO/docs/superpowers/specs/2026-08-26-phase0-6-d2b-execution-addendum.md"
D4B_ADDENDUM="$REPO/docs/superpowers/specs/2026-08-27-phase0-6-d4b-execution-addendum.md"
PHASE05_CONFIG="$REPO/configs/experiments/phase05.yaml"
PHASE05_PROTOCOL="$REPO/docs/results/phase05/raw/official_output/protocol_lock.json"
PY="$REPO/.venv/bin/python"

if [[ "$STAGE" == "d2b" ]]; then
  VALIDATION_RECORD="$D2B_VALIDATION_RECORD"
elif [[ "$STAGE" == "d4b" ]]; then
  VALIDATION_RECORD="$D4B_VALIDATION_RECORD"
else
  VALIDATION_RECORD="$CORE_VALIDATION_RECORD"
fi

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

if [[ "$STAGE" == "d2b" ]]; then
  [[ -f "$D2B_ADDENDUM" ]] || { echo "ERROR: missing $D2B_ADDENDUM"; exit 1; }
  grep -Eqi 'implementation status:.*READY FOR D2-B EXECUTION' "$VALIDATION_RECORD" || {
    echo "ERROR: D2-B validation record does not authorize D2-B execution."
    echo "Expected an implementation-status line containing: READY FOR D2-B EXECUTION"
    exit 1
  }
elif [[ "$STAGE" == "d4b" ]]; then
  [[ -f "$D4B_ADDENDUM" ]] || { echo "ERROR: missing $D4B_ADDENDUM"; exit 1; }
  grep -Eqi 'implementation status:.*READY FOR D4-B EXECUTION' "$VALIDATION_RECORD" || {
    echo "ERROR: D4-B validation record does not authorize D4-B execution."
    echo "Expected an implementation-status line containing: READY FOR D4-B EXECUTION"
    exit 1
  }
else
  grep -Eqi 'implementation status:.*READY FOR D1 EXECUTION' "$VALIDATION_RECORD" || {
    echo "ERROR: validation record does not authorize D1 execution."
    echo "Expected an implementation-status line containing: READY FOR D1 EXECUTION"
    exit 1
  }
fi

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
if [[ "$STAGE" == "d2b" ]]; then
  OUTPUT="${PHASE06_OUTPUT:-$REPO/outputs/phase06_d2b_${HEAD}}"
elif [[ "$STAGE" == "d4b" ]]; then
  OUTPUT="${PHASE06_OUTPUT:-$REPO/outputs/phase06_d4b_${HEAD}}"
else
  OUTPUT="${PHASE06_OUTPUT:-$REPO/outputs/phase06_${HEAD}}"
fi

echo "Current HEAD:   $HEAD"
echo "Validated SHA:  $VALIDATED_SHA"
echo "Output:         $OUTPUT"
echo

if [[ "$STAGE" != "d2b" && "$STAGE" != "d4b" ]]; then
  if ! git diff --quiet "$VALIDATED_SHA" "$HEAD" -- src configs pyproject.toml; then
    echo "ERROR: execution-critical source/config differs from the Task-12 validated SHA."
    git diff --stat "$VALIDATED_SHA" "$HEAD" -- src configs pyproject.toml
    exit 1
  fi
elif [[ "$STAGE" == "d2b" ]]; then
  D2B_CRITICAL_PATHS=(
    src
    configs
    pyproject.toml
    tools/execution/phase06/run_stage.sh
    tools/execution/phase06/start_stage.sh
    tools/monitoring/phase06/monitor_stage.py
    docs/superpowers/specs/2026-08-26-phase0-6-d2b-execution-addendum.md
  )
  if ! git diff --quiet "$VALIDATED_SHA" "$HEAD" -- "${D2B_CRITICAL_PATHS[@]}"; then
    echo "ERROR: D2-B execution-critical code/tooling differs from the validated SHA."
    git diff --stat "$VALIDATED_SHA" "$HEAD" -- "${D2B_CRITICAL_PATHS[@]}"
    exit 1
  fi
else
  D4B_CRITICAL_PATHS=(
    src
    configs
    pyproject.toml
    tools/execution/phase06/run_stage.sh
    tools/execution/phase06/start_stage.sh
    tools/monitoring/phase06/monitor_stage.py
    docs/superpowers/specs/2026-08-27-phase0-6-d4b-execution-addendum.md
  )
  if ! git diff --quiet "$VALIDATED_SHA" "$HEAD" -- "${D4B_CRITICAL_PATHS[@]}"; then
    echo "ERROR: D4-B execution-critical code/tooling differs from the validated SHA."
    git diff --stat "$VALIDATED_SHA" "$HEAD" -- "${D4B_CRITICAL_PATHS[@]}"
    exit 1
  fi
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

PARENT_OUTPUT=""
CORE_PARENT_OUTPUT=""
D2B_PARENT_OUTPUT=""
if [[ "$STAGE" == "d2b" ]]; then
  [[ -n "${PHASE06_PARENT_OUTPUT:-}" ]] || {
    echo "ERROR: D2-B requires PHASE06_PARENT_OUTPUT to point at the frozen D1/D2-A/D3 parent output."
    exit 1
  }
  PARENT_OUTPUT="$(realpath -e "$PHASE06_PARENT_OUTPUT")"
  CHILD_CANONICAL="$(realpath -m "$OUTPUT")"
  if [[ "$PARENT_OUTPUT" == "$CHILD_CANONICAL" ]] \
      || [[ "$CHILD_CANONICAL" == "$PARENT_OUTPUT/"* ]] \
      || [[ "$PARENT_OUTPUT" == "$CHILD_CANONICAL/"* ]]; then
    echo "ERROR: D2-B parent and child output roots must be distinct and non-nested."
    exit 1
  fi
  [[ -f "$PARENT_OUTPUT/protocol_lock.json" ]] || {
    echo "ERROR: D2-B parent protocol lock is missing: $PARENT_OUTPUT/protocol_lock.json"
    exit 1
  }
  [[ -f "$PARENT_OUTPUT/stages/d1/COMPLETE" ]] || {
    echo "ERROR: D2-B parent D1 COMPLETE marker is missing."
    exit 1
  }
  [[ -f "$PARENT_OUTPUT/stages/d2a/COMPLETE" ]] || {
    echo "ERROR: D2-B parent D2-A COMPLETE marker is missing."
    exit 1
  }
  [[ -f "$PARENT_OUTPUT/analysis/phase06_d3_adjudication.json" ]] || {
    echo "ERROR: D2-B parent D3 adjudication is missing."
    exit 1
  }
  echo "Parent output:  $PARENT_OUTPUT"
  echo
elif [[ "$STAGE" == "d4b" ]]; then
  [[ -n "${PHASE06_CORE_PARENT_OUTPUT:-}" ]] || {
    echo "ERROR: D4-B requires PHASE06_CORE_PARENT_OUTPUT."
    exit 1
  }
  [[ -n "${PHASE06_D2B_PARENT_OUTPUT:-}" ]] || {
    echo "ERROR: D4-B requires PHASE06_D2B_PARENT_OUTPUT."
    exit 1
  }
  CORE_PARENT_OUTPUT="$(realpath -e "$PHASE06_CORE_PARENT_OUTPUT")"
  D2B_PARENT_OUTPUT="$(realpath -e "$PHASE06_D2B_PARENT_OUTPUT")"
  CHILD_CANONICAL="$(realpath -m "$OUTPUT")"
  if [[ "$CORE_PARENT_OUTPUT" == "$D2B_PARENT_OUTPUT" ]] \
      || [[ "$CORE_PARENT_OUTPUT" == "$CHILD_CANONICAL" ]] \
      || [[ "$D2B_PARENT_OUTPUT" == "$CHILD_CANONICAL" ]] \
      || [[ "$CHILD_CANONICAL" == "$CORE_PARENT_OUTPUT/"* ]] \
      || [[ "$CORE_PARENT_OUTPUT" == "$CHILD_CANONICAL/"* ]] \
      || [[ "$CHILD_CANONICAL" == "$D2B_PARENT_OUTPUT/"* ]] \
      || [[ "$D2B_PARENT_OUTPUT" == "$CHILD_CANONICAL/"* ]] \
      || [[ "$D2B_PARENT_OUTPUT" == "$CORE_PARENT_OUTPUT/"* ]] \
      || [[ "$CORE_PARENT_OUTPUT" == "$D2B_PARENT_OUTPUT/"* ]]; then
    echo "ERROR: D4-B output roots must be distinct and non-nested."
    exit 1
  fi
  [[ -f "$CORE_PARENT_OUTPUT/protocol_lock.json" ]] || {
    echo "ERROR: D4-B core parent protocol lock is missing."
    exit 1
  }
  [[ -f "$CORE_PARENT_OUTPUT/stages/d1/COMPLETE" ]] || {
    echo "ERROR: D4-B core parent D1 COMPLETE marker is missing."
    exit 1
  }
  [[ -f "$CORE_PARENT_OUTPUT/stages/d2a/COMPLETE" ]] || {
    echo "ERROR: D4-B core parent D2-A COMPLETE marker is missing."
    exit 1
  }
  [[ -f "$CORE_PARENT_OUTPUT/analysis/phase06_d3_adjudication.json" ]] || {
    echo "ERROR: D4-B core parent D3 adjudication is missing."
    exit 1
  }
  [[ -f "$D2B_PARENT_OUTPUT/protocol_lock.json" ]] || {
    echo "ERROR: D4-B D2-B parent protocol lock is missing."
    exit 1
  }
  [[ -f "$D2B_PARENT_OUTPUT/stages/d2b/COMPLETE" ]] || {
    echo "ERROR: D4-B D2-B parent COMPLETE marker is missing."
    exit 1
  }
  [[ -f "$D2B_PARENT_OUTPUT/analysis/phase06_d2b_adjudication.json" ]] || {
    echo "ERROR: D4-B D2-B parent adjudication is missing."
    exit 1
  }
  echo "Core parent:     $CORE_PARENT_OUTPUT"
  echo "D2-B parent:     $D2B_PARENT_OUTPUT"
  echo
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
elif [[ "$STAGE" == "d2a" ]]; then
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
elif [[ "$STAGE" == "d2b" ]]; then
  if [[ "$MODE" == "fresh" ]]; then
    if [[ -e "$OUTPUT" ]] && [[ -n "$(find "$OUTPUT" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
      echo "ERROR: D2-B fresh child output already contains data: $OUTPUT"
      echo "Use resume only for the intended interrupted child run."
      exit 1
    fi
  else
    [[ -f "$OUTPUT/protocol_lock.json" ]] || {
      echo "ERROR: D2-B resume requires the existing child protocol lock: $OUTPUT/protocol_lock.json"
      exit 1
    }
  fi
else
  if [[ "$MODE" == "fresh" ]]; then
    if [[ -e "$OUTPUT" ]] && [[ -n "$(find "$OUTPUT" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
      echo "ERROR: D4-B fresh child output already contains data: $OUTPUT"
      echo "Use resume only for the intended interrupted child run."
      exit 1
    fi
  else
    [[ -f "$OUTPUT/protocol_lock.json" ]] || {
      echo "ERROR: D4-B resume requires the existing child protocol lock: $OUTPUT/protocol_lock.json"
      exit 1
    }
  fi
fi

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
  if [[ "$STAGE" == "d2b" ]]; then
    echo "d2b_addendum=$D2B_ADDENDUM"
    echo "d2b_addendum_sha256=$(sha256sum "$D2B_ADDENDUM" | awk '{print $1}')"
    echo "parent_output=$PARENT_OUTPUT"
    echo "parent_protocol_sha256=$(sha256sum "$PARENT_OUTPUT/protocol_lock.json" | awk '{print $1}')"
    echo "parent_d3_sha256=$(sha256sum "$PARENT_OUTPUT/analysis/phase06_d3_adjudication.json" | awk '{print $1}')"
  elif [[ "$STAGE" == "d4b" ]]; then
    echo "d4b_addendum=$D4B_ADDENDUM"
    echo "d4b_addendum_sha256=$(sha256sum "$D4B_ADDENDUM" | awk '{print $1}')"
    echo "core_parent_output=$CORE_PARENT_OUTPUT"
    echo "core_parent_protocol_sha256=$(sha256sum "$CORE_PARENT_OUTPUT/protocol_lock.json" | awk '{print $1}')"
    echo "core_parent_d3_sha256=$(sha256sum "$CORE_PARENT_OUTPUT/analysis/phase06_d3_adjudication.json" | awk '{print $1}')"
    echo "d2b_parent_output=$D2B_PARENT_OUTPUT"
    echo "d2b_parent_protocol_sha256=$(sha256sum "$D2B_PARENT_OUTPUT/protocol_lock.json" | awk '{print $1}')"
    echo "d2b_parent_adjudication_sha256=$(sha256sum "$D2B_PARENT_OUTPUT/analysis/phase06_d2b_adjudication.json" | awk '{print $1}')"
  fi
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

if [[ "$STAGE" == "d2b" ]]; then
  D2B_ARGS=(
    --config "$PHASE06_CONFIG"
    --parent-output "$PARENT_OUTPUT"
    --output "$OUTPUT"
    --device cuda
  )
  if [[ "$MODE" == "resume" ]]; then
    D2B_ARGS+=(--resume)
  fi
  echo
  echo ">>> PHASE 0.6 D2B: starting one-worker CUDA complementary diagnostic stage"
  "$PY" -m afmc_fm.phase06.cli d2b "${D2B_ARGS[@]}"
elif [[ "$STAGE" == "d4b" ]]; then
  D4B_ARGS=(
    --config "$PHASE06_CONFIG"
    --core-parent-output "$CORE_PARENT_OUTPUT"
    --d2b-parent-output "$D2B_PARENT_OUTPUT"
    --output "$OUTPUT"
    --device cuda
  )
  if [[ "$MODE" == "resume" ]]; then
    D4B_ARGS+=(--resume)
  fi
  echo
  echo ">>> PHASE 0.6 D4B: starting one-worker CUDA optimization-stability diagnostic stage"
  "$PY" -m afmc_fm.phase06.cli d4b "${D4B_ARGS[@]}"
else
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
fi

if [[ "$STAGE" == "d2b" ]]; then
  [[ -f "$OUTPUT/stages/d2b/COMPLETE" ]] || {
    echo "ERROR: D2-B command returned successfully but D2-B COMPLETE marker is missing."
    exit 1
  }
elif [[ "$STAGE" == "d4b" ]]; then
  [[ -f "$OUTPUT/stages/d4b/COMPLETE" ]] || {
    echo "ERROR: D4-B command returned successfully but D4-B COMPLETE marker is missing."
    exit 1
  }
else
  [[ -f "$OUTPUT/stages/$STAGE/COMPLETE" ]] || {
    echo "ERROR: stage command returned successfully but COMPLETE marker is missing."
    exit 1
  }
fi

if [[ "$STAGE" == "d1" ]]; then
  [[ -f "$OUTPUT/analysis/phase06_d1_classification.json" ]] || {
    echo "ERROR: D1 command returned successfully but D1 classification is missing."
    exit 1
  }
elif [[ "$STAGE" == "d2a" ]]; then
  [[ -f "$OUTPUT/analysis/phase06_d2_n_shift_summary.json" ]] || {
    echo "ERROR: D2-A command returned successfully but D2-A analysis is missing."
    exit 1
  }
elif [[ "$STAGE" == "d2b" ]]; then
  [[ -f "$OUTPUT/analysis/phase06_d2b_n_shift_summary.json" ]] || {
    echo "ERROR: D2-B command returned successfully but D2-B analysis is missing."
    exit 1
  }
else
  [[ -f "$OUTPUT/analysis/phase06_d4b_bootstrap_diagnostics.json" ]] || {
    echo "ERROR: D4-B command returned successfully but D4-B analysis is missing."
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
elif [[ "$STAGE" == "d2a" ]]; then
  echo "HARD STOP: adjudication has NOT been run automatically."
  echo "Return D1/D2-A artifacts for audit before any later diagnostic stage."
elif [[ "$STAGE" == "d2b" ]]; then
  echo "HARD STOP: D2-B cross-array adjudication has NOT been run automatically."
  echo "Return the child artifacts for audit before any later intervention."
else
  echo "HARD STOP: D4-B adjudication has NOT been run automatically."
  echo "Audit the 100-cell child and analysis before the separate adjudicate-d4b action."
  echo "This launcher does not authorize D4_CAPACITY_TIME or any later intervention."
fi
