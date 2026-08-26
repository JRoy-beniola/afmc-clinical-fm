#!/usr/bin/env bash
set -euo pipefail

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

CONTROL="${CONTROL:-$HOME/phase06-control}"
SESSION="${PHASE06_TMUX_SESSION:-phase06-$STAGE}"

command -v tmux >/dev/null 2>&1 || {
  echo "tmux is required. Install it with: sudo apt install tmux"
  exit 1
}

[[ -x "$CONTROL/run_stage.sh" ]] || {
  echo "Missing executable: $CONTROL/run_stage.sh"
  exit 1
}
[[ -f "$CONTROL/monitor_stage.py" ]] || {
  echo "Missing: $CONTROL/monitor_stage.py"
  exit 1
}

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists."
  echo "Attach with: tmux attach -t $SESSION"
  exit 1
fi

tmux new-session -d -s "$SESSION" -n runner \
  "bash '$CONTROL/run_stage.sh' '$STAGE' '$MODE'; rc=\$?; echo; echo 'Runner exited with code' \$rc; exec bash"

tmux new-window -t "$SESSION" -n monitor \
  "python3 '$CONTROL/monitor_stage.py' '$STAGE'; exec bash"

tmux select-window -t "$SESSION:monitor"

echo "Started Phase 0.6 tmux session: $SESSION"
echo
echo "Windows:"
echo "  runner  - guarded CUDA stage execution + full log stream"
echo "  monitor - read-only live progress dashboard"
echo
echo "tmux controls:"
echo "  Ctrl-b 0   runner"
echo "  Ctrl-b 1   monitor"
echo "  Ctrl-b d   detach without stopping the run"
echo
echo "Attaching to monitor..."
sleep 1
exec tmux attach -t "$SESSION"
