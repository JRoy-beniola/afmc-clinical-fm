#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-fresh}"
CONTROL="${CONTROL:-$HOME/phase05-control}"
SESSION="${PHASE05_TMUX_SESSION:-phase05-stage1}"

command -v tmux >/dev/null 2>&1 || {
  echo "tmux is required. Install it with: sudo apt install tmux"
  exit 1
}

[[ -x "$CONTROL/run_stage1.sh" ]] || {
  echo "Missing executable: $CONTROL/run_stage1.sh"
  exit 1
}
[[ -f "$CONTROL/monitor_stage1.py" ]] || {
  echo "Missing: $CONTROL/monitor_stage1.py"
  exit 1
}

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists."
  echo "Attach with: tmux attach -t $SESSION"
  exit 1
fi

tmux new-session -d -s "$SESSION" -n runner \
  "bash '$CONTROL/run_stage1.sh' '$MODE'; rc=\$?; echo; echo 'Runner exited with code' \$rc; exec bash"

tmux new-window -t "$SESSION" -n monitor \
  "python3 '$CONTROL/monitor_stage1.py'; exec bash"

tmux select-window -t "$SESSION:monitor"

echo "Started Phase-0.5 Stage-I tmux session: $SESSION"
echo
echo "Windows:"
echo "  runner  - actual scientific execution + full log stream"
echo "  monitor - live progress dashboard"
echo
echo "tmux controls:"
echo "  Ctrl-b 0   runner"
echo "  Ctrl-b 1   monitor"
echo "  Ctrl-b d   detach without stopping the run"
echo
echo "Attaching to monitor..."
sleep 1
exec tmux attach -t "$SESSION"
