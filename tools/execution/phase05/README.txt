AFMC Phase-0.5 Official Stage-I Execution Kit
==============================================

Purpose
-------
Run the validated Phase-0.5 official Stage-I development sequence and Stage-II
freeze directly on the local WSL machine, without keeping Codex in the loop.

Pinned current PR head:
50a94c06bc1c419ca55738f15f074cc06ccc3f36

Validated execution-code commit:
3fb62ff71e3d9f9750b6dbc1ebf5525db3f71e71

The runner performs:
  preflight
  -> phase05 calibrate
  -> phase05 develop
       flow
       jump
       uncertainty
       representation timing audit
  -> phase05 freeze
  -> HARD STOP

It never runs phase05 confirm or robustness.

Install
-------
mkdir -p ~/phase05-control
Copy these files into that directory:
  run_stage1.sh
  monitor_stage1.py
  start_stage1.sh

Then:
chmod +x ~/phase05-control/run_stage1.sh
chmod +x ~/phase05-control/monitor_stage1.py
chmod +x ~/phase05-control/start_stage1.sh

Preflight
---------
cd ~/afmc-clinical-fm
git rev-parse HEAD

It must print:
50a94c06bc1c419ca55738f15f074cc06ccc3f36

The runner expects the completed Phase-0 official metrics at:
~/afmc-clinical-fm/outputs/phase0_full_cuda_d6f105eee73fcb8e9cc5987d292b1bb98a687382/metrics.csv

If your actual Phase-0 metrics file is elsewhere, set it explicitly:
export PHASE0_METRICS=/absolute/path/to/metrics.csv

Fresh official launch
---------------------
~/phase05-control/start_stage1.sh fresh

Detach safely:
Ctrl-b d

Reattach:
tmux attach -t phase05-stage1

Switch windows:
Ctrl-b 0   runner
Ctrl-b 1   monitor

Resume after a genuine interruption
------------------------------------
Do not delete or edit the official output.

First understand why the process stopped. Then:
tmux kill-session -t phase05-stage1 2>/dev/null || true
~/phase05-control/start_stage1.sh resume

Official output
---------------
~/afmc-clinical-fm/outputs/phase05_official_50a94c06bc1c419ca55738f15f074cc06ccc3f36

Expected Stage-I cell counts, assuming all prerequisite gates pass:
  flow           60
  jump           60
  uncertainty   180
  timing_audit  120
  total         420

After successful completion
---------------------------
The monitor must show:
STAGE I COMPLETE / STAGE II FROZEN — HARD STOP BEFORE CONFIRMATION

Do NOT run confirmation yet.

Bring these artifacts back for independent audit:
  protocol_lock.json
  development/flow_gate.csv
  development/jump_gate.csv
  development/uncertainty_gate.csv
  development/representation_timing_audit.csv
  development/mechanism_metrics.csv
  frozen_candidate.json
  any capacity-audit artifact written by freeze

Also retain:
  ~/phase05-control/logs/phase05_stage1_official.log
  ~/phase05-control/stage1_launch_manifest.txt
