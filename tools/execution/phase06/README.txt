AFMC Phase 0.6 D1 / D2-A CUDA Execution Kit
=============================================

Purpose
-------
Run the validated Phase 0.6 diagnostic stages on the local WSL/CUDA machine
with persistent logs and a read-only tmux progress dashboard.

The kit deliberately exposes only:
  D1   exact 40-cell diagnostic reproduction
  D2-A exact 100-cell orthogonal seed diagnostic

It does not run adjudicate, D2-B, D4, confirmatory seeds, or any continuation
of the terminated Phase 0.5 development sequence.

Scientific boundary
-------------------
Phase 0.5 remains terminated at Stage I-A. Phase 0.6 is diagnostic/exploratory.
The reserved confirmatory seed namespaces remain forbidden.

Before this kit is used, Task 12 must have completed successfully and this
file must exist:
  docs/superpowers/validation/2026-08-26-phase0-6-core-validation.md

The validation record must contain:
  implementation status: READY FOR D1 EXECUTION

Install
-------
From the repository checkout:

mkdir -p ~/phase06-control
cp tools/execution/phase06/run_stage.sh ~/phase06-control/
cp tools/execution/phase06/start_stage.sh ~/phase06-control/
cp tools/monitoring/phase06/monitor_stage.py ~/phase06-control/

chmod +x ~/phase06-control/run_stage.sh
chmod +x ~/phase06-control/start_stage.sh
chmod +x ~/phase06-control/monitor_stage.py

The repository files are intentionally usable even if GitHub did not preserve
an executable bit; the chmod step makes the local control copy executable.

Preflight
---------
The runner refuses execution unless all of the following hold:
  - Task-12 validation record authorizes D1 execution.
  - The validation record exposes its exact branch/head SHA.
  - src/, configs/, and pyproject.toml are unchanged from that validated SHA.
  - The tracked and staged working tree is clean.
  - The project .venv exists.
  - Phase 0.6 config/spec and archived Phase 0.5 evidence are present.
  - nvidia-smi is available and PyTorch reports CUDA available.
  - Existing Phase 0.6 protocol identity, when present, matches current HEAD.

Untracked files are reported but do not alter the Git execution identity.

D1 fresh launch
---------------
After Task 12 and the final validation commit have passed CI:

~/phase06-control/start_stage.sh d1 fresh

This creates tmux session:
  phase06-d1

Windows:
  0 runner   guarded CUDA execution and complete log stream
  1 monitor  read-only progress/GPU dashboard

Detach safely without stopping execution:
  Ctrl-b d

Reattach:
  tmux attach -t phase06-d1

Switch windows:
  Ctrl-b 0   runner
  Ctrl-b 1   monitor

D1 resume after a genuine interruption
---------------------------------------
Do not delete, rename, or manually edit the Phase 0.6 output tree.
First determine why the process stopped. Then, if resume is scientifically and
operationally appropriate:

tmux kill-session -t phase06-d1 2>/dev/null || true
~/phase06-control/start_stage.sh d1 resume

Resume reuses the hash-bound protocol/output root and the Phase 0.6 store will
accept only valid completed cell bundles.

D1 hard stop
------------
A successful D1 launcher stops after D1 COMPLETE and D1 analysis are written.
It does not start D2-A automatically.

Bring the D1 artifacts back for independent audit before starting D2-A.

D2-A fresh launch
-----------------
D2-A is available only after the same output root contains:
  stages/d1/COMPLETE
  analysis/phase06_d1_classification.json

After D1 has been audited and D2-A is authorized:

~/phase06-control/start_stage.sh d2a fresh

The default tmux session is:
  phase06-d2a

Detach:
  Ctrl-b d

Reattach:
  tmux attach -t phase06-d2a

D2-A resume after a genuine interruption
-----------------------------------------
tmux kill-session -t phase06-d2a 2>/dev/null || true
~/phase06-control/start_stage.sh d2a resume

D2-A hard stop
--------------
A successful D2-A launcher stops after D2-A COMPLETE and D2-A analysis are
written. The execution kit does not run adjudicate automatically.

Adjudication must be an explicit audited action after D1/D2-A evidence has
been inspected. D2-B and D4 remain absent until the predeclared D3 decision
selects a later stage and the required addendum is frozen.

Output identity
---------------
By default the runner uses:
  ~/afmc-clinical-fm/outputs/phase06_<CURRENT_HEAD>

D1 and D2-A must use the same execution HEAD and the same protocol-bound output
root. To use an explicitly known output path, set PHASE06_OUTPUT before launch:

export PHASE06_OUTPUT=/absolute/path/to/outputs/phase06_<sha>

Do not point PHASE06_OUTPUT at an output created by a different execution SHA;
the runner and Phase 0.6 store will reject the identity mismatch.

Control files
-------------
The local control plane is kept outside the repository:

~/phase06-control/
  d1.status
  d1.pid
  d1.started
  d1.exit
  d1_launch_manifest.txt
  d2a.status
  d2a.pid
  d2a.started
  d2a.exit
  d2a_launch_manifest.txt
  logs/
    phase06_d1.log
    phase06_d2a.log

The launch manifest records execution/validation identities, config/evidence
hashes, Python/PyTorch/CUDA information, and GPU metadata.

Monitor semantics
-----------------
The monitor reads persisted cell JSON files. It does not infer completion from
stdout. It reports:
  - total cells / remaining / percentage
  - average and recent throughput and ETA
  - progress by N
  - progress by flow mode
  - COMPLETE marker and analysis state
  - process status/PID/elapsed time
  - current Git HEAD and output root
  - GPU utilization, VRAM, temperature, and power
  - invalid/corrupt cell JSON count

Expected D1 counts:
  total          40
  none           20
  time_scaled    20
  N=5            10
  N=10           10
  N=20           10
  N=40           10

Expected D2-A counts:
  total         100
  none           50
  time_scaled    50
  N=5            50
  N=40           50

Ctrl+C in the monitor exits only the monitor process. The runner continues in
the tmux runner window.

Never do these through this kit
-------------------------------
Do not use this kit to run:
  - reserved confirmatory seeds
  - Phase 0.5 jump/uncertainty/timing/confirmation/robustness continuation
  - D2-B before D3 authorizes it
  - any D4 intervention before its separate frozen execution addendum

A negative or ambiguous D1/D2 scientific conclusion is a valid result and is
not an execution failure.
