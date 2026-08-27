AFMC Phase 0.6 D1 / D2-A / D2-B CUDA Execution Kit
====================================================

Purpose
-------
Run the validated Phase 0.6 diagnostic stages on the local WSL/CUDA machine
with persistent logs and a read-only tmux progress dashboard.

The kit exposes only:
  D1   exact 40-cell diagnostic reproduction
  D2-A exact 100-cell orthogonal seed diagnostic
  D2-B exact 100-cell complementary orthogonal seed diagnostic, but only after
       the frozen D3 parent evidence authorizes D2-B and the D2-B validation
       record declares the implementation ready.

Execution and adjudication are deliberately separate. The execution kit does
not run adjudicate or adjudicate-d2b automatically, and it does not authorize D4.

Scientific boundary
-------------------
Phase 0.5 remains terminated at Stage I-A. Phase 0.6 is diagnostic/exploratory.
Reserved confirmatory seed namespaces remain forbidden.

D2-B is not a continuation of Phase 0.5. It is the predeclared complementary
array selected by the completed Phase 0.6 D3 decision. The D1/D2-A/D3 parent
output remains immutable evidence; D2-B always writes to a separate child root.

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

Validation boundaries
---------------------
D1 and D2-A retain the original Phase 0.6 core validation boundary:
  docs/superpowers/validation/2026-08-26-phase0-6-core-validation.md

That record must contain:
  implementation status: READY FOR D1 EXECUTION

D2-B has its own later validation boundary:
  docs/superpowers/validation/2026-08-26-phase0-6-d2b-validation.md

That record must contain:
  implementation status: READY FOR D2-B EXECUTION

The launcher also requires the current execution-critical source, configuration,
D2-B addendum, runner, starter, and monitor to remain byte-equivalent to the
validated D2-B implementation SHA. A later validation-document-only commit may
move HEAD without changing those execution-critical files.

Common preflight
----------------
The runner refuses execution unless all of the following hold:
  - the stage-specific validation record exists and authorizes execution;
  - the validation record exposes its exact validated implementation SHA;
  - execution-critical files have not drifted from that SHA;
  - the tracked and staged working tree is clean;
  - the project .venv exists;
  - Phase 0.6 config/spec and archived Phase 0.5 evidence are present;
  - nvidia-smi is available and PyTorch reports CUDA available;
  - an existing writable output protocol lock, when present, matches HEAD.

Untracked files are reported but do not alter the Git execution identity.

D1 fresh launch
---------------
After the core validation record and CI are green:

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

D1 hard stop
------------
A successful D1 launcher stops after D1 COMPLETE and D1 analysis are written.
It does not start D2-A automatically.

D2-A fresh launch
-----------------
D2-A is available only after the same parent output root contains:
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

D2-B parent binding
-------------------
D2-B requires the completed, audited D1/D2-A/D3 parent output. Set its exact
absolute path before launch:

export PHASE06_PARENT_OUTPUT=/absolute/path/to/outputs/phase06_<parent_sha>

The launcher requires this parent root to contain:
  protocol_lock.json
  stages/d1/COMPLETE
  stages/d2a/COMPLETE
  analysis/phase06_d3_adjudication.json

The Python D2-B CLI then performs the stronger cryptographic validation of the
parent protocol, execution identity, config/spec identity, D3 artifact hash,
and D3 authorization before any D2-B cell is planned or trained.

The parent is read-only evidence. Never point PHASE06_OUTPUT at the same path.

D2-B fresh launch
-----------------
After the D2-B validation record says READY FOR D2-B EXECUTION and final CI is
green:

export PHASE06_PARENT_OUTPUT=/absolute/path/to/outputs/phase06_<parent_sha>
~/phase06-control/start_stage.sh d2b fresh

This creates tmux session:
  phase06-d2b

The default child output is:
  ~/afmc-clinical-fm/outputs/phase06_d2b_<CURRENT_HEAD>

D2-B runs exactly 100 complementary diagnostic cells under the child execution
identity. It does not write into the parent root.

Detach:
  Ctrl-b d

Reattach:
  tmux attach -t phase06-d2b

D2-B resume after a genuine interruption
-----------------------------------------
First diagnose the interruption and preserve the child output. If resume is
appropriate:

tmux kill-session -t phase06-d2b 2>/dev/null || true
export PHASE06_PARENT_OUTPUT=/absolute/path/to/outputs/phase06_<parent_sha>
~/phase06-control/start_stage.sh d2b resume

Resume revalidates the frozen parent and reuses only the hash-bound child store.

D2-B hard stop
--------------
A successful D2-B launcher stops after D2-B COMPLETE and D2-B analysis are
written. It does not run adjudicate-d2b. Cross-array adjudication is a separate,
audited action after the child artifacts are inspected.

The execution result, whether supportive, contradictory, diffuse, or unresolved,
does not authorize D4. Any later intervention remains a separate design and
execution decision.

Output identity
---------------
For D1 and D2-A the default runner output is:
  ~/afmc-clinical-fm/outputs/phase06_<CURRENT_HEAD>

D1 and D2-A must use the same execution HEAD and protocol-bound output root.

For D2-B the default child output is:
  ~/afmc-clinical-fm/outputs/phase06_d2b_<CURRENT_HEAD>

To use an explicitly known child output path, set PHASE06_OUTPUT before launch.
It must remain distinct from PHASE06_PARENT_OUTPUT.

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
  d2b.status
  d2b.pid
  d2b.started
  d2b.exit
  d2b_launch_manifest.txt
  logs/
    phase06_d1.log
    phase06_d2a.log
    phase06_d2b.log

For D2-B, the launch manifest additionally records the parent output path,
parent protocol hash, parent D3 hash, and D2-B addendum hash.

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

Expected D2-B counts:
  total         100
  none           50
  time_scaled    50
  N=5            50
  N=40           50

Ctrl+C in the monitor exits only the monitor process. The runner continues in
the tmux runner window.

Hard boundaries
---------------
Do not use this kit for reserved confirmatory seeds, Phase 0.5 continuation, or
any D4 intervention. D2-B is the only newly exposed stage in this addendum.

A negative or ambiguous diagnostic conclusion is a valid scientific result and
is not an execution failure.
