AFMC Phase 0.6 D1 / D2-A / D2-B / D4-B CUDA Execution Kit
===========================================================

Purpose
-------
Run validated Phase 0.6 diagnostic stages on the local WSL/CUDA machine with
persistent logs and a read-only tmux progress dashboard.

The kit exposes only:
  D1   exact 40-cell diagnostic reproduction
  D2-A exact 100-cell orthogonal seed diagnostic
  D2-B exact 100-cell complementary orthogonal seed diagnostic
  D4-B exact 100-cell optimization/initialization stability diagnostic

Execution and adjudication are deliberately separate. The execution kit does
not run adjudicate, adjudicate-d2b, or adjudicate-d4b automatically. D2-B does
not authorize D4 by itself; D4-B is exposed only under its separately frozen
addendum and readiness boundary.

Scientific boundary
-------------------
Phase 0.5 remains terminated at Stage I-A. Phase 0.6 is diagnostic. Reserved
confirmatory seed namespaces remain forbidden.

D2-B consumes the immutable D1/D2-A/D3 parent. D4-B consumes two immutable
roots: that same core parent plus the completed D2-B child. Neither parent may
be rewritten by D4-B.

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

Validation boundaries
---------------------
D1 and D2-A:
  docs/superpowers/validation/2026-08-26-phase0-6-core-validation.md
  implementation status: READY FOR D1 EXECUTION

D2-B:
  docs/superpowers/validation/2026-08-26-phase0-6-d2b-validation.md
  implementation status: READY FOR D2-B EXECUTION

D4-B:
  docs/superpowers/validation/2026-08-27-phase0-6-d4b-validation.md
  implementation status: READY FOR D4-B EXECUTION

The runner extracts the validated implementation SHA from the selected record
and rejects execution-critical drift. For D2-B and D4-B this includes source,
configuration, pyproject.toml, runner, starter, monitor, and the corresponding
frozen execution addendum. A later validation-document-only commit may move
HEAD only when those critical paths remain byte-identical to the validated SHA.

Common preflight
----------------
The runner refuses execution unless all required validation/evidence files are
present, the tracked/staged working tree is clean, the project .venv exists,
nvidia-smi is available, PyTorch reports CUDA available, and any existing child
protocol lock is bound to the current execution HEAD.

D1 fresh launch
---------------
~/phase06-control/start_stage.sh d1 fresh

Default tmux session:
  phase06-d1

D1 resume:
  tmux kill-session -t phase06-d1 2>/dev/null || true
  ~/phase06-control/start_stage.sh d1 resume

D2-A fresh launch
-----------------
After D1 is complete and audited:

~/phase06-control/start_stage.sh d2a fresh

Default tmux session:
  phase06-d2a

D2-A resume:
  tmux kill-session -t phase06-d2a 2>/dev/null || true
  ~/phase06-control/start_stage.sh d2a resume

D2-B parent binding and launch
------------------------------
Bind the immutable core parent:

export PHASE06_PARENT_OUTPUT=/absolute/path/to/outputs/phase06_<parent_sha>

Fresh launch:

~/phase06-control/start_stage.sh d2b fresh

Default tmux session:
  phase06-d2b

Default child output:
  ~/afmc-clinical-fm/outputs/phase06_d2b_<CURRENT_HEAD>

Resume after diagnosing an interruption:

  tmux kill-session -t phase06-d2b 2>/dev/null || true
  export PHASE06_PARENT_OUTPUT=/absolute/path/to/outputs/phase06_<parent_sha>
  ~/phase06-control/start_stage.sh d2b resume

D2-B hard stop
--------------
A successful D2-B launcher stops after D2-B COMPLETE and D2-B analysis. It
does not run adjudicate-d2b. Cross-array adjudication remains a separate audited
action. D2-B execution itself does not authorize D4.

D4-B two-parent binding
-----------------------
D4-B requires both immutable evidence roots:

export PHASE06_CORE_PARENT_OUTPUT=/absolute/path/to/outputs/phase06_<core_parent_sha>
export PHASE06_D2B_PARENT_OUTPUT=/absolute/path/to/outputs/phase06_d2b_<d2b_parent_sha>

The launcher requires the core parent to expose its protocol, D1/D2-A COMPLETE
markers, and D3 adjudication. It requires the D2-B parent to expose its protocol,
D2-B COMPLETE marker, and D2-B cross-array adjudication. The Python CLI performs
the deeper canonical/provenance validation and recomputes the frozen D2-B
adjudication before creating or resuming the D4-B child.

All three roots must be pairwise distinct and non-nested in either direction.

D4-B fresh launch
-----------------
Only after the D4-B validation record says READY FOR D4-B EXECUTION and final
CI/review gates are green:

export PHASE06_CORE_PARENT_OUTPUT=/absolute/path/to/outputs/phase06_<core_parent_sha>
export PHASE06_D2B_PARENT_OUTPUT=/absolute/path/to/outputs/phase06_d2b_<d2b_parent_sha>
~/phase06-control/start_stage.sh d4b fresh

Default tmux session:
  phase06-d4b

Default child output:
  ~/afmc-clinical-fm/outputs/phase06_d4b_<CURRENT_HEAD>

D4-B runs exactly 100 CUDA cells: five fixed cohort/subset contexts, ten new
model seeds 1001..1010, and paired none/time_scaled flow modes at N=40.

D4-B resume after a genuine interruption
-----------------------------------------
First diagnose the interruption and preserve the child output. Then, if resume
is appropriate:

  tmux kill-session -t phase06-d4b 2>/dev/null || true
  export PHASE06_CORE_PARENT_OUTPUT=/absolute/path/to/outputs/phase06_<core_parent_sha>
  export PHASE06_D2B_PARENT_OUTPUT=/absolute/path/to/outputs/phase06_d2b_<d2b_parent_sha>
  ~/phase06-control/start_stage.sh d4b resume

Resume revalidates both immutable parents and reuses only the hash-bound D4-B
child protocol.

D4-B hard stop
--------------
A successful D4-B launcher stops after D4-B COMPLETE and the five required
analysis artifacts are written. It does not run adjudicate-d4b automatically.
The separate adjudication may return D4_CAPACITY_TIME as a route token only when
the frozen stability gate passes.

This kit does not authorize D4-D, capacity/time execution, full-factorial seed
expansion, Phase 0.5 continuation, or confirmatory-seed execution.

Tmux controls
-------------
For every stage:
  Ctrl-b 0   runner
  Ctrl-b 1   monitor
  Ctrl-b d   detach without stopping the run

Reattach with:
  tmux attach -t phase06-d1
  tmux attach -t phase06-d2a
  tmux attach -t phase06-d2b
  tmux attach -t phase06-d4b

Output identity
---------------
D1/D2-A default output:
  ~/afmc-clinical-fm/outputs/phase06_<CURRENT_HEAD>

D2-B default child:
  ~/afmc-clinical-fm/outputs/phase06_d2b_<CURRENT_HEAD>

D4-B default child:
  ~/afmc-clinical-fm/outputs/phase06_d4b_<CURRENT_HEAD>

Set PHASE06_OUTPUT only when an explicitly known child path is required. It must
respect the corresponding parent-root isolation rules.

Launch manifests
----------------
Each stage records its execution HEAD, validated SHA, validation-record hash,
config/spec hashes, Phase 0.5 evidence hashes, Python/Torch/CUDA information,
and GPU identity.

D2-B additionally records its parent path, parent protocol/D3 hashes, and D2-B
addendum hash.

D4-B additionally records:
  core parent path and protocol/D3 hashes
  D2-B parent path and protocol/adjudication hashes
  D4-B addendum hash

Monitor semantics
-----------------
The monitor reads persisted cell JSON files and reports total progress, counts by
N and flow mode, throughput/ETA, COMPLETE/analysis state, process status, HEAD,
output root, GPU telemetry, and visible corrupt cell JSON count.

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

Expected D4-B counts:
  total         100
  none           50
  time_scaled    50
  N=40          100

Ctrl+C exits only the monitor. The tmux runner continues.

Hard boundaries
---------------
Do not use this kit for reserved confirmatory seeds, Phase 0.5 continuation,
full-factorial expansion, or any unimplemented later D4 intervention. A fragile,
ambiguous, or negative D4-B result is a valid scientific stopping result.
