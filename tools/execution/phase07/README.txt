Phase 0.7 execution boundary
============================

This directory documents the execution-facing Phase 0.7 commands. It does not contain an official-execution authorization artifact.

Safe pre-execution commands:

  afmc-phase07 protocol --output <protocol.json>
  afmc-phase07 plan --output <plan.json>
  afmc-phase07 verify --plan <plan.json>
  afmc-phase07 manifest --output <manifest.json>
  afmc-phase07 smoke --device cpu
  afmc-phase07 smoke --device cuda

The smoke command is explicitly non-official. It exercises the real Phase 0.5/0.7 PyTorch stopping-policy path on a tiny synthetic fixture and does not materialize or execute any of the frozen 200 Phase 0.7 cells.

Official execution is fail-closed and requires a clean checkout whose imported Phase 0.7 source tree HEAD exactly matches the authorized execution commit:

  afmc-phase07 official --authorization <authorization.json> --output <output-dir> --device cuda

The clean-worktree guard is anchored to the repository containing the imported Phase 0.7 execution module, not to the caller's current working directory. Running the CLI from another clean Git checkout cannot mask dirty imported execution or analysis source.

The Phase 0.5 and simulator configurations are loaded once for an invocation. The manifest hashes those exact loaded configuration objects, authorization binds those hashes, and the same objects are passed into official execution. They are not reloaded after authorization.

The authorization JSON must exactly bind the execution commit, canonical loaded Phase 0.5 configuration SHA-256, canonical loaded simulator configuration SHA-256, protocol-lock SHA-256, and Phase 0.7 plan SHA-256 emitted by the execution manifest. No such artifact is committed by this implementation PR. Creating or supplying one is a separate scientific authorization step.

Official execution persists each completed cell through the crash-resilient Phase07Store. A cell bundle is staged, hashed, validated, marked complete, and atomically renamed into its authoritative location. The stage-level COMPLETE marker is written only after the exact frozen 200-cell set validates.

If an authorized run is interrupted, restart from the same clean execution checkout, with the same authorization and output root, using:

  afmc-phase07 official --authorization <authorization.json> --output <output-dir> --device cuda --resume

Resume is explicit. It validates the persisted run identity and every authoritative completed cell, discards only abandoned same-run staging directories, skips valid completed cells, and reruns the interrupted/current cell from epoch 1. Corrupt or conflicting authoritative evidence fails closed and is never silently deleted or rerun.

The root phase07_metrics.csv is derived evidence only. It is rebuilt from validated cell bundles and is never trusted to determine resume state.

Official analysis is completion-gated:

  afmc-phase07 analyze --output <output-dir>

The analysis command requires the exact execution checkout, validates the persisted run identity, requires a valid stage COMPLETE marker and the exact frozen 200-cell bundle set, then rebuilds/loads metrics through Phase07Store before invoking the Phase 0.7 statistical analysis and adjudication. A loose or hand-edited CSV cannot be used as official evidence.

execution_provenance.json records invocation-level runtime metadata, completed-before/completed-after counts, the last attempted cell, catchable failures, timestamps, wall time, device, and the complete execution identity. It is informative only; authoritative resume state is reconstructed from validated cell bundles.

Do not place ad-hoc authorization files, exploratory results, or modified protocol inputs in this directory.
