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

Official execution is fail-closed:

  afmc-phase07 official --authorization <authorization.json> --output <output-dir> --device cuda

The authorization JSON must exactly bind the execution commit, protocol-lock SHA-256, and Phase 0.7 plan SHA-256 emitted by the execution manifest. No such artifact is committed by this implementation PR. Creating or supplying one is a separate scientific authorization step.

Do not place ad-hoc authorization files, exploratory results, or modified protocol inputs in this directory.
