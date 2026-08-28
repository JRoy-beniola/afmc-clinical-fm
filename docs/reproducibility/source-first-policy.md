# Source-First Reproducibility Policy

This policy applies prospectively to AFMC research phases and to documentary records created after the historical Phase 0.6 line. Its purpose is to make future evidence and reports reproducible without rewriting old archives or presenting reconstructed metadata as if it had been captured historically.

## 1. Execution-time environment capture

Every future official execution must capture its reproducibility environment at execution time. The immutable execution record must include, at minimum:

- exact Python version;
- installed package versions or an immutable package lock and its hash;
- operating-system and platform information;
- PyTorch version where applicable;
- CUDA runtime/toolkit information where applicable;
- GPU model and relevant accelerator metadata where applicable;
- exact Git commit and working-tree cleanliness state;
- experiment config, protocol, and specification paths with cryptographic hashes;
- executed command or entry point;
- seed sets and protected-seed policy where applicable;
- input/parent evidence paths and hashes;
- output inventory and checksums.

A dependency file reconstructed after an execution is evidence for a reconstructed environment only. It must not be labeled an exact historical environment snapshot.

## 2. Canonical report source before archival

Future reports are source-first. Narrative, tables, figure declarations, captions, and section ordering must exist in version-controlled source before an official report is archived.

Scientific figures must retain their generator entry point, evidence inputs, parameters, and tool/commit provenance. Where conceptual diagrams exist, their editable diagram source must remain in Git; a rendered PNG, SVG, or PDF alone is not canonical source. PlantUML, Mermaid, Graphviz DOT, or another deterministic editable representation may be used.

A report manifest must bind the documentary source to the exact evidence inputs used to produce it and must declare the environment-capture record on which the reported execution depends.

## 3. Explicit archive/freeze operation

Promotion into an official evidence archive requires a separate, explicit **archive/freeze** operation after verification and review. Archive/freeze must validate source/evidence bindings, capture checksums, record the scientific classification, and make the resulting evidence boundary immutable.

`rebuild` and `rerun` are reproduction operations only. They **must not promote** generated, copied, or rerun artifacts into `docs/results/`, must not replace an existing official report, and must not alter a frozen decision record.

A candidate report or rerun output under `outputs/reproduction/` is never official evidence merely because it reproduces an earlier result.

## 4. Scientific decision and exploratory boundaries

Source-first documentation does not change the evidentiary status of the work it describes. Exploratory results remain exploratory unless a separately preregistered and authorized prospective experiment establishes a stronger status. Historical decisions remain frozen.

For the post-Phase-0.6 optimization-conditioned analysis, the source-first documentary record therefore preserves the exploratory classification while Phase 0.6 remains `D4-B AMBIGUOUS -> STOP`.

## 5. Non-retroactivity

Historical gaps remain gaps. Missing exact environment snapshots, omitted checkpoints, unavailable parents, or absent canonical launch commands must be reported as unknown, reconstructed, blocked, or unsupported as appropriate. The reproducibility layer must not fabricate completeness by inference.
