# AFMC Reproducibility Closure

This directory is the reproducibility index for the completed AFMC synthetic research line through the post-Phase-0.6 exploratory analysis.

It does **not** redefine historical evidence, reopen a scientific decision, authorize a new experiment, or make a historical execution environment look more exact than the archived record supports.

## Public interface

After installing the repository, use:

```text
afmc-reproduce status

afmc-reproduce verify phase0
afmc-reproduce verify phase05
afmc-reproduce verify phase06
afmc-reproduce verify phase06-posthoc
afmc-reproduce verify all

afmc-reproduce rebuild phase0
afmc-reproduce rebuild phase05
afmc-reproduce rebuild phase06
afmc-reproduce rebuild all

afmc-reproduce rerun phase0
afmc-reproduce rerun phase05
afmc-reproduce rerun phase06
```

A single supported rebuild may also use `--destination`, but the destination must remain beneath `outputs/reproduction/<phase>/rebuild/`. `rebuild all` does not accept a custom destination.

`rerun` is plan-first. Supplying a phase without `--execute` performs guarded readiness planning only; it must not launch historical computation. `--run-id` selects a fresh isolated destination beneath `outputs/reproduction/<phase>/rerun/`. `--execute` is an explicit acknowledgement boundary and is accepted only for a plan that is actually `READY`. There is no `rerun all` command.

The registered historical phases currently remain rerun-unsupported, so real historical execution is not enabled merely because the isolation/planning machinery exists. A fixture rerun exercises that machinery against an exact historical Git commit in an isolated worktree without claiming that the real historical archives contain every prerequisite needed for re-execution.

`status` reports archive and manifest availability, documentary-source availability, registered environment status, result kind, rebuild support, rerun support, rerun readiness, and the frozen decision. Software support and historical readiness are distinct claims.

`verify` is read-only. It checks registered paths, frozen decision text, Git-SHA formatting, checksum-manifest syntax and paths, and SHA-256 digests over archived bytes that are available in the repository checkout. It never repairs, regenerates, or overwrites evidence.

`rebuild` is documentary reconstruction only. It does not train models, execute historical experiments, consume confirmatory seeds, or alter the scientific decision. Supported rebuilds first pass read-only historical verification and accepted documentary-source validation, then write only beneath `outputs/reproduction/`.

A reproduction output under `outputs/reproduction/` is never treated as official historical evidence.

## Documentary source

### Historical official reports

R2 recovered deterministic, version-controlled documentary source for the Phase 0, Phase 0.5, and Phase 0.6 official DOCX reports. Each recovered bundle under `docs/reproducibility/<phase>/` contains:

- `report-source.md`, a non-generative deterministic rendering of the DOCX body;
- `extraction.json`, the ordered OOXML extraction snapshot;
- `report.yaml`, which cryptographically binds the recovered source and snapshot to the SHA-256 of the immutable official DOCX and records the accepted audit state.

The official DOCX under `docs/results/` remains authoritative. The recovered Markdown is a reproducibility aid, not a replacement historical report. Re-auditing re-extracts the DOCX read-only and must leave every historical archive byte unchanged.

### Post-Phase-0.6 source-first record

The post-Phase-0.6 optimization-conditioned analysis has no registered official historical DOCX, so R4 does **not** invent one. Instead, `docs/reproducibility/phase06_posthoc/report-source.md` is a canonical source-first documentary record backed by `report.yaml` and the already-frozen post-hoc decision record, execution provenance, checksum manifest, and archived analysis artifacts.

That source states the same exploratory classification as the frozen archive and preserves the Phase 0.6 terminal boundary exactly: `D4-B AMBIGUOUS -> STOP`. It does not promote the retrospective analysis to confirmatory evidence and does not authorize Phase 0.7, protected-confirmatory seed use, D4 capacity/time execution, or any reinterpretation of Phase 0.6.

Future official work should follow `source-first-policy.md`: narrative, table, figure, and editable diagram source belongs in version control before archival; execution-time environment/provenance capture must be explicit; and archival/freeze is a distinct operation from `rebuild` or `rerun`.

## Deterministic rebuild semantics

R3 establishes deterministic documentary rebuild support for Phase 0, Phase 0.5, and Phase 0.6. The post-Phase-0.6 exploratory archive remains rebuild-unsupported because there is no historical official report target to reconstruct. Its source-first documentary record does not create a retroactive official report.

A rebuilt DOCX is compared to the immutable official report by normalized structure and content. The comparison checks ordered paragraph content, heading semantics, tables, captions, and image-placeholder counts. It explicitly records `byte_identical: false`; structural equivalence must never be described as byte-for-byte DOCX identity.

Every rebuild writes `rebuild-report.json` beside the candidate report. Artifact provenance is explicit:

- `generated` means a deterministic generator actually recreated that artifact;
- `reference-copy` means an immutable historical artifact was copied into the isolated reproduction tree for documentary completeness;
- a `reference-copy` is **not** evidence that the historical computational pipeline was rerun or that the artifact was regenerated.

The current historical rebuild manifests are conservative: legacy tables and figures are labeled `reference-copy`. Report reconstruction is source-driven. No legacy asset is silently relabeled as generated.

## Historical rerun semantics

R4 adds guarded rerun specification, planning, worktree isolation, environment classification, execution provenance, and layered comparison primitives. Those primitives are tested with a fixture that proves execution occurs from the requested historical commit rather than current `HEAD`, and that output remains isolated from the historical archive.

That software capability must not be confused with real-phase readiness. Phase 0, Phase 0.5, and Phase 0.6 remain registered as rerun-unsupported because the preserved record does not justify a complete safe re-execution claim. Their rerun specifications document the relevant historical commands and limitations rather than silently substituting current code or reconstructed prerequisites. The post-Phase-0.6 exploratory phase has no historical computational rerun specification.

A blocked or unsupported historical rerun is a valid reproducibility result. Missing historical commits, checkpoints, parents, environment evidence, commands, or policy-safe seed bindings must be surfaced rather than reconstructed by assumption. `--execute` never turns a blocked plan into an execution.

## Phase 0.6 checkpoint availability

The immutable Phase 0.6 `FINAL_MANIFEST.sha256` records historical PyTorch checkpoint paths ending in `.pt`, but those checkpoint binaries are not present in a fresh repository checkout. The repository historically ignores `*.pt`, so the verifier treats a missing target as non-failing only when the individual registered manifest explicitly declares the `*.pt` availability pattern.

Such entries are reported as `manifest_target_unavailable`, not as SHA-256 matches. Their archived digest is preserved in the historical manifest, but the absent binary cannot be re-hashed from the repository checkout. Present manifest targets continue to require normal SHA-256 verification, and an undeclared missing file remains a verification failure. This exception does not authorize training or reconstruct the omitted checkpoint binaries.

## Status vocabulary

- `environment=exact` means an exact historical environment lock is available and verified.
- `environment=reconstructed` means a best-effort reconstruction is explicitly documented.
- `environment=unknown` means the archive does not justify either stronger label.
- `result_kind=historical` identifies a registered historical result; `result_kind=exploratory` preserves the post-Phase-0.6 retrospective boundary.
- `rebuild=yes` means deterministic documentary rebuild support has been implemented and validated.
- `rerun=yes` would mean a real historical phase has separately been registered for guarded computational rerun support.
- `rerun_readiness=UNSUPPORTED` means the phase is not registered for real historical rerun execution; readiness is not inferred from the existence of generic rerun machinery.

At R4 closure, Phase 0, Phase 0.5, and Phase 0.6 have `rebuild_supported=true`; post-Phase-0.6 remains false. `rerun_supported` remains false for every registered real phase, and public status therefore reports `rerun_readiness=UNSUPPORTED`. This is intentional: reproducibility closure records the limits of the archive instead of fabricating completeness.

## Closure immutability gate

The R4 closure tests snapshot all four official evidence roots and exercise:

- read-only verification;
- historical report-source audits;
- supported documentary rebuilds under isolated reproduction destinations;
- historical environment classification;
- real-phase rerun planning;
- an isolated fixture rerun from a historical Git commit.

The before/after SHA-256 tree snapshots for the four registered evidence roots must remain identical. Reproduction outputs are also checked for isolation, and test cleanup cannot be counted as evidence regeneration.

No Phase 0.7 execution and no protected confirmatory execution are part of reproducibility closure.

## Evidence hierarchy

For each registered phase, the historical archive under `docs/results/` remains authoritative. The registry and `artifact-map.yaml` point to that evidence; they do not supersede it. Checksum manifests are interpreted using their historically recorded path base, and the verifier rejects manifest path traversal.

The registered line is:

1. Phase 0 — historical validation result.
2. Phase 0.5 — historical mechanistic validation result.
3. Phase 0.6 — historical diagnostic result, terminal at `D4-B AMBIGUOUS -> STOP`.
4. Post-Phase-0.6 optimization-conditioned analysis — exploratory only.

See `research-lineage.md` for the scientific transition boundary, `artifact-map.yaml` for machine-readable bindings, and `source-first-policy.md` for prospective capture requirements.

## Non-retroactivity rule

Reproducibility closure must not improve the appearance of the old record by editing archived results, changing a frozen classification, silently substituting regenerated outputs, or relabeling exploratory work as confirmatory evidence. If a historical environment or source artifact is unavailable, the correct status is explicit unavailability or `unknown`, not reconstruction by assumption.
