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
```

A single supported phase may also use `--destination`, but the destination must remain beneath `outputs/reproduction/<phase>/rebuild/`. `rebuild all` does not accept a custom destination.

`status` reports the registered evidence roots, manifest availability, report-source availability, environment status, and whether rebuild or rerun support has actually been established.

`verify` is read-only. It checks registered paths, frozen decision text, Git-SHA formatting, checksum-manifest syntax and paths, and SHA-256 digests over archived bytes that are available in the repository checkout. It never repairs, regenerates, or overwrites evidence.

`rebuild` is documentary reconstruction only. It does not train models, execute historical experiments, consume confirmatory seeds, or alter the scientific decision. Supported rebuilds first pass read-only historical verification and accepted R2 source validation, then write only beneath `outputs/reproduction/`.

A reproduction output under `outputs/reproduction/` is never treated as official historical evidence.

## Historical report-source recovery

R2 recovered deterministic, version-controlled documentary source for the Phase 0, Phase 0.5, and Phase 0.6 official DOCX reports. Each recovered bundle under `docs/reproducibility/<phase>/` contains:

- `report-source.md`, a non-generative deterministic rendering of the DOCX body;
- `extraction.json`, the ordered OOXML extraction snapshot;
- `report.yaml`, which cryptographically binds the recovered source and snapshot to the SHA-256 of the immutable official DOCX and records the accepted audit state.

The official DOCX under `docs/results/` remains authoritative. The recovered Markdown is a reproducibility aid, not a replacement historical report. Re-auditing re-extracts the DOCX read-only and must leave every historical archive byte unchanged.

No canonical report source is claimed for the post-Phase-0.6 exploratory archive because there is no registered official historical report for that phase.

## Deterministic rebuild semantics

R3 establishes deterministic documentary rebuild support for Phase 0, Phase 0.5, and Phase 0.6. The post-Phase-0.6 exploratory archive remains rebuild-unsupported because it has no registered official report source.

A rebuilt DOCX is compared to the immutable official report by normalized structure and content. The comparison checks ordered paragraph content, heading semantics, tables, captions, and image-placeholder counts. It explicitly records `byte_identical: false`; structural equivalence must never be described as byte-for-byte DOCX identity.

Every rebuild writes `rebuild-report.json` beside the candidate report. Artifact provenance is explicit:

- `generated` means a deterministic generator actually recreated that artifact;
- `reference-copy` means an immutable historical artifact was copied into the isolated reproduction tree for documentary completeness;
- a `reference-copy` is **not** evidence that the historical computational pipeline was rerun or that the artifact was regenerated.

The current historical rebuild manifests are conservative: legacy tables and figures are labeled `reference-copy`. Report reconstruction is source-driven. No legacy asset is silently relabeled as generated.

## Phase 0.6 checkpoint availability

The immutable Phase 0.6 `FINAL_MANIFEST.sha256` records historical PyTorch checkpoint paths ending in `.pt`, but those checkpoint binaries are not present in a fresh repository checkout. The repository historically ignores `*.pt`, so the verifier treats a missing target as non-failing only when the individual registered manifest explicitly declares the `*.pt` availability pattern.

Such entries are reported as `manifest_target_unavailable`, not as SHA-256 matches. Their archived digest is preserved in the historical manifest, but the absent binary cannot be re-hashed from the repository checkout. Present manifest targets continue to require normal SHA-256 verification, and an undeclared missing file remains a verification failure. This exception does not authorize training or reconstruct the omitted checkpoint binaries.

## Status vocabulary

- `environment=exact` means an exact historical environment lock is available and verified.
- `environment=reconstructed` means a best-effort reconstruction is explicitly documented.
- `environment=unknown` means the archive does not justify either stronger label.
- `rebuild=yes` means deterministic documentary rebuild support has been implemented and validated.
- `rerun=yes` would mean historical computational rerun support has separately been established.

At the end of R3, Phase 0, Phase 0.5, and Phase 0.6 have `rebuild_supported=true`; post-Phase-0.6 remains false. `rerun_supported` remains false for every registered phase. R3 therefore closes report reconstruction, not historical experiment re-execution.

## Evidence hierarchy

For each registered phase, the historical archive under `docs/results/` remains authoritative. The registry and `artifact-map.yaml` point to that evidence; they do not supersede it. Checksum manifests are interpreted using their historically recorded path base, and the verifier rejects manifest path traversal.

The registered line is:

1. Phase 0 — historical validation result.
2. Phase 0.5 — historical mechanistic validation result.
3. Phase 0.6 — historical diagnostic result, terminal at `D4-B AMBIGUOUS -> STOP`.
4. Post-Phase-0.6 optimization-conditioned analysis — exploratory only.

See `research-lineage.md` for the scientific transition boundary and `artifact-map.yaml` for machine-readable bindings.

## Non-retroactivity rule

Reproducibility closure must not improve the appearance of the old record by editing archived results, changing a frozen classification, silently substituting regenerated outputs, or relabeling exploratory work as confirmatory evidence. If a historical environment or source artifact is unavailable, the correct status is explicit unavailability or `unknown`, not reconstruction by assumption.
