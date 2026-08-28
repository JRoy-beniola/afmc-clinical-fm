# AFMC Reproducibility Closure

This directory is the read-only reproducibility index for the completed AFMC synthetic research line through the post-Phase-0.6 exploratory analysis.

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
```

`status` reports the registered evidence roots, manifest availability, report-source availability, environment status, and whether rebuild or rerun support has actually been established.

`verify` is read-only. It checks registered paths, frozen decision text, Git-SHA formatting, checksum-manifest syntax and paths, and SHA-256 digests over the archived bytes. It never repairs, regenerates, or overwrites evidence.

A reproduction output under `outputs/reproduction/` is never treated as official historical evidence.

## Historical report-source recovery

R2 recovered deterministic, version-controlled documentary source for the Phase 0, Phase 0.5, and Phase 0.6 official DOCX reports. Each recovered bundle under `docs/reproducibility/<phase>/` contains:

- `report-source.md`, a non-generative deterministic rendering of the DOCX body;
- `extraction.json`, the ordered OOXML extraction snapshot;
- `report.yaml`, which cryptographically binds the recovered source and snapshot to the SHA-256 of the immutable official DOCX and records the accepted audit state.

The official DOCX under `docs/results/` remains authoritative. The recovered Markdown is a reproducibility aid, not a replacement historical report. Re-auditing re-extracts the DOCX read-only and must leave every historical archive byte unchanged.

No canonical report source is claimed for the post-Phase-0.6 exploratory archive because there is no registered official historical report for that phase.

## Status vocabulary

- `environment=exact` means an exact historical environment lock is available and verified.
- `environment=reconstructed` means a best-effort reconstruction is explicitly documented.
- `environment=unknown` means the archive does not yet justify either stronger label.
- `rebuild=yes` or `rerun=yes` is shown only after that capability has been implemented and validated. Report-source recovery alone does not establish either capability.

At the end of R2, the environment status remains unchanged and `rebuild_supported` / `rerun_supported` remain false for every registered phase. Those capabilities, if supportable, are established only by later reproducibility stages with their own verification gates.

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
