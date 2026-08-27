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

## Status vocabulary

- `environment=exact` means an exact historical environment lock is available and verified.
- `environment=reconstructed` means a best-effort reconstruction is explicitly documented.
- `environment=unknown` means the archive does not yet justify either stronger label.
- `rebuild=yes` or `rerun=yes` is shown only after that capability has been implemented and validated. R1 does not infer support from the mere existence of historical code.

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
