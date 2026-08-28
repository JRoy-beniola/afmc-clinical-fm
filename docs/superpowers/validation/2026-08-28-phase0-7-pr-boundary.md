# Phase 0.7 PR and Execution Boundary

Status: **IMPLEMENTATION IN PROGRESS — OFFICIAL EXECUTION FORBIDDEN**

Branch: `phase0-7-optimization-horizon-intervention`

Base at branch creation: `854934a43d6819f5e26761a51364aa7e6c55a1b7`

Frozen design: `docs/superpowers/specs/2026-08-28-phase0-7-optimization-horizon-intervention-design.md`

Implementation plan: `docs/superpowers/plans/2026-08-28-phase0-7-optimization-horizon-intervention.md`

## Workflow lock

1. All implementation changes stay on the Phase 0.7 branch.
2. All changes are reviewed through the Phase 0.7 pull request.
3. CI must run on the exact PR head after each material implementation update.
4. Shared training-code changes require Phase 0.5 and Phase 0.6 regression coverage.
5. The PR may not be merged while required CI is failing or while important review findings remain unresolved.
6. Merge must use an expected-head guard.
7. After merge, the exact tested implementation commit is recorded as the candidate execution SHA.
8. No official Phase 0.7 training cell may run without a separate explicit human authorization after implementation and CI closure.

This record does not authorize execution of the official 200-cell matrix.