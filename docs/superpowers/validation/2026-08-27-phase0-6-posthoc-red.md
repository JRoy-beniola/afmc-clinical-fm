# Phase 0.6 post-hoc optimization analysis — RED checkpoint

This branch is explicitly downstream of the merged Phase 0.6 terminal boundary at `38d7e238d36ad2abd97007c9b1d8c84575100784`.

The post-hoc analysis is exploratory only. It may not modify `docs/results/phase06/`, revise `D4-B AMBIGUOUS -> STOP`, authorize D4_CAPACITY_TIME, or touch confirmatory seeds.

TDD RED contract is defined in `tests/phase06/test_posthoc_optimization.py`. The tests intentionally import the not-yet-implemented `afmc_fm.phase06.posthoc_optimization` module and must fail until the production analysis implementation exists.
