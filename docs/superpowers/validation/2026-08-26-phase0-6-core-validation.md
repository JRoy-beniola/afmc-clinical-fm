# Phase 0.6 Core Task 12 Validation

Date: 2026-08-26

## Status and scope

- Implementation status: **READY FOR D1 EXECUTION**
- Branch: `phase0-6-diagnostics`
- Validated implementation SHA: `f1c1a4679d8b88c5cfe947f6e23278cb63ea48f8`
- Requested baseline SHA: `1bcde660ebcb44d488b570d4e99bbe6a8f7e8c2c`
- Validation fix: `f1c1a4679d8b88c5cfe947f6e23278cb63ea48f8` makes two behavior-preserving Ruff corrections in the Phase 0.6 progress monitor. The baseline SHA itself did not pass the full-repository Ruff gate.
- Scope: Phase 0.6 Task 12 implementation readiness only. No D1 or D2 scientific cell was executed. No confirmatory seed was used. D2-B and D4 were not implemented or executed.

## Bound artifact hashes

| Artifact | SHA-256 |
|---|---|
| `docs/superpowers/specs/2026-08-26-phase0-6-diagnostics-design.md` | `1e86f8f3f1848f1c1aa83a5f611fd3967a4d501a6213a8fbc270ae30442ef81c` |
| `configs/experiments/phase06.yaml` | `b07a87ff2958b6b5e91a2a93cde81fc7d78d93910d5e1769978437f0bdc951f2` |
| `configs/experiments/phase05.yaml` | `508d6703f6d7b3ae2b92068510b58f16cffa985a233a340e9228d96f476d4693` |

Command:

```bash
sha256sum \
  docs/superpowers/specs/2026-08-26-phase0-6-diagnostics-design.md \
  configs/experiments/phase06.yaml \
  configs/experiments/phase05.yaml
```

## Runtime

- Python: `3.14.4`
- Platform: `Linux-6.6.114.1-microsoft-standard-WSL2-x86_64-with-glibc2.43`
- Ruff: `0.16.4`
- pytest: `9.1.1`

The inherited `TEMP` and `TMP` variables pointed pytest at the mounted Windows temporary directory, where pytest capture cleanup produced `FileNotFoundError`. The successful validation runs prepared the WSL-local temporary directory once, then invoked the Task 12 pytest commands unchanged:

```bash
unset TEMP TMP
export TMPDIR=/tmp
```

## Ruff

Command:

```bash
./.venv/bin/python -m ruff check .
```

Observed result:

```text
All checks passed!
```

## Test suites

Commands:

```bash
./.venv/bin/python -m pytest tests/phase06 -q
./.venv/bin/python -m pytest tests/phase05 -q
./.venv/bin/python -m pytest -q
```

Observed results:

```text
146 passed in 30.21s
154 passed in 55.13s
537 passed in 259.44s (0:04:19)
```

- Phase 0.6 test count/result: **146 passed**
- Phase 0.5 regression count/result: **154 passed**
- Full-suite count/result: **537 passed**

## Scientific-boundary audits

Commands:

```bash
! grep -R "701\|702\|703\|704\|705\|706\|707\|708\|709\|710" -n src/afmc_fm/phase06 tests/phase06 configs/experiments/phase06.yaml | grep -v forbidden
! grep -R "confirmation\|robustness" -n src/afmc_fm/phase06/cli.py
! grep -R "afmc_fm.phase06" -n src/afmc_fm/phase05/training.py
```

The second and third negative greps passed with no output. The first literal grep found only the explicit firewall definitions and tests:

```text
src/afmc_fm/phase06/config.py:12:_FORBIDDEN_COHORT_SEEDS = tuple(range(701, 711))
src/afmc_fm/phase06/protocol.py:25:_FORBIDDEN_COHORT_SEEDS = tuple(range(701, 711))
tests/phase06/test_planning.py:162:        (701, 501, 601),
tests/phase06/test_protocol.py:30:@pytest.mark.parametrize("seed", range(701, 711))
tests/phase06/test_protocol.py:108:        "cohort": list(range(701, 711)),
```

Task 12 explicitly permits replacing that grep when tests legitimately contain the forbidden values. The executable-plan audit loaded the locked configuration and constructed plans without running any cell. Its observed output was:

```text
D1 cell count: 40
D1 unique cell IDs: 40
D1 stages: ['d1']
D1 reserved-seed intersections: cohort=[], subset=[], model=[]
D2-A cell count: 100
D2-A unique cell IDs: 100
D2-A stages: ['d2a']
D2-A reserved-seed intersections: cohort=[], subset=[], model=[]
Planned stages: ['d1', 'd2a']
D2-B planned: False
D4 planned: False
```

Audit conclusions:

- Seed firewall: **PASS**; executable D1 and D2-A plans have no reserved cohort, subset, or model seeds.
- Phase 0.6 CLI confirmation/robustness boundary: **PASS**.
- Phase 0.5 training import boundary: **PASS**.
- D1 planned cells: **40**.
- D2-A planned cells: **100**.
- D2-B implementation: **absent**.
- D4 implementation: **intentionally absent pending D3/addendum**.

## D0 non-interference proof

Command:

```bash
./.venv/bin/python -m pytest \
  tests/phase06/test_d0_instrumentation.py::test_diagnostics_are_byte_identical_to_default_training_path \
  -vv
```

Observed result:

```text
tests/phase06/test_d0_instrumentation.py::test_diagnostics_are_byte_identical_to_default_training_path PASSED [100%]

============================== 1 passed in 3.89s ===============================
```

D0 non-interference result: **PASS**. Diagnostics are byte-identical to the default training path.

## Readiness conclusion

All Task 12 implementation-readiness gates passed against `f1c1a4679d8b88c5cfe947f6e23278cb63ea48f8`. This authorizes starting D1 only; it does not execute D1, authorize D2-B, permit confirmatory seeds, or authorize D4.

**READY FOR D1 EXECUTION**
