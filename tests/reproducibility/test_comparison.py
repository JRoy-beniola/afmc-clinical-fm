from pathlib import Path

import pandas as pd
from afmc_fm.reproducibility.comparison import (
    ComparisonPolicy,
    NumericFieldPolicy,
    NumericTablePolicy,
    ScientificDecisionPolicy,
    StructuralTablePolicy,
    compare_reproduction,
)


def _write_fixture(root: Path, *, metadata: str, values: list[float], decision: str) -> None:
    root.mkdir(parents=True)
    (root / "metadata.json").write_text(metadata, encoding="utf-8")
    pd.DataFrame(
        {
            "cell": ["a", "b"],
            "seed": [101, 202],
            "value": values,
        }
    ).to_csv(root / "metrics.csv", index=False)
    (root / "decision.md").write_text(decision + "\n", encoding="utf-8")


def _policy(*, equal_nan: bool = False) -> ComparisonPolicy:
    return ComparisonPolicy(
        structural_tables=(
            StructuralTablePolicy(
                path=Path("metrics.csv"),
                key_columns=("cell", "seed"),
                required_columns=("value",),
            ),
        ),
        exact_files=(Path("metadata.json"),),
        numeric_tables=(
            NumericTablePolicy(
                path=Path("metrics.csv"),
                key_columns=("cell", "seed"),
                fields=(
                    NumericFieldPolicy(
                        column="value",
                        atol=1e-6,
                        rtol=1e-6,
                        equal_nan=equal_nan,
                    ),
                ),
            ),
        ),
        scientific_decision=ScientificDecisionPolicy(
            path=Path("decision.md"),
            expected_classification="D4-B AMBIGUOUS → STOP",
        ),
    )


def test_exact_identity_is_exact(tmp_path: Path):
    official = tmp_path / "official"
    rerun = tmp_path / "rerun"
    _write_fixture(
        official,
        metadata='{"protocol":"locked"}\n',
        values=[1.0, 2.0],
        decision="D4-B AMBIGUOUS → STOP",
    )
    _write_fixture(
        rerun,
        metadata='{"protocol":"locked"}\n',
        values=[1.0, 2.0],
        decision="d4-b ambiguous -> stop",
    )

    report = compare_reproduction(official, rerun, _policy())

    assert report.verdict == "EXACT"
    assert report.structural.ok
    assert report.exact.ok
    assert report.numerical.ok
    assert report.scientific.ok


def test_exact_failure_with_numeric_tolerance_is_numerically_reproduced(tmp_path: Path):
    official = tmp_path / "official"
    rerun = tmp_path / "rerun"
    _write_fixture(
        official,
        metadata='{"generated_at":"historical"}\n',
        values=[1.0, 2.0],
        decision="D4-B AMBIGUOUS → STOP",
    )
    _write_fixture(
        rerun,
        metadata='{"generated_at":"reproduction"}\n',
        values=[1.0000005, 2.0],
        decision="D4-B AMBIGUOUS -> STOP",
    )

    report = compare_reproduction(official, rerun, _policy())

    assert report.verdict == "NUMERICALLY_REPRODUCED"
    assert report.structural.ok
    assert not report.exact.ok
    assert report.numerical.ok
    assert report.scientific.ok
    assert any(not check.ok for check in report.exact.checks)


def test_numeric_failure_with_same_decision_is_scientifically_reproduced(tmp_path: Path):
    official = tmp_path / "official"
    rerun = tmp_path / "rerun"
    _write_fixture(
        official,
        metadata='{"protocol":"locked"}\n',
        values=[1.0, 2.0],
        decision="D4-B AMBIGUOUS → STOP",
    )
    _write_fixture(
        rerun,
        metadata='{"protocol":"locked"}\n',
        values=[1.25, 2.0],
        decision="D4-B AMBIGUOUS -> STOP",
    )

    report = compare_reproduction(official, rerun, _policy())

    assert report.verdict == "SCIENTIFICALLY_REPRODUCED"
    assert report.structural.ok
    assert report.exact.ok
    assert not report.numerical.ok
    assert report.scientific.ok
    assert any(not check.ok for check in report.numerical.checks)


def test_changed_scientific_decision_fails_reproduction(tmp_path: Path):
    official = tmp_path / "official"
    rerun = tmp_path / "rerun"
    _write_fixture(
        official,
        metadata='{"protocol":"locked"}\n',
        values=[1.0, 2.0],
        decision="D4-B AMBIGUOUS → STOP",
    )
    _write_fixture(
        rerun,
        metadata='{"protocol":"locked"}\n',
        values=[1.0, 2.0],
        decision="D4-B PASS",
    )

    report = compare_reproduction(official, rerun, _policy())

    assert report.verdict == "FAILED_REPRODUCTION"
    assert report.structural.ok
    assert report.exact.ok
    assert report.numerical.ok
    assert not report.scientific.ok
    assert any(not check.ok for check in report.scientific.checks)


def test_structural_key_mismatch_is_preserved_under_scientific_equivalence(tmp_path: Path):
    official = tmp_path / "official"
    rerun = tmp_path / "rerun"
    _write_fixture(
        official,
        metadata='{"protocol":"locked"}\n',
        values=[1.0, 2.0],
        decision="D4-B AMBIGUOUS → STOP",
    )
    _write_fixture(
        rerun,
        metadata='{"protocol":"locked"}\n',
        values=[1.0, 2.0],
        decision="D4-B AMBIGUOUS -> STOP",
    )
    rerun_table = pd.read_csv(rerun / "metrics.csv")
    rerun_table.loc[1, "seed"] = 303
    rerun_table.to_csv(rerun / "metrics.csv", index=False)

    report = compare_reproduction(official, rerun, _policy())

    assert report.verdict == "SCIENTIFICALLY_REPRODUCED"
    assert not report.structural.ok
    assert not report.numerical.ok
    assert report.scientific.ok
    assert any(not check.ok for check in report.structural.checks)


def test_nan_equality_is_explicitly_controlled_by_field_policy(tmp_path: Path):
    official = tmp_path / "official"
    rerun = tmp_path / "rerun"
    _write_fixture(
        official,
        metadata='{"protocol":"locked"}\n',
        values=[float("nan"), 2.0],
        decision="D4-B AMBIGUOUS → STOP",
    )
    _write_fixture(
        rerun,
        metadata='{"protocol":"locked"}\n',
        values=[float("nan"), 2.0],
        decision="D4-B AMBIGUOUS -> STOP",
    )

    strict = compare_reproduction(official, rerun, _policy(equal_nan=False))
    permissive = compare_reproduction(official, rerun, _policy(equal_nan=True))

    assert strict.verdict == "SCIENTIFICALLY_REPRODUCED"
    assert not strict.numerical.ok
    assert permissive.verdict == "EXACT"
    assert permissive.numerical.ok
