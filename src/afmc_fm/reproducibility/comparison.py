from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from .integrity import CheckResult
from .verify import _normalize_classification

ReproductionVerdict = Literal[
    "EXACT",
    "NUMERICALLY_REPRODUCED",
    "SCIENTIFICALLY_REPRODUCED",
    "FAILED_REPRODUCTION",
]


@dataclass(frozen=True)
class StructuralTablePolicy:
    path: Path
    key_columns: tuple[str, ...]
    required_columns: tuple[str, ...]


@dataclass(frozen=True)
class NumericFieldPolicy:
    column: str
    atol: float
    rtol: float
    equal_nan: bool


@dataclass(frozen=True)
class NumericTablePolicy:
    path: Path
    key_columns: tuple[str, ...]
    fields: tuple[NumericFieldPolicy, ...]


@dataclass(frozen=True)
class ScientificDecisionPolicy:
    path: Path
    expected_classification: str


@dataclass(frozen=True)
class ComparisonPolicy:
    structural_tables: tuple[StructuralTablePolicy, ...]
    exact_files: tuple[Path, ...]
    numeric_tables: tuple[NumericTablePolicy, ...]
    scientific_decision: ScientificDecisionPolicy


@dataclass(frozen=True)
class ComparisonLayer:
    name: str
    checks: tuple[CheckResult, ...]

    @property
    def ok(self) -> bool:
        return all(check.ok for check in self.checks)


@dataclass(frozen=True)
class ReproductionComparison:
    verdict: ReproductionVerdict
    structural: ComparisonLayer
    exact: ComparisonLayer
    numerical: ComparisonLayer
    scientific: ComparisonLayer


@dataclass(frozen=True)
class _TablePair:
    official: pd.DataFrame | None
    rerun: pd.DataFrame | None
    checks: tuple[CheckResult, ...]



def _check(code: str, ok: bool, subject: str, detail: str) -> CheckResult:
    return CheckResult(code=code, ok=ok, subject=subject, detail=detail)



def _read_table_pair(
    official_root: Path,
    rerun_root: Path,
    relative_path: Path,
) -> _TablePair:
    official_path = official_root / relative_path
    rerun_path = rerun_root / relative_path
    checks: list[CheckResult] = []

    official: pd.DataFrame | None = None
    rerun: pd.DataFrame | None = None
    for label, path in (("official", official_path), ("rerun", rerun_path)):
        if not path.is_file():
            checks.append(
                _check(
                    "comparison_path_missing",
                    False,
                    relative_path.as_posix(),
                    f"{label} comparison table is missing: {path}",
                )
            )
            continue
        try:
            table = pd.read_csv(path)
        except (OSError, UnicodeDecodeError, pd.errors.ParserError) as exc:
            checks.append(
                _check(
                    "comparison_table_unreadable",
                    False,
                    relative_path.as_posix(),
                    f"unable to read {label} comparison table: {exc}",
                )
            )
            continue
        if label == "official":
            official = table
        else:
            rerun = table

    return _TablePair(official=official, rerun=rerun, checks=tuple(checks))



def _required_columns_check(
    table: pd.DataFrame,
    *,
    label: str,
    relative_path: Path,
    required: tuple[str, ...],
) -> CheckResult:
    missing = tuple(column for column in required if column not in table.columns)
    return _check(
        "comparison_columns_match" if not missing else "comparison_columns_missing",
        not missing,
        relative_path.as_posix(),
        f"{label} table contains required columns"
        if not missing
        else f"{label} table is missing columns: {', '.join(missing)}",
    )



def _duplicate_keys_check(
    table: pd.DataFrame,
    *,
    label: str,
    relative_path: Path,
    key_columns: tuple[str, ...],
) -> CheckResult:
    duplicated = int(table.duplicated(list(key_columns), keep=False).sum())
    return _check(
        "comparison_keys_unique" if duplicated == 0 else "comparison_keys_duplicate",
        duplicated == 0,
        relative_path.as_posix(),
        f"{label} table keys are unique"
        if duplicated == 0
        else f"{label} table contains {duplicated} rows with duplicate keys",
    )



def _key_set(table: pd.DataFrame, key_columns: tuple[str, ...]) -> set[tuple[object, ...]]:
    return set(table.loc[:, list(key_columns)].itertuples(index=False, name=None))



def _structural_layer(
    official_root: Path,
    rerun_root: Path,
    policies: tuple[StructuralTablePolicy, ...],
) -> ComparisonLayer:
    checks: list[CheckResult] = []
    for policy in policies:
        pair = _read_table_pair(official_root, rerun_root, policy.path)
        checks.extend(pair.checks)
        if pair.official is None or pair.rerun is None:
            continue

        required = tuple(dict.fromkeys((*policy.key_columns, *policy.required_columns)))
        official_columns = _required_columns_check(
            pair.official,
            label="official",
            relative_path=policy.path,
            required=required,
        )
        rerun_columns = _required_columns_check(
            pair.rerun,
            label="rerun",
            relative_path=policy.path,
            required=required,
        )
        checks.extend((official_columns, rerun_columns))
        if not official_columns.ok or not rerun_columns.ok:
            continue

        official_unique = _duplicate_keys_check(
            pair.official,
            label="official",
            relative_path=policy.path,
            key_columns=policy.key_columns,
        )
        rerun_unique = _duplicate_keys_check(
            pair.rerun,
            label="rerun",
            relative_path=policy.path,
            key_columns=policy.key_columns,
        )
        checks.extend((official_unique, rerun_unique))
        if not official_unique.ok or not rerun_unique.ok:
            continue

        official_keys = _key_set(pair.official, policy.key_columns)
        rerun_keys = _key_set(pair.rerun, policy.key_columns)
        missing = official_keys - rerun_keys
        unexpected = rerun_keys - official_keys
        same_keys = not missing and not unexpected
        checks.append(
            _check(
                "structural_keys_match" if same_keys else "structural_keys_mismatch",
                same_keys,
                policy.path.as_posix(),
                f"row/key set matches ({len(official_keys)} keys)"
                if same_keys
                else (
                    "row/key set differs: "
                    f"missing={len(missing)}, unexpected={len(unexpected)}"
                ),
            )
        )

    return ComparisonLayer(name="structural", checks=tuple(checks))



def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()



def _exact_layer(
    official_root: Path,
    rerun_root: Path,
    files: tuple[Path, ...],
) -> ComparisonLayer:
    checks: list[CheckResult] = []
    for relative_path in files:
        official = official_root / relative_path
        rerun = rerun_root / relative_path
        if not official.is_file() or not rerun.is_file():
            missing = [
                label
                for label, path in (("official", official), ("rerun", rerun))
                if not path.is_file()
            ]
            checks.append(
                _check(
                    "exact_path_missing",
                    False,
                    relative_path.as_posix(),
                    f"exact comparison file missing from: {', '.join(missing)}",
                )
            )
            continue

        official_digest = _sha256(official)
        rerun_digest = _sha256(rerun)
        identical = official_digest == rerun_digest
        checks.append(
            _check(
                "exact_file_match" if identical else "exact_file_mismatch",
                identical,
                relative_path.as_posix(),
                "deterministic file SHA-256 matches"
                if identical
                else (
                    f"deterministic file differs: official={official_digest}, "
                    f"rerun={rerun_digest}"
                ),
            )
        )

    return ComparisonLayer(name="exact", checks=tuple(checks))



def _numeric_values(
    table: pd.DataFrame,
    column: str,
) -> np.ndarray | None:
    try:
        values = pd.to_numeric(table[column], errors="raise")
    except (TypeError, ValueError):
        return None
    return values.to_numpy(dtype=float)



def _aligned_tables(
    official: pd.DataFrame,
    rerun: pd.DataFrame,
    key_columns: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    if official.duplicated(list(key_columns), keep=False).any():
        return None
    if rerun.duplicated(list(key_columns), keep=False).any():
        return None
    if _key_set(official, key_columns) != _key_set(rerun, key_columns):
        return None

    keys = list(key_columns)
    official_indexed = official.set_index(keys).sort_index()
    rerun_indexed = rerun.set_index(keys).sort_index()
    return official_indexed, rerun_indexed



def _numeric_layer(
    official_root: Path,
    rerun_root: Path,
    policies: tuple[NumericTablePolicy, ...],
) -> ComparisonLayer:
    checks: list[CheckResult] = []
    for policy in policies:
        pair = _read_table_pair(official_root, rerun_root, policy.path)
        checks.extend(pair.checks)
        if pair.official is None or pair.rerun is None:
            continue

        field_names = tuple(field.column for field in policy.fields)
        required = tuple(dict.fromkeys((*policy.key_columns, *field_names)))
        official_columns = _required_columns_check(
            pair.official,
            label="official",
            relative_path=policy.path,
            required=required,
        )
        rerun_columns = _required_columns_check(
            pair.rerun,
            label="rerun",
            relative_path=policy.path,
            required=required,
        )
        checks.extend((official_columns, rerun_columns))
        if not official_columns.ok or not rerun_columns.ok:
            continue

        aligned = _aligned_tables(pair.official, pair.rerun, policy.key_columns)
        if aligned is None:
            checks.append(
                _check(
                    "numeric_keys_mismatch",
                    False,
                    policy.path.as_posix(),
                    "numeric comparison requires unique, identical row/key sets",
                )
            )
            continue

        official_table, rerun_table = aligned
        for field in policy.fields:
            official_values = _numeric_values(official_table, field.column)
            rerun_values = _numeric_values(rerun_table, field.column)
            if official_values is None or rerun_values is None:
                checks.append(
                    _check(
                        "numeric_column_invalid",
                        False,
                        f"{policy.path.as_posix()}:{field.column}",
                        "numeric comparison column contains non-numeric values",
                    )
                )
                continue

            close = np.isclose(
                official_values,
                rerun_values,
                atol=field.atol,
                rtol=field.rtol,
                equal_nan=field.equal_nan,
            )
            matches = bool(np.all(close))
            mismatch_count = int((~close).sum())
            finite = np.isfinite(official_values) & np.isfinite(rerun_values)
            max_abs_error = (
                float(np.max(np.abs(official_values[finite] - rerun_values[finite])))
                if np.any(finite)
                else None
            )
            detail = (
                f"numeric field matches atol={field.atol}, rtol={field.rtol}, "
                f"equal_nan={field.equal_nan}"
                if matches
                else (
                    f"numeric field has {mismatch_count} mismatches at "
                    f"atol={field.atol}, rtol={field.rtol}, "
                    f"equal_nan={field.equal_nan}; max_abs_error={max_abs_error}"
                )
            )
            checks.append(
                _check(
                    "numeric_field_match" if matches else "numeric_field_mismatch",
                    matches,
                    f"{policy.path.as_posix()}:{field.column}",
                    detail,
                )
            )

    return ComparisonLayer(name="numerical", checks=tuple(checks))



def _classification_check(
    root: Path,
    policy: ScientificDecisionPolicy,
    *,
    label: str,
) -> CheckResult:
    path = root / policy.path
    if not path.is_file():
        return _check(
            "scientific_decision_missing",
            False,
            policy.path.as_posix(),
            f"{label} scientific decision record is missing",
        )
    try:
        observed = _normalize_classification(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as exc:
        return _check(
            "scientific_decision_unreadable",
            False,
            policy.path.as_posix(),
            f"unable to read {label} scientific decision: {exc}",
        )

    expected = _normalize_classification(policy.expected_classification)
    matches = expected in observed
    return _check(
        "scientific_decision_match" if matches else "scientific_decision_mismatch",
        matches,
        policy.path.as_posix(),
        f"{label} evidence resolves frozen classification: {policy.expected_classification}"
        if matches
        else (
            f"{label} evidence does not resolve frozen classification: "
            f"{policy.expected_classification}"
        ),
    )



def _scientific_layer(
    official_root: Path,
    rerun_root: Path,
    policy: ScientificDecisionPolicy,
) -> ComparisonLayer:
    return ComparisonLayer(
        name="scientific",
        checks=(
            _classification_check(official_root, policy, label="official"),
            _classification_check(rerun_root, policy, label="rerun"),
        ),
    )



def _verdict(
    structural: ComparisonLayer,
    exact: ComparisonLayer,
    numerical: ComparisonLayer,
    scientific: ComparisonLayer,
) -> ReproductionVerdict:
    if not scientific.ok:
        return "FAILED_REPRODUCTION"
    if structural.ok and exact.ok and numerical.ok:
        return "EXACT"
    if structural.ok and numerical.ok:
        return "NUMERICALLY_REPRODUCED"
    return "SCIENTIFICALLY_REPRODUCED"



def compare_reproduction(
    official_root: Path,
    rerun_root: Path,
    policy: ComparisonPolicy,
) -> ReproductionComparison:
    """Compare one isolated rerun against frozen evidence without mutating either tree."""

    official = Path(official_root)
    rerun = Path(rerun_root)
    structural = _structural_layer(official, rerun, policy.structural_tables)
    exact = _exact_layer(official, rerun, policy.exact_files)
    numerical = _numeric_layer(official, rerun, policy.numeric_tables)
    scientific = _scientific_layer(official, rerun, policy.scientific_decision)
    return ReproductionComparison(
        verdict=_verdict(structural, exact, numerical, scientific),
        structural=structural,
        exact=exact,
        numerical=numerical,
        scientific=scientific,
    )
