from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath

import pandas as pd
import pytest

from afmc_fm.execution.jobs import CellResult, ShardSpec
from afmc_fm.execution.persistence import RunStore
from afmc_fm.experiments.runner import ExperimentConfig
from afmc_fm.simulator.config import SimulatorConfig


def test_canonical_config_hash_is_order_independent_and_task8_compatible():
    from afmc_fm.execution.persistence import canonical_config_bytes, canonical_config_hash
    from afmc_fm.execution.scheduler import _run_identity

    first = {"unicode": "μ", "b": (2, 3), "a": 1}
    second = {"a": 1, "b": [2, 3], "unicode": "μ"}

    assert canonical_config_bytes(first) == b'{"a":1,"b":[2,3],"unicode":"\\u03bc"}'
    assert canonical_config_bytes(second) == canonical_config_bytes(first)
    assert canonical_config_hash(first) == (
        "746ce46518133feecdfe652e15db2af8f8a338ef363186a7bac1dfa500b939b9"
    )
    identity = _run_identity(SimulatorConfig(), ExperimentConfig())
    assert identity.simulator_config_hash == (
        "04335fa381a667767d8a76c98a121324d3ba9f396fe43b7cba4dcd767d278cfc"
    )
    assert identity.experiment_config_hash == (
        "4803eff41a0f10b546c0eed52b6d34bfb4b068786f2adb178b5f04939880be21"
    )


def test_canonical_config_normalizes_nested_dataclasses_sets_and_path_flavours():
    from afmc_fm.execution.persistence import canonical_config_bytes

    @dataclass(frozen=True)
    class NestedConfig:
        labels: frozenset[str]
        root: object

    first = {
        "choices": {2, 1},
        "nested": NestedConfig(
            labels=frozenset(("zeta", "alpha")),
            root=PureWindowsPath("C:\\research\\run"),
        ),
        "posix": PurePosixPath("/research/run"),
    }
    second = {
        "posix": PurePosixPath("/research/run"),
        "nested": NestedConfig(
            root=PureWindowsPath("C:/research/run"),
            labels=frozenset(("alpha", "zeta")),
        ),
        "choices": {1, 2},
    }

    expected = (
        b'{"choices":{"__afmc_type__":"set","items":[1,2]},'
        b'"nested":{"labels":{"__afmc_type__":"frozenset",'
        b'"items":["alpha","zeta"]},"root":{"__afmc_type__":"path",'
        b'"flavour":"windows","value":"C:/research/run"}},'
        b'"posix":{"__afmc_type__":"path","flavour":"posix",'
        b'"value":"/research/run"}}'
    )
    assert canonical_config_bytes(first) == expected
    assert canonical_config_bytes(second) == expected


def test_canonical_config_rejects_host_dependent_native_paths():
    from afmc_fm.execution.persistence import canonical_config_bytes

    with pytest.raises(
        TypeError,
        match="native Path is host-dependent; use PurePosixPath or PureWindowsPath",
    ):
        canonical_config_bytes({"root": Path("research/run")})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_canonical_config_rejects_nonfinite_values_at_any_depth(value):
    from afmc_fm.execution.persistence import canonical_config_bytes

    with pytest.raises(ValueError, match="finite"):
        canonical_config_bytes({"nested": [value]})


def test_canonical_config_rejects_unsupported_nested_types():
    from afmc_fm.execution.persistence import canonical_config_bytes

    with pytest.raises(TypeError, match="unsupported canonical config type.*object"):
        canonical_config_bytes({"nested": object()})


def test_cpu_manifest_reports_available_gpu_independently_of_selected_device(
    monkeypatch,
):
    from afmc_fm.execution.manifest import collect_runtime_metadata

    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_name", lambda index: "Test GPU")
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda index: (8, 9))

    runtime = collect_runtime_metadata("cpu")

    assert runtime["cuda_available"] is True
    assert runtime["gpu_name"] == "Test GPU"
    assert runtime["gpu_compute_capability"] == "8.9"


def test_build_run_manifest_records_complete_reproducibility_and_failure_counts():
    from afmc_fm.execution.manifest import build_run_manifest
    from afmc_fm.execution.scheduler import ShardFailure

    simulator = SimulatorConfig(cohort_size=30, followup_days=45.0)
    experiment = ExperimentConfig(
        train_sizes=(5,),
        worlds=("smooth", "jumps"),
        cohort_seeds=(17,),
        subset_seeds=(23,),
        model_seeds=(31,),
        models=("engineered_linear", "representation_mlp"),
        ablations=("none",),
        max_epochs=1,
        patience=1,
    )
    expected = {
        "smooth__cohort17__subset23__model31": frozenset({"cell-a", "cell-b"}),
        "jumps__cohort17__subset23__model31": frozenset({"cell-c"}),
        "cancelled__cohort17__subset23__model31": frozenset(
            {"cell-d", "cell-e"}
        ),
    }
    completed = {
        "smooth__cohort17__subset23__model31": frozenset({"cell-a", "cell-b"}),
    }
    failure = ShardFailure(
        shard_id="jumps__cohort17__subset23__model31",
        world="jumps",
        cohort_seed=17,
        subset_seed=23,
        model_seed=31,
        exception_type="RuntimeError",
        exception_message="controlled failure",
        device="cpu",
    )
    runtime = {
        "python_version": "3.12.4",
        "library_versions": {
            "matplotlib": "3.9.1",
            "numpy": "2.0.1",
            "pandas": "2.2.2",
            "scikit_learn": "1.5.1",
            "scipy": "1.14.0",
            "torch": "2.3.1",
        },
        "os": "Linux-6.6-test",
        "wsl": True,
        "cuda_available": False,
        "cuda_runtime": None,
        "gpu_name": None,
        "gpu_compute_capability": None,
        "cpu_logical_count": 16,
    }

    manifest = build_run_manifest(
        simulator=simulator,
        experiment=experiment,
        resolved_device="cpu",
        worker_count=2,
        backend_ids={
            "engineered_linear": "torch_ridge",
            "representation_mlp": "torch_lbfgs",
        },
        started_at=datetime(2026, 8, 23, 12, 0, tzinfo=UTC),
        ended_at=datetime(2026, 8, 23, 12, 0, 3, tzinfo=UTC),
        wall_time_seconds=3.25,
        expected_by_shard=expected,
        completed_by_shard=completed,
        failures=(failure,),
        cancelled_shard_ids=frozenset(
            {"cancelled__cohort17__subset23__model31"}
        ),
        execution_commit_sha="a" * 40,
        runtime_metadata=runtime,
        template_seed=7,
    )

    assert manifest["protocol_anchor"] == (
        "be5a66b2e45362f60c90844e4e25673fb7bb3e21"
    )
    assert manifest["execution_commit_sha"] == "a" * 40
    assert manifest["simulator_config_hash"] == (
        "3572312e08e3e5778284dcd077981a7a418229cc5b6a204a1fbd0e14868b8f8e"
    )
    assert manifest["experiment_config_hash"] == (
        "72c2fe753219d8aba2c605b91b15d418baa85f9116906d9c9a893805bfa4ceaf"
    )
    assert manifest["python_version"] == "3.12.4"
    assert manifest["library_versions"] == runtime["library_versions"]
    assert manifest["os"] == "Linux-6.6-test"
    assert manifest["wsl"] is True
    assert manifest["resolved_device"] == "cpu"
    assert manifest["cuda_available"] is False
    assert manifest["cuda_runtime"] is None
    assert manifest["gpu_name"] is None
    assert manifest["gpu_compute_capability"] is None
    assert manifest["cpu_logical_count"] == 16
    assert manifest["worker_count"] == 2
    assert manifest["backend_ids"] == {
        "engineered_linear": "torch_ridge",
        "representation_mlp": "torch_lbfgs",
    }
    assert manifest["started_at"] == "2026-08-23T12:00:00+00:00"
    assert manifest["ended_at"] == "2026-08-23T12:00:03+00:00"
    assert manifest["generated_at"] == manifest["ended_at"]
    assert manifest["wall_time_seconds"] == 3.25
    assert manifest["expected_shard_count"] == 3
    assert manifest["completed_shard_count"] == 1
    assert manifest["failed_shard_count"] == 1
    assert manifest["cancelled_shard_count"] == 1
    assert manifest["incomplete_shard_count"] == 0
    assert manifest["cancelled_shard_ids"] == [
        "cancelled__cohort17__subset23__model31"
    ]
    assert manifest["incomplete_shard_ids"] == []
    assert manifest["expected_cell_count"] == 5
    assert manifest["completed_cell_count"] == 2
    assert manifest["failed_cell_count"] == 1
    assert manifest["cancelled_cell_count"] == 2
    assert manifest["incomplete_cell_count"] == 0
    assert manifest["failures"] == [
        {
            "cohort_seed": 17,
            "device": "cpu",
            "exception_message": "controlled failure",
            "exception_type": "RuntimeError",
            "model_seed": 31,
            "shard_id": "jumps__cohort17__subset23__model31",
            "subset_seed": 23,
            "world": "jumps",
        }
    ]
    assert manifest["output_schema_version"] == 1
    assert manifest["schema_version"] == 1
    assert manifest["synthetic"] is True
    assert manifest["template_seed"] == 7


def _aggregation_experiment() -> ExperimentConfig:
    return ExperimentConfig(
        train_sizes=(5,),
        worlds=("smooth",),
        cohort_seeds=(17,),
        subset_seeds=(23,),
        model_seeds=(31,),
        models=(
            "engineered_linear",
            "representation_linear",
            "gru_from_scratch",
            "flow_jump",
        ),
        ablations=("none", "no_flow"),
        max_epochs=1,
        patience=1,
    )


def _metric_cell(
    shard: ShardSpec,
    model: str,
    ablation: str,
    value: float,
) -> CellResult:
    return CellResult(
        shard=shard,
        benchmark="low_n",
        n_train=5,
        model=model,
        ablation=ablation,
        metrics=pd.DataFrame(
            [
                {
                    "model": model,
                    "ablation": ablation,
                    "n_train": 5,
                    "n_fit": 4,
                    "n_validation": 1,
                    "seed": shard.subset_seed,
                    "cohort_seed": shard.cohort_seed,
                    "subset_seed": shard.subset_seed,
                    "model_seed": shard.model_seed,
                    "split": "test",
                    "site_or_shift": "all",
                    "metric": "mae",
                    "value": value,
                    "trainable_parameters": 17,
                    "backend": "torch_ridge" if model.endswith("linear") else None,
                    "world": shard.world,
                    "benchmark": "low_n",
                }
            ]
        ),
    )


def _persist_complete_aggregation_fixture(output: Path):
    from afmc_fm.execution.manifest import execution_commit_sha
    from afmc_fm.execution.scheduler import _run_identity

    simulator = SimulatorConfig(cohort_size=30, followup_days=45.0)
    experiment = _aggregation_experiment()
    shard = ShardSpec("smooth", 17, 23, 31)
    cells = [
        _metric_cell(shard, "engineered_linear", "none", 0.4),
        _metric_cell(shard, "representation_linear", "none", 0.2),
        _metric_cell(shard, "gru_from_scratch", "none", 0.3),
        _metric_cell(shard, "flow_jump", "none", 0.1),
        _metric_cell(shard, "flow_jump", "no_flow", 0.5),
    ]
    store = RunStore(output, _run_identity(simulator, experiment))
    expected = frozenset(cell.cell_id for cell in cells)
    store.initialize_run(
        {shard.shard_id: expected},
        execution_commit_sha=execution_commit_sha(),
        original_started_at=datetime(2026, 8, 23, 12, tzinfo=UTC),
        resume=False,
    )
    for cell in reversed(cells):
        store.write_cell(cell)
    store.mark_shard_complete(shard.shard_id, expected)
    return simulator, experiment, store, cells


def test_reaggregation_is_deterministic_and_never_trains_or_simulates(
    tmp_path,
    monkeypatch,
):
    from afmc_fm.execution import manifest

    DERIVED_ARTIFACT_NAMES = manifest.DERIVED_ARTIFACT_NAMES
    SCIENTIFIC_SORT_KEY = manifest.SCIENTIFIC_SORT_KEY
    reaggregate_benchmark_outputs = manifest.reaggregate_benchmark_outputs

    output = tmp_path / "run"
    simulator, experiment, store, cells = _persist_complete_aggregation_fixture(output)
    first = reaggregate_benchmark_outputs(simulator, experiment, output)
    first_bytes = {
        name: (output / name).read_bytes() for name in DERIVED_ARTIFACT_NAMES
    }
    root_before = (output / "run_record.json").read_bytes()
    for name in DERIVED_ARTIFACT_NAMES:
        (output / name).unlink()

    original_iter = RunStore.iter_metric_rows

    def reversed_metric_rows(self, expected_cell_ids=None):
        return iter(reversed(list(original_iter(self, expected_cell_ids))))

    def forbidden(*args, **kwargs):
        raise AssertionError("reaggregation attempted simulation or model fitting")

    monkeypatch.setattr(RunStore, "iter_metric_rows", reversed_metric_rows)
    monkeypatch.setattr("afmc_fm.execution.scheduler.run_shard", forbidden)
    monkeypatch.setattr("afmc_fm.execution.jobs.simulate_cohort", forbidden)
    monkeypatch.setattr("afmc_fm.execution.jobs.simulate_world", forbidden)
    monkeypatch.setattr(
        "afmc_fm.execution.jobs._run_configured_benchmarks_on_cohort", forbidden
    )
    monkeypatch.setattr("afmc_fm.simulator.cohort.simulate_cohort", forbidden)
    monkeypatch.setattr("afmc_fm.simulator.cohort.simulate_world", forbidden)
    monkeypatch.setattr("afmc_fm.experiments.runner._run_low_n_on_cohort", forbidden)
    monkeypatch.setattr(
        "afmc_fm.experiments.runner._run_observation_shift_on_cohort", forbidden
    )
    monkeypatch.setattr("afmc_fm.experiments.runner._fit_neural", forbidden)
    monkeypatch.setattr("afmc_fm.models.baselines.TorchRidgeRegressor.fit", forbidden)
    monkeypatch.setattr(
        "afmc_fm.models.baselines.GradientBoostingRegressorBaseline.fit",
        forbidden,
    )
    monkeypatch.setattr(
        "afmc_fm.models.baselines.TorchMLPRegressorBaseline.fit", forbidden
    )
    monkeypatch.setattr("afmc_fm.models.baselines.MLPRegressorBaseline.fit", forbidden)
    monkeypatch.setattr(manifest, "execution_commit_sha", forbidden)

    second = reaggregate_benchmark_outputs(simulator, experiment, output)

    assert list(first.columns) == [
        "model",
        "ablation",
        "n_train",
        "n_fit",
        "n_validation",
        "seed",
        "cohort_seed",
        "subset_seed",
        "model_seed",
        "split",
        "site_or_shift",
        "metric",
        "value",
        "trainable_parameters",
        "backend",
        "world",
        "benchmark",
    ]
    assert first.loc[:, list(SCIENTIFIC_SORT_KEY)].to_records(index=False).tolist() == sorted(
        first.loc[:, list(SCIENTIFIC_SORT_KEY)].to_records(index=False).tolist()
    )
    pd.testing.assert_frame_equal(first, second)
    assert {
        name: (output / name).read_bytes() for name in DERIVED_ARTIFACT_NAMES
    } == first_bytes
    assert (output / "run_record.json").read_bytes() == root_before
    assert pd.read_csv(output / "ablation_metrics.csv")["ablation"].tolist() == [
        "no_flow"
    ]
    gate = pd.read_csv(output / "gate_summary.csv")
    assert gate.loc[0, "flow_jump_beats_both_primary_comparators"]
    assert len(cells) == len(store.load_completed_cell_ids()) == 5


def test_reaggregation_rejects_valid_but_unplanned_persisted_cells(tmp_path):
    from afmc_fm.execution.jobs import CellResult
    from afmc_fm.execution.manifest import reaggregate_benchmark_outputs

    output = tmp_path / "run"
    simulator, experiment, store, cells = _persist_complete_aggregation_fixture(output)
    stale_shard = ShardSpec("jumps", 17, 23, 31)
    stale = _metric_cell(stale_shard, "engineered_linear", "none", -999.0)
    store.write_cell(
        CellResult(
            shard=stale_shard,
            benchmark=stale.benchmark,
            n_train=stale.n_train,
            model=stale.model,
            ablation=stale.ablation,
            metrics=stale.metrics,
        )
    )

    with pytest.raises(ValueError, match="unplanned persisted cells.*jumps"):
        reaggregate_benchmark_outputs(simulator, experiment, output)

    assert len(cells) == 5


def test_reaggregation_rejects_duplicate_scientific_metric_keys(tmp_path):
    import json

    from afmc_fm.execution.manifest import reaggregate_benchmark_outputs

    output = tmp_path / "run"
    simulator, experiment, _, _ = _persist_complete_aggregation_fixture(output)
    path = next(output.glob("shards/*/cells/*.json"))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["metric_rows"].append(dict(payload["metric_rows"][0]))
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate scientific metric keys"):
        reaggregate_benchmark_outputs(simulator, experiment, output)


def test_reaggregation_requires_the_exact_root_run_record(tmp_path):
    from afmc_fm.execution.manifest import reaggregate_benchmark_outputs

    output = tmp_path / "run"
    simulator, experiment, _, _ = _persist_complete_aggregation_fixture(output)
    (output / "run_record.json").unlink()

    with pytest.raises(ValueError, match="missing run root record"):
        reaggregate_benchmark_outputs(simulator, experiment, output)


def test_artifact_generation_failure_preserves_prior_complete_generation(
    tmp_path,
    monkeypatch,
):
    import json

    from afmc_fm.execution import manifest

    output = tmp_path / "run"
    simulator, experiment, _, _ = _persist_complete_aggregation_fixture(output)
    manifest.reaggregate_benchmark_outputs(simulator, experiment, output)
    original = {
        name: (output / name).read_bytes()
        for name in manifest.DERIVED_ARTIFACT_NAMES
    }
    cell_path = next(output.glob("shards/*/cells/*.json"))
    payload = json.loads(cell_path.read_text(encoding="utf-8"))
    payload["metric_rows"][0]["value"] += 10.0
    cell_path.write_text(json.dumps(payload), encoding="utf-8")

    def injected_plot_failure(*args, **kwargs):
        raise RuntimeError("injected staged plot failure")

    monkeypatch.setattr(manifest, "_plot_learning_curves", injected_plot_failure)

    with pytest.raises(RuntimeError, match="injected staged plot failure"):
        manifest.reaggregate_benchmark_outputs(simulator, experiment, output)

    assert {
        name: (output / name).read_bytes()
        for name in manifest.DERIVED_ARTIFACT_NAMES
    } == original
    assert not list(output.glob(".artifacts-*"))


def test_scheduler_writes_partial_artifacts_and_truthful_structured_failure_manifest(
    tmp_path,
):
    import json

    from afmc_fm.execution.manifest import DERIVED_ARTIFACT_NAMES
    from afmc_fm.execution.scheduler import ExecutionOptions, run_scheduled_benchmark

    simulator = SimulatorConfig(cohort_size=30, followup_days=45.0)
    experiment = ExperimentConfig(
        train_sizes=(5,),
        worlds=("not_a_world", "smooth"),
        cohort_seeds=(17,),
        subset_seeds=(23,),
        model_seeds=(31,),
        models=("engineered_linear",),
        ablations=("none",),
        max_epochs=1,
        patience=1,
    )
    output = tmp_path / "run"

    metrics = run_scheduled_benchmark(
        simulator,
        experiment,
        output,
        ExecutionOptions(device="cpu", workers=2, resume=False, fail_fast=False),
        template_seed=7,
    )

    assert all((output / name).exists() for name in DERIVED_ARTIFACT_NAMES)
    persisted_metrics = pd.read_csv(output / "metrics.csv")
    pd.testing.assert_frame_equal(
        metrics.fillna(value={"backend": "<missing>"}).reset_index(drop=True),
        persisted_metrics.fillna(value={"backend": "<missing>"}),
        check_dtype=False,
    )
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["expected_shard_count"] == 2
    assert manifest["completed_shard_count"] == 1
    assert manifest["failed_shard_count"] == 1
    assert manifest["expected_cell_count"] == 2
    assert manifest["completed_cell_count"] == 1
    assert manifest["failed_cell_count"] == 1
    assert manifest["failures"] == metrics.attrs["failures"]
    assert manifest["worker_count"] == 2
    assert manifest["resolved_device"] == "cpu"
    assert manifest["template_seed"] == 7
    assert len(manifest["execution_commit_sha"]) == 40
    assert manifest["backend_ids"] == {"engineered_linear": "torch_ridge"}
    assert manifest["started_at"] <= manifest["ended_at"]
    assert manifest["wall_time_seconds"] >= 0.0
