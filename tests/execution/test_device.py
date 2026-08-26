import json

import pytest
import torch

from afmc_fm.cli import main
from afmc_fm.execution import device
from afmc_fm.execution.device import move_batch, resolve_device, runtime_diagnostics


def test_explicit_cpu_does_not_query_cuda(monkeypatch: pytest.MonkeyPatch):
    def fail_if_queried() -> bool:
        raise AssertionError("explicit CPU resolution queried CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", fail_if_queried)

    assert resolve_device("cpu").type == "cpu"


@pytest.mark.parametrize(
    ("cuda_available", "expected_type"),
    [(False, "cpu"), (True, "cuda")],
)
def test_auto_resolves_from_cuda_availability(
    monkeypatch: pytest.MonkeyPatch,
    cuda_available: bool,
    expected_type: str,
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)

    assert resolve_device("auto").type == expected_type


def test_explicit_cuda_fails_clearly_when_unavailable(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA was requested but is not available"):
        resolve_device("cuda")


def test_resolve_device_rejects_unknown_mode():
    with pytest.raises(ValueError, match="auto, cpu, or cuda"):
        resolve_device("tpu")


def test_move_batch_returns_all_tensors_on_requested_device():
    batch = {
        "features": torch.tensor([[1.0, 2.0]]),
        "targets": torch.tensor([3.0]),
    }

    moved = move_batch(batch, torch.device("cpu"))

    assert moved is not batch
    assert set(moved) == {"features", "targets"}
    assert all(tensor.device.type == "cpu" for tensor in moved.values())
    torch.testing.assert_close(moved["features"], torch.tensor([[1.0, 2.0]]))
    torch.testing.assert_close(moved["targets"], torch.tensor([3.0]))


def test_runtime_diagnostics_reports_required_cpu_metadata(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(device.os, "cpu_count", lambda: 12)

    diagnostics = runtime_diagnostics("cpu", workers=4)

    assert diagnostics["python_executable"] == device.sys.executable
    assert isinstance(diagnostics["platform"], str)
    assert diagnostics["platform"]
    assert diagnostics["wsl"] is device._is_wsl()
    assert diagnostics["torch_version"] == torch.__version__
    assert diagnostics["cuda_available"] is False
    assert diagnostics["cuda_runtime"] == torch.version.cuda
    assert diagnostics["gpu_name"] is None
    assert diagnostics["selected_device"] == "cpu"
    assert diagnostics["cpu_count"] == 12
    assert diagnostics["workers"] == 4
    assert "warning" not in diagnostics


def test_cpu_diagnostics_does_not_probe_gpu_name(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def fail_if_probed(index: int) -> str:
        raise AssertionError(f"CPU diagnostics probed CUDA device {index}")

    monkeypatch.setattr(torch.cuda, "get_device_name", fail_if_probed)

    diagnostics = runtime_diagnostics("cpu", workers=1)

    assert diagnostics["cuda_available"] is True
    assert diagnostics["gpu_name"] is None
    assert diagnostics["selected_device"] == "cpu"


def test_auto_diagnostics_reports_available_gpu(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: f"GPU {index}")

    diagnostics = runtime_diagnostics("auto", workers=2)

    assert diagnostics["cuda_available"] is True
    assert diagnostics["gpu_name"] == "GPU 0"
    assert diagnostics["selected_device"] == "cuda"


def test_runtime_diagnostics_warns_for_windows_python_inside_wsl(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
    monkeypatch.setattr(device.sys, "executable", "/mnt/c/Python/python.exe")

    diagnostics = runtime_diagnostics("cpu", workers=1)

    assert diagnostics["wsl"] is True
    assert "native Linux virtual environment" in diagnostics["warning"]


def test_diagnostics_cli_prints_stable_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    rc = main(["diagnostics", "--device", "cpu", "--workers", "3"])

    assert rc == 0
    printed = capsys.readouterr().out.strip()
    payload = json.loads(printed)
    assert payload["selected_device"] == "cpu"
    assert payload["workers"] == 3
    assert printed == json.dumps(payload, indent=2, sort_keys=True)


def test_diagnostics_cli_rejects_non_positive_worker_count(
    capsys: pytest.CaptureFixture[str],
):
    with pytest.raises(SystemExit, match="2"):
        main(["diagnostics", "--device", "cpu", "--workers", "0"])

    assert "workers must be at least 1" in capsys.readouterr().err
