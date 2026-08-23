"""Runtime diagnostics and Torch device helpers."""

import os
import platform
import sys

import torch


def _is_wsl() -> bool:
    release = platform.release().lower()
    return (
        "WSL_DISTRO_NAME" in os.environ
        or "WSL_INTEROP" in os.environ
        or "microsoft" in release
    )


def resolve_device(requested: str) -> torch.device:
    """Resolve an execution mode to a concrete Torch device."""
    if requested == "cpu":
        return torch.device("cpu")
    if requested not in {"auto", "cuda"}:
        raise ValueError("requested device must be auto, cpu, or cuda")

    cuda_available = torch.cuda.is_available()
    if requested == "cuda" and not cuda_available:
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device("cuda" if cuda_available else "cpu")


def move_batch(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Return a batch whose tensors are placed on ``device``."""
    return {name: tensor.to(device) for name, tensor in batch.items()}


def runtime_diagnostics(requested_device: str, workers: int) -> dict[str, object]:
    """Collect portable runtime and device metadata for benchmark preflight."""
    if workers < 1:
        raise ValueError("workers must be at least 1")

    selected_device = resolve_device(requested_device)
    cuda_available = (
        torch.cuda.is_available() if requested_device == "cpu" else selected_device.type == "cuda"
    )
    diagnostics: dict[str, object] = {
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "wsl": _is_wsl(),
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_runtime": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if selected_device.type == "cuda" else None,
        "selected_device": str(selected_device),
        "cpu_count": os.cpu_count(),
        "workers": workers,
    }
    if diagnostics["wsl"] and sys.executable.lower().endswith(".exe"):
        diagnostics["warning"] = (
            "Windows Python detected inside WSL; use a native Linux virtual environment."
        )
    return diagnostics
