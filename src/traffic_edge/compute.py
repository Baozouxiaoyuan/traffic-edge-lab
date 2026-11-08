"""Choose the inference device without requiring CUDA on CPU installations."""
from __future__ import annotations


def normalize_device(device: str) -> str:
    value = str(device).strip().lower()
    if value not in {"auto", "cpu", "cuda"}:
        raise ValueError("Compute device must be auto, cpu or cuda")
    return value


def resolve_device(device: str = "auto") -> str:
    value = normalize_device(device)
    if value == "cpu":
        return "cpu"
    import torch

    available = torch.cuda.is_available()
    if value == "cuda" and not available:
        raise ValueError(
            "NVIDIA CUDA is unavailable. Install an NVIDIA driver and a CUDA-enabled "
            "PyTorch build using SETUP_CAPSTONE.cmd cuda or python tools/setup.py --compute cuda. "
            "Choose CPU or Automatic to run on this computer now."
        )
    return "cuda:0" if available else "cpu"


def compute_status() -> dict:
    import torch

    available = torch.cuda.is_available()
    return {
        "cuda_available": available,
        "gpu_name": torch.cuda.get_device_name(0) if available else None,
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "automatic_device": "cuda:0" if available else "cpu",
    }
