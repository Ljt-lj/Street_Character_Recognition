"""Resolve Ultralytics `device` for CPU/GPU without duplicating logic."""

from __future__ import annotations


def resolve_yolo_device(spec: str | None) -> str | int:
    """
    Return a value suitable for Ultralytics `train(..., device=...)` / `predict(...)`.

    - ``auto`` / empty: CUDA GPU 0 if ``torch.cuda.is_available()`` else ``\"cpu\"``.
    - ``cpu``: CPU only.
    - Otherwise pass through (e.g. ``0``, ``1``, ``cuda:0``, ``0,1``).
    """
    s = (spec or "").strip()
    if not s or s.lower() == "auto":
        try:
            import torch

            return 0 if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    if s.lower() == "cpu":
        return "cpu"
    return s


def device_is_cpu(device: str | int | list) -> bool:
    if isinstance(device, str) and device.lower() == "cpu":
        return True
    return False
