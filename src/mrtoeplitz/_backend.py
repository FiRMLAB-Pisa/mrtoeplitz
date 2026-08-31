"""Reaching an MRI-NUFFT backend, and the conventions it answers in."""

from __future__ import annotations

__all__ = ["register_torch_cufinufft"]

from importlib import import_module
from math import prod, sqrt
from typing import Any

from ._torch_cufinufft import register_torch_cufinufft


def _require_mrinufft() -> Any:
    """Import MRI-NUFFT, or say which extra provides it.

    Registers the Torch-native CUFINUFFT adapter on the way through, so a CUDA
    build hands tensors straight to the backend rather than through CuPy.
    """
    try:
        module = import_module("mrinufft")
    except ImportError as error:
        raise ImportError(
            "gridding a transfer requires mri-nufft: pip install mrtoeplitz[nufft]"
        ) from error
    register_torch_cufinufft()
    return module


def _mrinufft_norm_factor(shape: tuple[int, ...]) -> float:
    """Return the normalization an mri-nufft operator on ``shape`` divides by."""
    return sqrt(prod(shape) * 2 ** len(shape))


def _base_fourier_operator(native_operator: Any) -> Any:
    """Return the undecorated Fourier operator beneath mri-nufft wrappers."""
    return getattr(native_operator, "_fourier_op", native_operator)
