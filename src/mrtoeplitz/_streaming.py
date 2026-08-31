"""Streaming a transfer that does not fit the device.

A compressed transfer for a high-resolution subspace reconstruction can be
larger than the card. Rather than refusing, it is held in pinned host memory
and brought over in chunks, with the copy of one chunk overlapping the
multiply of the one before it.
"""

from __future__ import annotations

__all__ = ["CudaStreaming"]

from dataclasses import dataclass, replace
from importlib import import_module
from typing import Any


def _torch() -> Any:
    try:
        return import_module("torch")
    except ImportError as error:
        raise ImportError("streaming requires Torch: pip install mrtoeplitz") from error


@dataclass(frozen=True)
class CudaStreaming:
    """How a transfer too large for the device is streamed onto it.

    The transfer stays in pinned host memory and reaches the device in chunks
    of ``transfer_chunk_size`` locations. Each device owns ``streams`` CUDA
    streams, so the copy of one chunk overlaps the multiply of the one before
    it and the device never holds more than a couple of chunks at once.

    By default every visible GPU takes part; ``cuda:N`` or an explicit
    ``devices`` tuple constrains the set. Coils are independent until the sum
    that ends them, so more than one device divides them.

    ``transfer_precision="auto"`` narrows a real transfer to bfloat16 wherever
    the device supports it natively, halving what it occupies, and keeps a
    complex one at complex64. Accumulation stays in single precision either
    way.

    Examples
    --------
    >>> import mrtoeplitz as mt
    >>> policy = mt.CudaStreaming(streams=2)
    >>> policy.streams, policy.device
    (2, 'cuda')
    """

    device: str = "cuda"
    devices: tuple[str, ...] | None = None
    streams: int = 2
    pin_memory: bool = True
    transfer_chunk_size: int = 1048576
    physics_batch_size: int = 1
    kernel_residency: str = "auto"
    transfer_precision: str = "auto"
    max_device_fraction: float = 0.85

    def __post_init__(self) -> None:
        torch = _torch()
        device = torch.device(self.device)
        if device.type != "cuda":
            raise ValueError("CudaStreaming.device must be a CUDA device")
        if self.devices is not None:
            if not self.devices:
                raise ValueError("CudaStreaming.devices cannot be empty")
            normalized = tuple(str(torch.device(item)) for item in self.devices)
            if any(torch.device(item).type != "cuda" for item in normalized):
                raise ValueError("CudaStreaming.devices must contain CUDA devices")
            if len(set(normalized)) != len(normalized):
                raise ValueError("CudaStreaming.devices must be unique")
            object.__setattr__(self, "devices", normalized)
        if self.streams not in {1, 2}:
            raise ValueError("CudaStreaming.streams must be one or two")
        if self.transfer_chunk_size < 1:
            raise ValueError("transfer_chunk_size must be positive")
        if self.physics_batch_size < 1:
            raise ValueError("physics_batch_size must be positive")
        if self.kernel_residency not in {"auto", "host", "device"}:
            raise ValueError("kernel_residency must be 'auto', 'host', or 'device'")
        if self.transfer_precision not in {"auto", "float32", "bfloat16"}:
            raise ValueError(
                "transfer_precision must be 'auto', 'float32', or 'bfloat16'"
            )
        if not 0.0 < self.max_device_fraction <= 1.0:
            raise ValueError("max_device_fraction must be in (0, 1]")

    @property
    def torch_device(self) -> Any:
        """Primary configured :class:`torch.device`."""
        return self.torch_devices[0]

    @property
    def torch_devices(self) -> tuple[Any, ...]:
        """CUDA devices participating in streamed execution.

        An explicit ``devices`` tuple always wins.  Otherwise ``cuda:N`` is
        deliberately single-device, while the default unindexed ``cuda``
        discovers every visible GPU.  This makes a two-GPU recon host useful
        without changing application code and still gives callers a precise
        opt-out.
        """
        torch = _torch()
        if self.devices is not None:
            return tuple(torch.device(item) for item in self.devices)
        configured = torch.device(self.device)
        if configured.index is not None:
            return (configured,)
        count = torch.cuda.device_count()
        if count:
            return tuple(torch.device("cuda", index) for index in range(count))
        return (torch.device("cuda", torch.cuda.current_device()),)

    @property
    def device_count(self) -> int:
        """Number of GPUs selected by this policy."""
        return len(self.torch_devices)

    @property
    def execution_devices(self) -> tuple[Any, ...]:
        """One device entry per stream worker, ordered round-robin by GPU."""
        return tuple(
            device for _ in range(self.streams) for device in self.torch_devices
        )

    def for_device(self, device: Any, *, streams: int | None = None) -> CudaStreaming:
        """Return an equivalent policy constrained to one CUDA device."""
        selected = str(device)
        return replace(
            self,
            device=selected,
            devices=(selected,),
            streams=self.streams if streams is None else streams,
        )

    def ensure_available(self) -> None:
        """Raise a clear error when the configured CUDA device is unavailable."""
        torch = _torch()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA streaming requested but CUDA is unavailable")
        for device in self.torch_devices:
            torch.empty(0, device=device)
