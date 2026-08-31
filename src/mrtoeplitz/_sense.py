"""Applying a kernel through coil sensitivities.

A SENSE normal is ``sum_c conj(m_c) * N(m_c x)`` with ``N`` the Toeplitz
normal. The coils are independent until that sum, so they are taken a batch at
a time and, where there is more than one device, divided between them.
"""

from __future__ import annotations

__all__ = ["apply_sense"]

from concurrent.futures import ThreadPoolExecutor
from importlib import import_module
from types import SimpleNamespace
from typing import Any

from ._backend import _base_fourier_operator
from ._coils import CoilKernels
from ._kernel import CompactToeplitzKernel, _device_is_full, as_torch


def _sense_maps(native_operator: Any, reference: Any) -> Any:
    """Return sensitivity maps as a Torch tensor, on whatever device holds them.

    A normal application reads one coil at a time, so maps the caller left on
    the host are staged coil by coil rather than moved whole -- the difference
    is the whole bank against one map of it.
    """
    torch = import_module("torch")
    base = _base_fourier_operator(native_operator)
    maps = getattr(base, "smaps", None)
    if maps is None:
        return torch.ones(
            (1, *base.shape),
            dtype=reference.dtype,
            device=reference.device,
        )
    if isinstance(maps, CoilKernels):
        # Already answers shape, ndim and coil slicing as the dense bank would,
        # and materialising it here to check that would be the whole point lost.
        return maps
    maps = as_torch(maps).to(reference.dtype)
    spatial_ndim = len(base.shape)
    if maps.ndim == spatial_ndim:
        return maps[None]
    if maps.ndim in {spatial_ndim + 1, spatial_ndim + 2}:
        return maps
    raise ValueError(
        "sensitivity maps must have shape (coils, *image_shape) or "
        "(batch, coils, *image_shape)"
    )


def _coils_split_across_devices(
    kernel: CompactToeplitzKernel,
    image: Any,
    maps: Any,
    streaming: Any,
    *,
    batched_maps: bool,
    n_coils: int,
) -> Any:
    """Sum a normal application over coils divided between CUDA devices.

    Coils are independent until the sum that ends them, so each device is given
    a share of them, its own copy of the transfer and its own copy of the
    image, and returns the part of the sum it computed.

    This has not been run on a machine with more than one GPU. What it assumes
    of a second device is what ``for_device`` and ``_apply_sense_toeplitz``
    already assume of the first.
    """
    devices = streaming.torch_devices[: min(streaming.device_count, n_coils)]
    edges = [(index * n_coils) // len(devices) for index in range(len(devices) + 1)]

    def share(position: int) -> Any:
        device = devices[position]
        start, stop = edges[position], edges[position + 1]
        coils = (maps[:, start:stop] if batched_maps else maps[start:stop]).to(device)
        held = SimpleNamespace(
            shape=kernel.image_shape,
            smaps=coils,
            uses_sense=True,
        )
        return apply_sense(
            kernel.for_device(device),
            image.to(device),
            held,
            coil_batch_size=1,
        )

    with ThreadPoolExecutor(max_workers=len(devices)) as workers:
        parts = list(workers.map(share, range(len(devices))))
    total = parts[0].to(image.device)
    for part in parts[1:]:
        total = total + part.to(image.device)
    return total


def apply_sense(
    kernel: CompactToeplitzKernel,
    image: Any,
    native_operator: Any,
    *,
    right_factors: Any | None = None,
    left_factors: Any | None = None,
    coil_batch_size: int = 1,
    streaming: Any | None = None,
) -> Any:
    """Apply a compact transfer between optional spatial factor banks."""
    torch = import_module("torch")
    if streaming is not None and streaming.device_count > 1:
        # Coils are independent until their final SENSE reduction.  Group at
        # least one coil per device so even a single-image reconstruction can
        # fan its Toeplitz work across a multi-GPU recon host.
        coil_batch_size = max(coil_batch_size, streaming.device_count)
    maps = _sense_maps(native_operator, image)
    # An image is (batch, *spatial) and unbatched maps are (coils, *spatial),
    # so the two carry the same rank; only the maps' own rank separates them.
    batched_maps = maps.ndim == len(kernel.image_shape) + 2
    if batched_maps:
        if maps.shape[0] == 1:
            maps = maps.expand(image.shape[0], *maps.shape[1:])
        elif maps.shape[0] != image.shape[0]:
            raise ValueError(
                "batched sensitivity maps must have one entry per image batch"
            )
        n_coils = maps.shape[1]
    else:
        n_coils = maps.shape[0]
    if (
        streaming is not None
        and streaming.device_count > 1
        and n_coils > 1
        and left_factors is None
        and right_factors is None
    ):
        return _coils_split_across_devices(
            kernel,
            image,
            maps,
            streaming,
            batched_maps=batched_maps,
            n_coils=n_coils,
        )
    result_rank = 1 if left_factors is not None else kernel.rank
    result = torch.zeros(
        (image.shape[0], result_rank, *kernel.image_shape),
        dtype=image.dtype,
        device=image.device,
    )
    if right_factors is not None:
        right_factors = as_torch(right_factors, device=image.device).to(image.dtype)
        right_factors = right_factors.reshape(kernel.rank, *kernel.image_shape)
    if left_factors is not None:
        left_factors = as_torch(left_factors, device=image.device).to(image.dtype)
        left_factors = left_factors.reshape(kernel.rank, *kernel.image_shape)

    staged_coils = None
    if (
        streaming is not None
        and image.device.type == "cpu"
        and streaming.pin_memory
        and coil_batch_size > 1
    ):
        staged_coils = torch.empty(
            (
                image.shape[0],
                min(coil_batch_size, n_coils),
                image.shape[1],
                *kernel.image_shape,
            ),
            dtype=image.dtype,
            device="cpu",
            pin_memory=True,
        )

    for start in range(0, n_coils, coil_batch_size):
        if batched_maps:
            coil_maps = maps[:, start : start + coil_batch_size].to(image.device)
            left = image[:, None]
            right = coil_maps[:, :, None]
        else:
            coil_maps = maps[start : start + coil_batch_size].to(image.device)
            left = image[:, None]
            right = coil_maps[None, :, None]
        coil_count = coil_maps.shape[1] if batched_maps else coil_maps.shape[0]
        resident_sense = (
            streaming is None
            and image.device.type == "cuda"
            and coil_count == 1
            and right_factors is None
            and left_factors is None
            # The fused lane folds the coil map into one resident pass over
            # the doubled grid; a transfer filed by parity has no such pass.
            and hasattr(kernel, "_apply_cuda_resident")
            and kernel._select_cuda_mode(image) == "resident"
        )
        if resident_sense:
            maps_batch = (
                coil_maps[:, 0]
                if batched_maps
                else coil_maps[0][None].expand(
                    image.shape[0],
                    *kernel.image_shape,
                )
            )
            factor = maps_batch[:, None].expand_as(image)
            try:
                kernel._apply_cuda_resident(
                    image,
                    right_factor=factor,
                    left_factor=factor.conj(),
                    output=result,
                )
            except RuntimeError as error:
                if kernel.cuda_mode == "resident" or not _device_is_full(error):
                    raise
                kernel._resident_refused()
            else:
                kernel._last_cuda_mode = "resident"
                continue
        fused_streaming = (
            streaming is not None and image.device.type == "cpu" and coil_count == 1
        )
        if fused_streaming:
            maps_batch = (
                coil_maps[:, 0]
                if batched_maps
                else coil_maps[0][None].expand(
                    image.shape[0],
                    *kernel.image_shape,
                )
            )
            fused_right = maps_batch[:, None].expand(
                image.shape[0],
                kernel.rank,
                *kernel.image_shape,
            )
            if right_factors is not None:
                fused_right = fused_right * right_factors[None]
            fused_left = maps_batch.conj()[:, None].expand(
                image.shape[0],
                kernel.rank,
                *kernel.image_shape,
            )
            if left_factors is not None:
                fused_left = fused_left * left_factors.conj()[None]
            transformed = kernel.apply_streamed(
                image,
                streaming,
                right_factor=fused_right,
                left_factor=fused_left,
            )
        elif staged_coils is None:
            coil_images = left * right
        else:
            coil_images = staged_coils[:, :coil_count]
            torch.mul(left, right, out=coil_images)
        if not fused_streaming:
            coil_images = coil_images.flatten(0, 1)
            if right_factors is not None:
                coil_images = coil_images * right_factors[None]
            transformed = (
                kernel.apply_streamed(coil_images, streaming)
                if streaming is not None and coil_images.device.type == "cpu"
                else kernel.apply(coil_images)
            )
        if left_factors is not None:
            transformed = (
                transformed.sum(dim=1, keepdim=True)
                if fused_streaming
                else (left_factors.conj()[None] * transformed).sum(
                    dim=1,
                    keepdim=True,
                )
            )
        transformed = transformed.unflatten(0, (image.shape[0], coil_count))
        if fused_streaming:
            result += transformed.sum(dim=1)
        else:
            result += (
                (transformed * coil_maps.conj()[None, :, None]).sum(dim=1)
                if not batched_maps
                else (transformed * coil_maps.conj()[:, :, None]).sum(dim=1)
            )
    kernel.settle_allocator()
    return result
