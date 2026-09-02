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

from ._coils import CoilKernels
from ._kernel import (
    CompactToeplitzKernel,
    PolyphaseToeplitzKernel,
    _device_is_full,
    as_torch,
)
from ._streaming import CudaStreaming


def _sense_maps(maps: Any, reference: Any, image_shape: tuple[int, ...]) -> Any:
    """Return sensitivities as something the apply can slice coil-wise.

    A normal application reads one coil at a time, so maps the caller left on
    the host are staged coil by coil rather than moved whole -- the difference
    is the whole bank against one map of it. A :class:`CoilKernels` bank is
    passed straight through: it already answers shape, rank and coil slicing
    as the dense tensor would, and materialising it here to check would be the
    whole point lost.
    """
    torch = import_module("torch")
    if maps is None:
        return torch.ones(
            (1, *image_shape),
            dtype=reference.dtype,
            device=reference.device,
        )
    if isinstance(maps, CoilKernels):
        return maps
    maps = as_torch(maps).to(reference.dtype)
    spatial_ndim = len(image_shape)
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


def _transfer_is_on_the_host(kernel: CompactToeplitzKernel) -> bool:
    """Whether the transfer this kernel applies is held on the host."""
    components = getattr(kernel, "components", None)
    parts = [part for _, part in components] if components else [kernel]
    return all(part.values.device.type == "cpu" for part in parts)


def _default_streaming(kernel: CompactToeplitzKernel, image: Any) -> Any:
    """Stream a host-held transfer onto a device rather than move it whole.

    A build leaves the transfer on the host, and pulling it across for an
    application puts the largest thing in the reconstruction on the card for
    the length of the solve. Streaming it is what the package is for, so it is
    what happens unless the caller has already put the transfer on the device,
    which is a decision to keep it there.
    """
    if image.device.type != "cuda" or not _transfer_is_on_the_host(kernel):
        return None
    torch = import_module("torch")
    count = torch.cuda.device_count()
    if count <= 1:
        return CudaStreaming(device=str(image.device))
    # Every visible card, with the image's own first: that one owns the answer
    # the others are summed into, so it is the one that never has to move it.
    index = image.device.index or 0
    order = [index, *(other for other in range(count) if other != index)]
    return CudaStreaming(
        device=f"cuda:{index}",
        devices=tuple(f"cuda:{other}" for other in order),
    )


def _streamed_extent(kernel: CompactToeplitzKernel) -> tuple[int, tuple[int, ...]]:
    """Return the support and grid a streamed application works over.

    A kernel filed by parity applies one component at a time over identically
    shaped images, so the largest component's support is what its buffers have
    to hold.
    """
    components = getattr(kernel, "components", None)
    if not components:
        return kernel.n_locations, kernel.spatial_shape
    locations = max(part.n_locations for _, part in components)
    return locations, components[0][1].spatial_shape


def _streamed_plan(
    kernel: CompactToeplitzKernel,
    image: Any,
    streaming: Any,
    n_coils: int,
) -> tuple[int, int]:
    """Choose how many coils share a walk of the transfer, and how many maps stay.

    Reading the transfer once per coil sends it across as many times as there
    are coils, and it is the same transfer every time. Coils grouped into the
    batch the fused multiply already indexes share one reading of it, at the
    cost of holding a spectrum per coil for the length of the walk. That
    spectrum is the only part of the apply that grows with the group, so what
    is free after everything else divided by one of them is how many coils fit
    -- one on a card with no room, all of them on a card with plenty.

    Whatever is free after that decides how many expanded maps are kept. A map
    is the same for every component of a transfer filed by parity, so one kept
    is one not expanded seven more times; a volume each is what that costs, and
    keeping none is slower rather than wrong.
    """
    torch = import_module("torch")
    device = streaming.torch_device
    locations, spatial_shape = _streamed_extent(kernel)
    element = image.element_size()
    per_coil = image.shape[0] * kernel.rank * locations * element
    if per_coil <= 0:
        return 1, 0
    volume = 1
    for size in spatial_shape:
        volume *= size
    image_volume = 1
    for size in kernel.image_shape:
        image_volume *= size
    packed_rows = kernel.rank * (kernel.rank + 1) // 2
    fixed = (
        # The padded volume and the transform's own workspace behind it.
        2 * image.shape[0] * volume * element
        # The answer, and the one coil map that is resident beside it.
        + (image.shape[0] * kernel.rank + 1) * image_volume * element
        # Support indices, kept as int32 and again as the int64 the scatter
        # insists on.
        + locations * 12
        # One transfer chunk per stream, on the device and staged on the host.
        + streaming.streams * streaming.transfer_chunk_size * packed_rows * 4
    )
    free, _ = torch.cuda.mem_get_info(device)
    # Torch does not return a freed block to the driver, so what its allocator
    # is holding but not using is free to this apply even though the driver
    # counts it as taken.
    reusable = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    headroom = int((free + reusable) * streaming.max_device_fraction) - fixed
    group = max(1, min(n_coils, headroom // per_coil))
    left = headroom - group * per_coil
    cached = max(0, min(n_coils, left // (image_volume * element)))
    return group, cached


def _apply_sense_streamed(
    kernel: CompactToeplitzKernel,
    image: Any,
    maps: Any,
    streaming: Any,
    n_coils: int,
) -> Any:
    """Stream a SENSE application, halving the coil group if it will not fit.

    The group is chosen from what the device reports free, which is a estimate
    of what an allocation will actually find. Where it is over, the answer is
    the same at a smaller group -- only more walks of the transfer.
    """
    group, cached = _streamed_plan(kernel, image, streaming, n_coils)
    while True:
        try:
            return kernel._apply_sense_streamed(
                image,
                maps,
                streaming,
                coil_group=group,
                map_cache=cached,
            )
        except RuntimeError as error:
            if group == 1 or not _device_is_full(error):
                raise
            group = max(1, group // 2)
            cached = cached // 2
            import_module("torch").cuda.empty_cache()


def _shares(count: int, parts: int) -> list[tuple[int, int]]:
    """Split ``count`` into ``parts`` spans that differ by at most one."""
    edges = [(index * count) // parts for index in range(parts + 1)]
    return [
        (edges[index], edges[index + 1])
        for index in range(parts)
        if edges[index] < edges[index + 1]
    ]


def _divided_across_devices(
    kernel: CompactToeplitzKernel,
    image: Any,
    maps: Any,
    streaming: Any,
    *,
    batched_maps: bool,
    n_coils: int,
) -> Any | None:
    """Divide one application between the devices, along the cheapest axis.

    Three axes will divide, and they do not cost the same.

    The batch is free: its entries are independent applications and there is
    nothing to sum at the end, so it goes first wherever there is more than one
    -- a stack of slices, which is what a two-dimensional acquisition is.

    Failing that, the parities. Each component of a transfer filed by parity
    owns its own share of the transfer, so dividing there *divides the
    transfer*: a card holds the components it was given and none of the rest.
    What it costs is one sum at the end, over a volume per coefficient.

    Coils divide too, and cost the same sum, but every card then needs the
    whole transfer rather than a share of it. So that is the last resort, for a
    transfer that has no parities to divide.

    Returns None when no axis divides, which is when the caller should go on
    and apply it here.
    """
    parts = streaming.device_count
    if image.shape[0] > 1:
        return _batches_split_across_devices(kernel, image, maps, streaming)
    components = getattr(kernel, "components", None)
    if components is not None and len(components) > 1:
        return _parities_split_across_devices(kernel, image, maps, streaming)
    if n_coils > 1 and parts > 1:
        return _coils_split_across_devices(
            kernel,
            image,
            maps,
            streaming,
            batched_maps=batched_maps,
            n_coils=n_coils,
        )
    return None


def _across_devices(streaming: Any, shares: list[tuple[int, int]], share: Any) -> list:
    """Run one share per device, at the same time."""
    devices = streaming.torch_devices[: len(shares)]
    with ThreadPoolExecutor(max_workers=len(devices)) as workers:
        return list(workers.map(share, range(len(devices))))


def _batches_split_across_devices(
    kernel: CompactToeplitzKernel,
    image: Any,
    maps: Any,
    streaming: Any,
) -> Any:
    """Apply to a stack of independent images, a share of them per device.

    Nothing is summed: each entry of the batch is its own application, so the
    shares are answered where they are computed and gathered back in order.
    """
    torch = import_module("torch")
    parts = min(streaming.device_count, image.shape[0])
    shares = _shares(image.shape[0], parts)
    devices = streaming.torch_devices[:parts]

    def share(position: int) -> Any:
        device = devices[position]
        start, stop = shares[position]
        return _apply_sense(
            kernel.for_device(device),
            image[start:stop].to(device),
            maps.to(device),
            streaming=streaming.for_device(device),
        )

    parts_done = _across_devices(streaming, shares, share)
    return torch.cat([part.to(image.device) for part in parts_done])


def _parities_split_across_devices(
    kernel: Any,
    image: Any,
    maps: Any,
    streaming: Any,
) -> Any:
    """Sum an application over the parities, a share of them per device.

    A share is a transfer in its own right over the same image grid, so each
    card is given one and holds only the components in it. The normalisation a
    parity-filed transfer carries is over the whole doubled grid rather than
    over the components present, so each share answers on the same scale and
    the shares add.
    """
    parts = min(streaming.device_count, len(kernel.components))
    shares = _shares(len(kernel.components), parts)
    devices = streaming.torch_devices[:parts]

    def share(position: int) -> Any:
        device = devices[position]
        start, stop = shares[position]
        held = PolyphaseToeplitzKernel(
            [
                (parity, part.for_device(device))
                for parity, part in kernel.components[start:stop]
            ],
            kernel.image_shape,
            kernel.rank,
            truncation_bound=kernel.truncation_bound,
        )
        return _apply_sense(
            held,
            image.to(device),
            maps.to(device),
            streaming=streaming.for_device(device),
        )

    parts_done = _across_devices(streaming, shares, share)
    total = parts_done[0].to(image.device)
    for part in parts_done[1:]:
        total.add_(part.to(image.device))
    return total


def _apply_sense(
    kernel: CompactToeplitzKernel,
    image: Any,
    maps: Any | None = None,
    *,
    right_factors: Any | None = None,
    left_factors: Any | None = None,
    coil_batch_size: int = 1,
    streaming: Any | None = None,
) -> Any:
    """Apply a transfer through coil sensitivities.

    Parameters
    ----------
    kernel
        The normal operator, from :func:`~mrtoeplitz.scalar_kernel` or a
        subspace builder.
    image
        ``(batch, rank, *image_shape)``, complex.
    maps
        Sensitivities as ``(coils, *image_shape)``, with a leading batch axis,
        or as a :class:`~mrtoeplitz.CoilKernels` bank. ``None`` applies the
        normal without coils.
    right_factors, left_factors
        Spatial factors folded into the pass either side of the transfer.
    coil_batch_size
        Coils per pass. One keeps the least in flight.
    streaming
        How a host-held transfer reaches the device. A transfer on the host is
        streamed whether or not this is given; giving it says how.

    Returns
    -------
    array
        ``sum_c conj(m_c) N(m_c x)``, shaped like ``image``.
    """
    torch = import_module("torch")
    if streaming is None:
        # A kernel built for a policy applies under it; one built without still
        # streams a transfer that is on the host, which is where a build leaves
        # it.
        streaming = getattr(kernel, "streaming", None) or _default_streaming(
            kernel, image
        )
    if streaming is not None and streaming.device_count > 1:
        # Coils are independent until their final SENSE reduction.  Group at
        # least one coil per device so even a single-image reconstruction can
        # fan its Toeplitz work across a multi-GPU recon host.
        coil_batch_size = max(coil_batch_size, streaming.device_count)
    maps = _sense_maps(maps, image, kernel.image_shape)
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
        and left_factors is None
        and right_factors is None
    ):
        divided = _divided_across_devices(
            kernel,
            image,
            maps,
            streaming,
            batched_maps=batched_maps,
            n_coils=n_coils,
        )
        if divided is not None:
            return divided
    if (
        streaming is not None
        and not batched_maps
        and right_factors is None
        and left_factors is None
        and hasattr(kernel, "_apply_sense_streamed")
    ):
        return _apply_sense_streamed(kernel, image, maps, streaming, n_coils)
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
                else kernel._apply(coil_images)
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


_SENSE_FUNCTION: Any = None


def _sense_function() -> Any:
    """Return the autograd Function the SENSE normal is applied through."""
    global _SENSE_FUNCTION
    if _SENSE_FUNCTION is not None:
        return _SENSE_FUNCTION
    torch = import_module("torch")

    class _SenseApply(torch.autograd.Function):
        """The SENSE normal, differentiable in the image.

        ``sum_c conj(m_c) N (m_c x)`` is Hermitian whenever ``N`` is, so like
        the bare transfer it is its own adjoint and the backward pass is one
        more application. Nothing from the forward pass is kept, which is what
        makes it usable inside an unrolled network at the sizes this package
        exists for.
        """

        @staticmethod
        def forward(image: Any, kernel: Any, maps: Any, settings: dict) -> Any:
            with torch.no_grad():
                return _apply_sense(kernel, image, maps, **settings)

        @staticmethod
        def setup_context(ctx: Any, inputs: tuple[Any, ...], _output: Any) -> None:
            ctx.held = inputs[1:]

        @staticmethod
        def backward(ctx: Any, grad_output: Any) -> tuple[Any, None, None, None]:
            kernel, maps, settings = ctx.held
            with torch.no_grad():
                grad = _apply_sense(kernel, grad_output.contiguous(), maps, **settings)
            return grad, None, None, None

    _SENSE_FUNCTION = _SenseApply
    return _SENSE_FUNCTION


def apply_sense(
    kernel: CompactToeplitzKernel,
    image: Any,
    maps: Any | None = None,
    *,
    right_factors: Any | None = None,
    left_factors: Any | None = None,
    coil_batch_size: int = 1,
    streaming: Any | None = None,
) -> Any:
    """Apply a transfer through coil sensitivities, with a gradient.

    Parameters
    ----------
    kernel
        The normal operator, from :func:`~mrtoeplitz.scalar_kernel` or a
        subspace builder.
    image
        ``(batch, rank, *image_shape)``, complex.
    maps
        Sensitivities as ``(coils, *image_shape)``, with a leading batch axis,
        or as a :class:`~mrtoeplitz.CoilKernels` bank. ``None`` applies the
        normal without coils.
    right_factors, left_factors
        Spatial factors folded into the pass either side of the transfer.
        Giving them makes the operator something other than its own adjoint,
        so the result is not differentiable.
    coil_batch_size
        Coils per pass. One keeps the least in flight.
    streaming
        How a host-held transfer reaches the device. A transfer on the host is
        streamed whether or not this is given; giving it says how.

    Returns
    -------
    array
        ``sum_c conj(m_c) N(m_c x)``, shaped like ``image``. Differentiable in
        ``image`` when no factors are given.

    Raises
    ------
    ValueError
        If a gradient is asked for through explicit factors.
    """
    settings = {
        "right_factors": right_factors,
        "left_factors": left_factors,
        "coil_batch_size": coil_batch_size,
        "streaming": streaming,
    }
    if right_factors is None and left_factors is None:
        return _sense_function().apply(image, kernel, maps, settings)
    if getattr(image, "requires_grad", False):
        raise ValueError(
            "explicit factors make the SENSE normal something other than its "
            "own adjoint, so it is not differentiable; apply them yourself, or "
            "drop them to differentiate"
        )
    return _apply_sense(kernel, image, maps, **settings)
