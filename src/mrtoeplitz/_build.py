"""Building a Toeplitz transfer by gridding its point spread function.

The transfer a normal operator multiplies by is the transform of the adjoint
of the sample weights, taken on a grid twice the image in every dimension.
That is how BART's ``compute_psf_int`` and MRFingerprintingRecon.jl's
``calculate_kernel_noncartesian`` build theirs, and it is not a choice: the
*exact* transfer -- the transform of the analytic point spread function -- is
dense, with Dirichlet tails putting 84% of peak in the corners of the cube
even for a strictly ball-supported trajectory, and cannot be truncated. The
gridded transfer holds weight only where the scan reached plus the
interpolation rim, which is what makes it compressible.

Where the transform runs follows the trajectory: a trajectory on the host is
gridded by FINUFFT, one on a device by CUFINUFFT. A kernel moves afterwards
with :meth:`~mrtoeplitz.CompactToeplitzKernel.to`.
"""

from __future__ import annotations

__all__ = [
    "cartesian_subspace_kernel",
    "scalar_kernel",
    "subspace_kernel",
    "toeplitz_options",
]

from collections.abc import Sequence
from contextlib import nullcontext, suppress
from importlib import import_module
from itertools import product
from math import isqrt, prod
from typing import Any

from ._kernel import (
    CompactToeplitzKernel,
    PolyphaseToeplitzKernel,
    as_torch,
    decoded_positions,
    polyphase_components,
)
from ._options import _support_locations, _toeplitz_options
from ._psf import (
    _UPSAMPLING,
    gridding_streams,
    psf_plan,
    within_psf_plans,
)

#: How much of a grid a scan walks at once when it only needs a reduction of
#: it. Small enough that the temporary is nothing beside the grid itself.
_SCAN_CHUNK = 1 << 24

#: Public name for the options validator.
toeplitz_options = _toeplitz_options


def _flat_samples(trajectory: Any, image_ndim: int) -> Any:
    """Read one trajectory as a flat ``(samples, axes)`` bank."""
    trajectory = as_torch(trajectory)
    if trajectory.shape[-1] != image_ndim:
        raise ValueError(
            f"trajectory's last axis is the {trajectory.shape[-1]} spatial "
            f"axes it names, which must match the {image_ndim}-dimensional "
            f"image grid"
        )
    if trajectory.ndim not in {2, 3}:
        raise ValueError(
            f"trajectory must be (points, axes) or (shots, points, axes), got "
            f"shape {tuple(trajectory.shape)}; a trajectory with a frames axis "
            f"belongs to subspace_kernel"
        )
    return trajectory.reshape(-1, image_ndim)


def _flat_density(density: Any | None, n_samples: int) -> Any | None:
    """Read a density as one weight per sample, or nothing."""
    if density is None:
        return None
    weights = as_torch(density).reshape(-1)
    if weights.numel() != n_samples:
        raise ValueError(
            f"density has {weights.numel()} weights for {n_samples} samples"
        )
    return weights


def _compute_toeplitz_transfer(
    samples: Any,
    image_shape: tuple[int, ...],
    weights: Any | None = None,
    tolerance: float | None = None,
) -> Any:
    """Return the transfer a Toeplitz normal operator multiplies by.

    The point-spread function is the adjoint of the sample weights taken on a
    grid twice the image in every dimension -- ones for a plain normal, the
    density for a compensated one, a basis product for a subspace frame or an
    off-resonance segment -- and the transfer is its transform.

    Gridding is what puts the weight where the trajectory is. The adjoint
    interpolates each sample onto the doubled grid with the transform's own
    kernel, so the transfer holds weight where the scan reached and in the rim
    that interpolation spreads into, and nowhere else. That is the same
    operator the forward NUFFT applies, so the normal is the Gram of the
    transform actually being inverted.
    """
    torch = import_module("torch")
    samples = as_torch(samples)
    spatial_shape = tuple(2 * size for size in image_shape)
    plan = psf_plan(spatial_shape, samples, tolerance)

    if weights is None:
        values = torch.ones(
            plan.n_samples,
            dtype=torch.complex64,
            device=samples.device,
        )
    else:
        values = as_torch(weights).reshape(-1).to(torch.complex64)

    # Backends differ on whether they take a bare sample vector, so the
    # batch and coil axes are stated and dropped again.
    psf = plan.grid(values).reshape(spatial_shape)
    axes = tuple(range(len(spatial_shape)))
    # ``adj_op`` answers a centred image and divides by the doubled grid's own
    # normalization, while the normal operator this stands in for carries the
    # image grid's twice -- once in the forward and once in the adjoint.
    # The transform is raw, and the convolution runs on the doubled grid,
    # so that grid's size is the whole normalization.
    scale = 1.0 / prod(spatial_shape)
    return torch.fft.fftn(torch.fft.ifftshift(psf, dim=axes), dim=axes) * scale


def _complement_of(indices: Any, count: int, device: Any) -> Any | None:
    """Return a mask over the locations ``indices`` leaves out, or None."""
    torch = import_module("torch")
    if indices.numel() >= count:
        return None
    left_out = torch.ones(count, dtype=torch.bool, device=device)
    left_out[indices.to(device=device, dtype=torch.int64)] = False
    return left_out


def _largest_left_out(stored: Any, left_out: Any | None) -> float:
    """Return the largest magnitude of ``stored`` at the locations left out.

    Walked in pieces. Taking the magnitude of the whole doubled grid at once
    is a second grid's worth of float, and selecting through the mask is a
    third; at the sizes this runs at that is what tips the build off the card.
    """
    if left_out is None:
        return 0.0
    flat = stored.reshape(-1)
    largest = 0.0
    for start in range(0, flat.numel(), _SCAN_CHUNK):
        stop = min(start + _SCAN_CHUNK, flat.numel())
        piece = flat[start:stop].abs()
        # Zero what is kept, so the maximum is over what is not.
        piece.mul_(left_out[start:stop])
        largest = max(largest, float(piece.max()))
    return largest


def _polyphase_wanted(
    options: dict[str, Any],
    spatial_shape: tuple[int, ...],
    rank: int,
    reference: Any,
) -> bool:
    """Whether to file the transfer by parity rather than keep it doubled.

    Asked for outright, or judged by what the device can hold: the doubled
    grid's two banks are the largest thing an application allocates, and once
    they no longer fit the budget the solve drops to a lane several times
    slower. Filing by parity puts every bank on the image grid instead, which
    is what keeps it resident.
    """
    setting = options.get("polyphase", "auto")
    if setting != "auto":
        return bool(setting)
    torch = import_module("torch")
    if not torch.cuda.is_available():
        return False
    banks = 2 * rank * prod(spatial_shape) * 8
    device = getattr(reference, "device", None)
    index = device if getattr(device, "type", None) == "cuda" else None
    total = torch.cuda.get_device_properties(index).total_memory
    return banks > options["cuda_max_device_fraction"] * total


def _make_kernel(
    values: Any,
    indices: Any,
    spatial_shape: tuple[int, ...],
    rank: int,
    *,
    image_shape: tuple[int, ...],
    options: dict[str, Any],
    truncation_bound: float = 0.0,
    streaming: Any | None = None,
) -> Any:
    """Build the kernel in the layout the options ask for.

    A transfer on the doubled grid is filed by the parity of its coordinates
    unless asked otherwise, which puts every transform the application makes
    on the image grid. One that is already the image's size -- a Cartesian
    subspace transfer, which needs no padding -- has no parities to split.
    """
    image_shape = tuple(int(size) for size in image_shape)
    spatial_shape = tuple(int(size) for size in spatial_shape)
    settings: dict[str, Any] = {
        "image_shape": image_shape,
        "chunk_size": options["chunk_size"],
        "cuda_mode": options["cuda_mode"],
        "cuda_max_device_fraction": options["cuda_max_device_fraction"],
        "cuda_transfer_precision": options["cuda_transfer_precision"],
    }
    doubled = tuple(2 * size for size in image_shape) == spatial_shape
    if not _polyphase_wanted(options, spatial_shape, rank, values) or not doubled:
        whole = CompactToeplitzKernel(
            values,
            indices,
            spatial_shape,
            rank,
            truncation_bound=truncation_bound,
            **settings,
        )
        whole.streaming = streaming
        return whole
    components = [
        (
            parity,
            CompactToeplitzKernel(part, where, image_shape, rank, **settings),
        )
        for parity, part, where in polyphase_components(
            values, indices, spatial_shape, image_shape
        )
    ]
    filed = PolyphaseToeplitzKernel(
        components,
        image_shape,
        rank,
        truncation_bound=truncation_bound,
    )
    filed.streaming = streaming
    return filed


def _selected_transfer(
    transfer: Any,
    indices: Any,
    *,
    streaming: Any | None,
) -> Any:
    """Select retained locations and optionally move them to host storage."""
    torch = import_module("torch")
    transfer = as_torch(transfer).flatten()
    selected = torch.index_select(
        transfer,
        0,
        indices.to(transfer.device, dtype=torch.int64),
    )
    return selected.to("cpu") if streaming is not None else selected


@within_psf_plans
def scalar_kernel(
    trajectory: Any,
    image_shape: tuple[int, ...],
    *,
    density: Any | None = None,
    options: dict[str, Any] | None = None,
    streaming: Any | None = None,
) -> CompactToeplitzKernel:
    """Build the rank-one normal operator for one trajectory.

    Parameters
    ----------
    trajectory
        ``(shots, points, axes)``, or ``(points, axes)`` for a single shot.
        ``axes`` is the image's dimensionality. Samples are in normalized
        k-space: ``-0.5`` is grid location ``-kN/2`` of a grid of size
        ``kN`` and ``0.5`` is ``+kN/2``, so the same numbers describe
        the image grid and the doubled one the transfer lives on.
    image_shape
        The image grid. The transfer is built on twice this in every dimension.
    density
        Sample weights, broadcastable to the trajectory without its axes.
        ``None`` weights every sample equally. Density inside the normal is
        the intended acceleration, not a defect: an adjoint applies it once,
        so the Gram does too.
    options
        As :func:`toeplitz_options`.
    streaming
        Where a transfer too large for the device is staged.

    Returns
    -------
    CompactToeplitzKernel
        The normal operator, taking ``(batch, 1, *image_shape)``.

    Raises
    ------
    ValueError
        If the trajectory's rank is not two or three, its last axis does not
        match the image's dimensionality, or the density does not fit the
        samples.

    Examples
    --------
    >>> import numpy as np
    >>> import mrtoeplitz as mt
    >>> angles = np.linspace(0, np.pi, 16, endpoint=False)
    >>> radius = np.linspace(-0.5, 0.5, 32, endpoint=False)
    >>> spokes = np.stack(
    ...     [np.outer(np.cos(angles), radius), np.outer(np.sin(angles), radius)],
    ...     axis=-1,
    ... ).astype(np.float32)
    >>> kernel = mt.scalar_kernel(spokes, (32, 32))
    >>> kernel.rank, kernel.image_shape, kernel.spatial_shape
    (1, (32, 32), (64, 64))
    """
    options = _toeplitz_options() if options is None else options
    image_shape = tuple(int(size) for size in image_shape)
    samples = _flat_samples(trajectory, len(image_shape))
    weights = _flat_density(density, samples.shape[0])
    spatial_shape = tuple(2 * size for size in image_shape)

    transfer = as_torch(
        _compute_toeplitz_transfer(
            samples, image_shape, weights, options["gridding_tolerance"]
        )
    ).flatten()
    indices = _support_locations(
        samples,
        spatial_shape,
        "cpu" if streaming is not None else transfer.device,
        options["compress"],
    )
    left_out = _complement_of(indices, transfer.numel(), transfer.device)
    values = _selected_transfer(transfer, indices, streaming=streaming).real[None]
    return _make_kernel(
        values,
        indices,
        spatial_shape,
        1,
        image_shape=image_shape,
        options=options,
        streaming=streaming,
        truncation_bound=_largest_left_out(transfer.real, left_out),
    )


def _centring_signs(indices: Any, spatial_shape: tuple[int, ...]) -> Any:
    """Return the sign that centres a transfer, over the locations it keeps.

    Shifting a point-spread function by half the grid before transforming it
    multiplies every output by ``(-1)`` raised to the sum of its coordinates,
    so the shift never has to be performed and no copy of the doubled grid is
    made to hold it.
    """
    torch = import_module("torch")
    flat = as_torch(indices)
    # One byte per location: it multiplies a complex transfer and a real one
    # alike, and as float it is four times the size of what it says. Walked in
    # pieces, because the coordinates it is read from are int64 and there is
    # one per axis -- held whole that is several times the sign itself.
    signs = torch.empty(flat.numel(), dtype=torch.int8, device=flat.device)
    for start in range(0, flat.numel(), _SCAN_CHUNK):
        stop = min(start + _SCAN_CHUNK, flat.numel())
        piece = flat[start:stop].to(torch.int64)
        parity = torch.zeros_like(piece)
        stride = 1
        for size in reversed(spatial_shape):
            parity.add_(piece.div(stride, rounding_mode="floor").remainder_(size))
            stride *= size
        parity.remainder_(2)
        signs[start:stop] = torch.where(parity == 0, 1, -1).to(torch.int8)
    return signs


def _subspace_pair_transfers(
    blocks: Sequence[tuple[Any, Any, Any]],
    image_shape: tuple[int, ...],
    samples: Any,
    counts: Sequence[int],
    indices: Any,
    *,
    keep_complex: bool = True,
    tolerance: float | None = None,
) -> tuple[Any, float]:
    """Grid one transfer per upper-triangular basis pair, over every sample.

    A pair's transfer is the adjoint of one weight per sample -- the frame's
    basis product, times whatever density the acquisition carries -- so the
    whole dynamic acquisition grids in a single pass and the count of NUFFTs
    is the size of the basis, not the length of the scan.

    Each is cut to ``indices`` as it is gridded and put down on the host in the
    form it is kept in, so a build holds one row of the device rather than the
    whole packed set twice over -- once complex and once made real. Returns
    the packed rows and the largest value that fell outside ``indices``.
    """
    torch = import_module("torch")
    spatial_shape = tuple(2 * size for size in image_shape)
    # One weight per sample, assembled in a single pass: a dynamic acquisition
    # has as many blocks as it has frames, and touching each of them per basis
    # pair is thousands of launches for one vector.
    weights = None
    if any(block[1] is not None for block in blocks):
        pieces = []
        for (_, density, _), count in zip(blocks, counts, strict=True):
            if density is None:
                pieces.append(torch.ones(count, device=samples.device))
                continue
            piece = as_torch(density).reshape(-1).to(samples.device)
            if piece.numel() != count:
                raise ValueError("density and samples must have the same length")
            pieces.append(piece)
        weights = torch.cat(pieces).to(torch.complex64)
    repeats = torch.tensor(counts, device=samples.device)
    coefficients = torch.stack([block[2] for block in blocks], dim=1)

    n_pairs = int(blocks[0][2].numel())
    n_locations = int(as_torch(indices).numel())
    row_dtype = torch.complex64 if keep_complex else torch.float32
    packed = torch.empty((n_pairs, n_locations), dtype=row_dtype, device="cpu")
    # Where the rows go is settled before anything is planned, because whether
    # the gridding gets a stream of its own depends on there being something to
    # overlap it with.
    ring = _staging_ring(n_locations, row_dtype, samples.device.type == "cuda")
    compute, staging = (
        gridding_streams(samples.device) if ring is not None else (None, None)
    )
    plan = psf_plan(spatial_shape, samples, tolerance, streamed=ring is not None)
    axes = tuple(range(len(spatial_shape)))
    # The rows are assembled where they are gridded, so the sign belongs there
    # too rather than wherever the caller left the locations.
    signs = _centring_signs(indices, spatial_shape).to(samples.device)
    # index_select takes int32, and the support is held as int32: promoting it
    # would double the largest index this build keeps.
    selection = as_torch(indices).to(device=samples.device, dtype=torch.int32)
    # The transform is raw, and the convolution runs on the doubled grid,
    # so that grid's size is the whole normalization.
    scale = 1.0 / prod(spatial_shape)
    left_out = _complement_of(indices, prod(spatial_shape), samples.device)
    dropped = 0.0
    pending: list[tuple[Any, int] | None] = [None, None]

    coefficients = coefficients.to(device=samples.device, dtype=torch.complex64)
    # A pair's transfer leaves for the host while the next one is gridded. The
    # plan issues its own work to the compute stream, so the build runs there
    # too and the staging stream is the only thing that has to wait.
    with _on_stream(compute):
        for pair in range(n_pairs):
            values = torch.repeat_interleave(coefficients[pair], repeats)
            if weights is not None:
                values.mul_(weights)
            values_view = values.reshape(1, 1, -1)
            del values
            psf = plan.grid(values_view).reshape(spatial_shape)
            # Transformed in place, and the centring folded into a sign on the
            # locations kept: shifting the point-spread function by half the
            # grid is the same as alternating the sign of what comes out, and a
            # copy of the doubled grid is the largest thing a build holds after
            # the plan.
            torch.fft.fftn(psf, dim=axes, out=psf)
            flat = psf.reshape(-1)
            if left_out is not None:
                stored = flat.real if not keep_complex else flat
                dropped = max(dropped, _largest_left_out(stored, left_out))
            selected = torch.index_select(flat, 0, selection)
            del psf, flat
            # In place: each of these as a fresh tensor is another copy of the
            # support, and there are three of them.
            selected.mul_(signs).mul_(scale)
            row = selected.real if not keep_complex else selected
            if ring is None:
                packed[pair].copy_(row)
                del row, selected
                continue
            slot = pair % len(ring)
            # This buffer may still hold an earlier pair. Landing it is a host
            # copy, and it runs while the device grids the next one.
            _land(packed, ring, pending, slot)
            staging.wait_stream(compute)
            with torch.cuda.stream(staging):
                ring[slot].copy_(row, non_blocking=True)
                arrived = torch.cuda.Event()
                arrived.record(staging)
            # The allocator must not hand this block to the next pair while the
            # copy off it is still in flight.
            selected.record_stream(staging)
            pending[slot] = (arrived, pair)
            del row, selected
        for slot in range(0 if ring is None else len(ring)):
            _land(packed, ring, pending, slot)
    if compute is not None:
        torch.cuda.current_stream(samples.device).wait_stream(compute)
    return packed, dropped


def _on_stream(stream: Any) -> Any:
    """Run on ``stream``, ordered behind whatever the caller was on."""
    torch = import_module("torch")
    if stream is None:
        return nullcontext()
    stream.wait_stream(torch.cuda.current_stream(stream.device))
    return torch.cuda.stream(stream)


def _staging_ring(n_locations: int, dtype: Any, staged: bool) -> list[Any] | None:
    """Return pinned buffers a finished row leaves on, or None to copy directly.

    Only pinned memory takes an asynchronous copy; out of pageable memory the
    driver stages it through a buffer of its own and the call blocks, which is
    the overlap lost. Pinning the whole transfer would buy that overlap and cost
    seconds to page-lock -- more than the copying it saves -- so what is pinned
    is two rows, and each is landed in the transfer while the device grids the
    next pair.
    """
    torch = import_module("torch")
    if not staged:
        return None
    wanted = 2 * n_locations * torch.empty(0, dtype=dtype).element_size()
    try:
        import psutil

        if wanted > 0.25 * psutil.virtual_memory().available:
            return None
    except Exception:
        return None
    with suppress(RuntimeError):
        return [
            torch.empty(n_locations, dtype=dtype, pin_memory=True) for _ in range(2)
        ]
    return None


def _land(packed: Any, ring: list[Any], pending: list[Any], slot: int) -> None:
    """Copy whatever a staging buffer holds into the row it belongs to."""
    held = pending[slot]
    if held is None:
        return
    arrived, pair = held
    arrived.synchronize()
    packed[pair].copy_(ring[slot])
    pending[slot] = None


def _decomposed_pair_transfers(
    blocks: Sequence[tuple[Any, Any, Any]],
    image_shape: tuple[int, ...],
    samples: Any,
    counts: Sequence[int],
    shared: Any,
    *,
    keep_complex: bool = True,
    tolerance: float | None = None,
) -> tuple[list[Any], float]:
    """Grid every parity component of the transfer onto the image grid.

    The doubled-grid transfer read at the locations whose coordinates are
    congruent to a parity is itself a transfer over the image grid: the point
    spread function folded onto that grid, modulated by the half cell the
    parity stands for. Both are weights on the samples, so each component is a
    gridding onto the image grid and the doubled grid is never made -- which is
    what the build costs most, and what stops it fitting at higher resolutions.

    It is the same operator the doubled build produces, to what the gridding
    resolves. It is also eight times the spreading, since every component
    spreads every sample, so it is worth taking only when the doubled grid is
    the binding constraint.

    Returns the components in parity order and the largest value left out.
    """
    torch = import_module("torch")
    ndim = len(image_shape)
    device = samples.device
    doubled = tuple(2 * size for size in image_shape)
    n_pairs = int(blocks[0][2].numel())
    row_dtype = torch.complex64 if keep_complex else torch.float32

    weights = None
    if any(block[1] is not None for block in blocks):
        pieces = []
        for (_, density, _), count in zip(blocks, counts, strict=True):
            if density is None:
                pieces.append(torch.ones(count, device=device))
                continue
            piece = as_torch(density).reshape(-1).to(device)
            if piece.numel() != count:
                raise ValueError("density and samples must have the same length")
            pieces.append(piece)
        weights = torch.cat(pieces).to(torch.complex64)
    repeats = torch.tensor(counts, device=device)
    coefficients = torch.stack([block[2] for block in blocks], dim=1).to(
        device=device, dtype=torch.complex64
    )
    # The convolution runs on the doubled grid, so that grid's size is the
    # whole normalization, whichever grid the components were made on.
    scale = 1.0 / prod(doubled)
    lookup = shared.to(device=device, dtype=torch.int32)
    dropped = 0.0
    components = []

    # One buffer for the shifted coordinates and one mask for what falls
    # outside the support, both reused by every parity.
    shifted = samples.clone()
    outside = None
    if int(prod(image_shape)) > lookup.numel():
        outside = torch.ones(prod(image_shape), dtype=torch.bool, device=device)
        outside[lookup.to(torch.int64)] = False

    for parity in product((0, 1), repeat=ndim):
        # A parity is a shift of half a doubled-grid cell, and the fold onto
        # the image grid is what the transfer at that shift sums.
        factor = torch.ones(samples.shape[0], dtype=torch.complex64, device=device)
        for axis, bit in enumerate(parity):
            size = image_shape[axis]
            column = samples[:, axis]
            shifted[:, axis] = column - bit / (2 * size)
            fold = torch.exp(-2j * torch.pi * size * column) + float((-1) ** bit)
            # A gridding answers centred modes and the fold wants them from
            # zero, which is one more phase on the sample.
            offset = torch.exp(1j * torch.pi * size * shifted[:, axis])
            factor.mul_(fold.to(torch.complex64)).mul_(offset.to(torch.complex64))
        sign = float((-1) ** sum(parity))

        rows = torch.empty((n_pairs, lookup.numel()), dtype=row_dtype, device="cpu")
        # A pair's component leaves for the host while the next is gridded, on
        # the same pair of streams the doubled build stages through.
        ring = _staging_ring(lookup.numel(), row_dtype, device.type == "cuda")
        compute, staging = (
            gridding_streams(device) if ring is not None else (None, None)
        )
        pending: list[tuple[Any, int] | None] = [None, None]
        # The parities differ only in where the samples sit, so one plan serves
        # them all: setpts retargets it for a fraction of what making it costs,
        # and what making it costs is half a gigabyte that destroying it does
        # not give back.
        plan = psf_plan(image_shape, shifted, tolerance, streamed=ring is not None)
        with _on_stream(compute):
            for pair in range(n_pairs):
                values = torch.repeat_interleave(coefficients[pair], repeats)
                if weights is not None:
                    values.mul_(weights)
                values.mul_(factor)
                psf = plan.grid(values.reshape(1, 1, -1)).reshape(image_shape)
                del values
                torch.fft.fftn(psf, dim=tuple(range(ndim)), out=psf)
                flat = psf.reshape(-1)
                if outside is not None:
                    dropped = max(dropped, _largest_outside(flat, outside))
                selected = torch.index_select(flat, 0, lookup)
                selected.mul_(sign * scale)
                row = selected.real if not keep_complex else selected
                if ring is None:
                    rows[pair].copy_(row)
                    del psf, flat, row, selected
                    continue
                slot = pair % len(ring)
                _land(rows, ring, pending, slot)
                staging.wait_stream(compute)
                with torch.cuda.stream(staging):
                    ring[slot].copy_(row, non_blocking=True)
                    arrived = torch.cuda.Event()
                    arrived.record(staging)
                selected.record_stream(staging)
                pending[slot] = (arrived, pair)
                del psf, flat, row, selected
            for slot in range(0 if ring is None else len(ring)):
                _land(rows, ring, pending, slot)
        if compute is not None:
            torch.cuda.current_stream(device).wait_stream(compute)
        components.append(rows)
        del factor
    return components, dropped


def _largest_outside(flat: Any, outside: Any) -> float:
    """Return the largest magnitude of ``flat`` where ``outside`` is set.

    Marked rather than gathered, and walked in pieces: the grid this reduces
    over is the one the build exists to avoid holding copies of.
    """
    largest = 0.0
    for start in range(0, flat.numel(), _SCAN_CHUNK):
        stop = min(start + _SCAN_CHUNK, flat.numel())
        piece = flat[start:stop].abs()
        piece.mul_(outside[start:stop])
        largest = max(largest, float(piece.max()))
    return largest


def _decomposed_wanted(
    options: dict[str, Any],
    spatial_shape: tuple[int, ...],
    reference: Any,
) -> bool:
    """Whether to grid the parities directly rather than make the doubled grid.

    Asked for outright, or judged by what the device can hold. Gridding onto
    the doubled grid needs that grid and the working one the transform spreads
    onto, and past a resolution the two no longer fit; the parities need
    neither, at eight times the spreading.
    """
    setting = options.get("decomposed_build", "auto")
    if setting != "auto":
        return bool(setting)
    torch = import_module("torch")
    device = getattr(reference, "device", None)
    if getattr(device, "type", None) != "cuda":
        return False
    # The grid the transform answers on, and the one it spreads onto.
    wanted = (1 + _UPSAMPLING ** len(spatial_shape)) * prod(spatial_shape) * 8
    total = torch.cuda.get_device_properties(device).total_memory
    return wanted > options["cuda_max_device_fraction"] * total


def _decomposed_kernel(
    blocks: Sequence[tuple[Any, Any, Any]],
    image_shape: tuple[int, ...],
    samples: Any,
    counts: Sequence[int],
    indices: Any,
    rank: int,
    *,
    options: dict[str, Any],
    keep_complex: bool,
    dtype: Any,
    streaming: Any | None,
) -> Any:
    """Build the transfer as its parity components, without the doubled grid."""
    torch = import_module("torch")
    ndim = len(image_shape)
    spatial_shape = tuple(2 * size for size in image_shape)
    position, _ = decoded_positions(indices, spatial_shape, image_shape)
    shared = torch.unique(position)
    del position
    components, dropped = _decomposed_pair_transfers(
        blocks,
        image_shape,
        samples,
        counts,
        shared,
        keep_complex=keep_complex,
        tolerance=options["gridding_tolerance"],
    )
    settings: dict[str, Any] = {
        "image_shape": image_shape,
        "chunk_size": options["chunk_size"],
        "cuda_mode": options["cuda_mode"],
        "cuda_max_device_fraction": options["cuda_max_device_fraction"],
        "cuda_transfer_precision": options["cuda_transfer_precision"],
    }
    where = shared.to(device="cpu" if streaming is not None else shared.device)
    where = where.to(torch.int32)
    parts = [
        (
            parity,
            CompactToeplitzKernel(
                rows.to(dtype) if keep_complex else rows.real.to(dtype),
                where,
                image_shape,
                rank,
                **settings,
            ),
        )
        for parity, rows in zip(product((0, 1), repeat=ndim), components, strict=True)
    ]
    filed = PolyphaseToeplitzKernel(parts, image_shape, rank, truncation_bound=dropped)
    filed.streaming = streaming
    return filed


def _subspace_kernel_from_blocks(
    blocks: Sequence[tuple[Any, Any, Any]],
    image_shape: tuple[int, ...],
    *,
    options: dict[str, Any] | None = None,
    streaming: Any | None = None,
) -> CompactToeplitzKernel:
    """Grid one transfer per basis pair, over pre-grouped trajectory blocks.

    The engine beneath :func:`subspace_kernel`, which is the entry point a
    caller wants. Each block is ``(samples, weights, coefficients)`` for one
    distinct trajectory, with ``coefficients`` the upper-triangular basis
    products summed over the frames sharing it, ordered as
    ``torch.triu_indices(rank, rank)`` gives the pairs.
    """
    torch = import_module("torch")
    options = _toeplitz_options() if options is None else options
    if not blocks:
        raise ValueError("a subspace kernel needs at least one trajectory block")

    pairs = int(as_torch(blocks[0][2]).reshape(-1).shape[0])
    rank = (isqrt(8 * pairs + 1) - 1) // 2
    if rank * (rank + 1) // 2 != pairs:
        raise ValueError(f"{pairs} coefficients do not form a packed upper triangle")

    image_shape = tuple(int(size) for size in image_shape)
    spatial_shape = tuple(2 * size for size in image_shape)
    ndim = len(image_shape)

    # The support is the union of what the trajectories reached, read off the
    # samples and needing none of their transfers -- so it is known before the
    # first one is gridded, and each can be cut as it comes.
    counts = [as_torch(block[0]).reshape(-1, ndim).shape[0] for block in blocks]
    samples = torch.cat([as_torch(block[0]).reshape(-1, ndim) for block in blocks])
    coefficients = as_torch(blocks[0][2])
    stripped = [(None, block[1], block[2]) for block in blocks]
    # The support is read where the gridding happens. A policy puts the
    # kernel's copy of it on the host afterwards; computing it there instead
    # would send it back for every basis pair.
    indices = _support_locations(
        samples,
        spatial_shape,
        samples.device,
        options["compress"],
    )
    if _decomposed_wanted(options, spatial_shape, samples):
        return _decomposed_kernel(
            stripped,
            image_shape,
            samples,
            counts,
            indices,
            rank,
            options=options,
            keep_complex=bool(coefficients.is_complex()),
            dtype=coefficients.dtype,
            streaming=streaming,
        )
    packed, dropped = _subspace_pair_transfers(
        stripped,
        image_shape,
        samples,
        counts,
        indices,
        keep_complex=bool(coefficients.is_complex()),
        tolerance=options["gridding_tolerance"],
    )
    if streaming is not None:
        indices = indices.to("cpu")
    values = (
        packed.to(coefficients.dtype)
        if coefficients.is_complex()
        else packed.real.to(coefficients.dtype)
    )
    return _make_kernel(
        values,
        indices,
        spatial_shape,
        rank,
        image_shape=image_shape,
        options=options,
        streaming=streaming,
        truncation_bound=dropped,
    )


def _basis_as_rank_by_frames(basis: Any, n_frames: int | None) -> Any:
    """Return the basis as ``(rank, frames)``, whichever way round it came.

    Which axis is which is read off the data. When the trajectory states a
    frame count, the axis matching it is the frames. When it does not, the
    longer axis is: a subspace has fewer components than the frames it
    compresses.

    A square basis satisfies both readings, and is taken as ``(frames, rank)``
    -- the form the documentation leads with -- whether or not the trajectory
    stated a count. A basis that compresses nothing is a real case: it is what
    a full temporal basis looks like before it has been cut to a rank.
    """
    torch = import_module("torch")
    basis = torch.as_tensor(basis)
    if basis.ndim != 2:
        raise ValueError(
            f"basis must be (frames, rank) or (rank, frames), got shape "
            f"{tuple(basis.shape)}"
        )
    if basis.shape[0] == basis.shape[1]:
        if n_frames is not None and basis.shape[0] != n_frames:
            raise ValueError(
                f"basis {tuple(basis.shape)} has no axis matching the "
                f"{n_frames} frames the trajectory carries"
            )
        return basis.T
    if n_frames is not None:
        if basis.shape[0] == n_frames:
            return basis.T
        if basis.shape[1] == n_frames:
            return basis
        raise ValueError(
            f"basis {tuple(basis.shape)} has no axis matching the "
            f"{n_frames} frames the trajectory carries"
        )
    return basis if basis.shape[0] < basis.shape[1] else basis.T


def _trajectory_frames(
    trajectory: Any,
    image_ndim: int,
) -> tuple[Any, int | None]:
    """Split a trajectory into per-frame samples and the frame count it states.

    ``(shots, points, axes)`` is one trajectory every frame shares and states
    no count; ``(frames, shots, points, axes)`` states one.
    """
    trajectory = as_torch(trajectory)
    if trajectory.shape[-1] != image_ndim:
        raise ValueError(
            f"trajectory's last axis is the {trajectory.shape[-1]} spatial "
            f"axes it names, which must match the {image_ndim}-dimensional "
            f"image grid"
        )
    if trajectory.ndim == 3:
        return trajectory.reshape(-1, image_ndim)[None], None
    if trajectory.ndim == 4:
        n_frames = int(trajectory.shape[0])
        return trajectory.reshape(n_frames, -1, image_ndim), n_frames
    raise ValueError(
        f"trajectory must be (shots, points, axes) or "
        f"(frames, shots, points, axes), got shape {tuple(trajectory.shape)}"
    )


def _grouped_frames(samples: Any) -> dict[bytes, list[int]]:
    """Group frames by the trajectory they were acquired on.

    Frames that share a trajectory share the transform that grids it, so
    carrying them separately multiplies the samples every basis pair is
    gridded over -- a thousand times, for a fingerprinting scan whose frames
    cycle through a handful of rotations.
    """
    import hashlib

    groups: dict[bytes, list[int]] = {}
    for frame in range(samples.shape[0]):
        row = samples[frame].detach().cpu().contiguous()
        digest = hashlib.blake2b(row.numpy().tobytes(), digest_size=16).digest()
        groups.setdefault(digest, []).append(frame)
    return groups


@within_psf_plans
def subspace_kernel(
    trajectory: Any,
    basis: Any,
    image_shape: tuple[int, ...],
    *,
    density: Any | None = None,
    options: dict[str, Any] | None = None,
    streaming: Any | None = None,
) -> CompactToeplitzKernel:
    """Build the normal operator for a subspace-constrained acquisition.

    A subspace normal costs ``rank (rank + 1) / 2`` gridding transforms -- one
    per basis pair, over every frame's samples at once -- never one per frame,
    which for a fingerprinting scan is a thousand of them. Frames acquired on
    the same trajectory are grouped before any of that, so a scan whose frames
    cycle through a handful of rotations grids those rotations once.

    Parameters
    ----------
    trajectory
        ``(shots, points, axes)`` for one trajectory every frame shares, or
        ``(frames, shots, points, axes)`` for one per frame. ``axes`` is the
        image's dimensionality. Samples are in normalized k-space: ``-0.5``
        is grid location ``-kN/2`` of a grid of size ``kN`` and ``0.5``
        is ``+kN/2``, so the same numbers describe the image grid and
        the doubled one the transfer lives on.
    basis
        ``(frames, rank)`` or ``(rank, frames)`` -- both are accepted, and
        which one it is comes from the data rather than from the caller. The
        axis matching the trajectory's frame count is the frames; where the
        trajectory states no count, the longer axis is. A square basis reads
        as ``(frames, rank)``.

        The frames axis is contrasts for a qMRI scan and time for a dynamic
        one; nothing here needs to know which.
    image_shape
        The image grid. The transfer is built on twice this in every dimension.
    density
        Sample weights, broadcastable to the trajectory without its axes --
        ``(points,)``, ``(shots, points)`` or ``(frames, shots, points)``.
        ``None`` weights every sample equally.
    options
        As :func:`toeplitz_options`.
    streaming
        Where a transfer too large for the device is staged.

    Returns
    -------
    CompactToeplitzKernel
        The packed normal operator, taking ``(batch, rank, *image_shape)``.

    Raises
    ------
    ValueError
        If the trajectory's rank is neither three nor four, its last axis does
        not match the image's dimensionality, the basis has no axis matching
        the frames, or the density does not broadcast onto the samples.

    Examples
    --------
    >>> import numpy as np
    >>> import mrtoeplitz as mt
    >>> angles = np.linspace(0, np.pi, 8, endpoint=False)
    >>> radius = np.linspace(-0.5, 0.5, 32, endpoint=False)
    >>> spokes = np.stack(
    ...     [np.outer(np.cos(angles), radius), np.outer(np.sin(angles), radius)],
    ...     axis=-1,
    ... ).astype(np.float32)
    >>> basis = np.eye(4, 16, dtype=np.float32)

    One trajectory shared by all sixteen frames:

    >>> kernel = mt.subspace_kernel(spokes, basis, (32, 32))
    >>> kernel.rank
    4
    """
    torch = import_module("torch")
    options = _toeplitz_options() if options is None else options
    image_shape = tuple(int(size) for size in image_shape)

    samples, stated_frames = _trajectory_frames(trajectory, len(image_shape))
    matrix = _basis_as_rank_by_frames(basis, stated_frames)
    rank, n_frames = int(matrix.shape[0]), int(matrix.shape[1])

    if stated_frames is None:
        samples = samples.expand(n_frames, *samples.shape[1:])
    elif stated_frames != n_frames:
        raise ValueError(
            f"the trajectory carries {stated_frames} frames and the basis {n_frames}"
        )

    weights = None
    if density is not None:
        weights = as_torch(density).reshape(-1)
        per_frame = samples.shape[1]
        if weights.numel() == per_frame:
            weights = weights[None].expand(n_frames, per_frame)
        elif weights.numel() == n_frames * per_frame:
            weights = weights.reshape(n_frames, per_frame)
        else:
            raise ValueError(
                f"density has {weights.numel()} weights, which is neither one "
                f"per sample of a frame ({per_frame}) nor one per sample of "
                f"the whole acquisition ({n_frames * per_frame})"
            )

    rows, columns = torch.triu_indices(rank, rank, device=matrix.device)
    blocks = []
    for members in _grouped_frames(samples).values():
        coefficients = sum(
            matrix[rows, frame] * matrix[columns, frame].conj() for frame in members
        )
        first = members[0]
        blocks.append(
            (
                samples[first],
                None if weights is None else weights[first],
                coefficients,
            )
        )

    return _subspace_kernel_from_blocks(
        blocks,
        image_shape,
        options=options,
        streaming=streaming,
    )


@within_psf_plans
def cartesian_subspace_kernel(
    masks: Any,
    basis: Any,
    *,
    options: dict[str, Any] | None = None,
    streaming: Any | None = None,
) -> CompactToeplitzKernel:
    """Build an exact packed Cartesian subspace transfer without 2x padding.

    A Cartesian encoding needs no gridding and no doubled grid: the normal is
    the sampling mask itself, so the transfer is exact and lives on the image
    grid.

    Parameters
    ----------
    masks
        Sampling masks, ``(frames, *image_shape)`` or ``(1, *image_shape)`` for
        one mask shared by every frame. Centred, the way k-space is written.
    basis
        ``(frames, rank)`` or ``(rank, frames)``; as
        :func:`subspace_kernel`, the orientation is read off the data.
    options
        As :func:`toeplitz_options`.
    streaming
        Where a transfer too large for the device is staged.

    Returns
    -------
    CompactToeplitzKernel
        The packed normal operator, on the image grid.

    Raises
    ------
    ValueError
        If the mask count is neither one nor the number of frames.
    """
    torch = import_module("torch")
    options = _toeplitz_options() if options is None else options
    masks = as_torch(masks)
    stated_frames = int(masks.shape[0]) if masks.shape[0] > 1 else None
    basis = _basis_as_rank_by_frames(basis, stated_frames)
    rank, n_frames = basis.shape

    if streaming is not None:
        masks = masks.to("cpu")
    image_shape = tuple(int(size) for size in masks.shape[1:])
    masks = masks.reshape(-1, *image_shape)
    if masks.shape[0] == 1:
        masks = masks.expand(n_frames, *image_shape)
    elif masks.shape[0] != n_frames:
        raise ValueError(
            "a Cartesian subspace mask must be shared or have one mask per frame"
        )
    spatial_axes = tuple(range(-len(image_shape), 0))
    masks = torch.fft.ifftshift(masks, dim=spatial_axes).abs().square()

    # A Cartesian mask fills its own grid, so there is nothing to leave out.
    indices = _support_locations(None, image_shape, masks.device, options["compress"])
    rows, columns = torch.triu_indices(rank, rank, device=basis.device)
    packed = torch.zeros(
        (rows.numel(), indices.numel()),
        dtype=torch.promote_types(basis.dtype, masks.dtype),
        device=masks.device,
    )
    basis = basis.to(masks.device)
    for frame in range(n_frames):
        mixing = (
            basis[rows.to(masks.device), frame]
            * basis[columns.to(masks.device), frame].conj()
        )
        sampled_mask = torch.index_select(masks[frame].flatten(), 0, indices)
        packed += mixing[:, None] * sampled_mask[None]
    packed = (
        packed.to(basis.dtype) if basis.is_complex() else packed.real.to(basis.dtype)
    )
    return _make_kernel(
        packed,
        indices,
        image_shape,
        rank,
        image_shape=image_shape,
        options=options,
        streaming=streaming,
    )
