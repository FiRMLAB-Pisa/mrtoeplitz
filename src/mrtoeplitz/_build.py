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
from importlib import import_module
from math import isqrt, prod
from typing import Any

from ._kernel import (
    CompactToeplitzKernel,
    PolyphaseToeplitzKernel,
    as_torch,
    polyphase_components,
)
from ._options import _support_locations, _toeplitz_options
from ._psf import psf_plan, within_psf_plans

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
    plan = psf_plan(spatial_shape, samples)

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
    """Return the largest magnitude of ``stored`` at the locations left out."""
    if left_out is None:
        return 0.0
    return float(stored.abs().flatten()[left_out].max())


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
        return CompactToeplitzKernel(
            values,
            indices,
            spatial_shape,
            rank,
            truncation_bound=truncation_bound,
            **settings,
        )
    components = [
        (
            parity,
            CompactToeplitzKernel(part, where, image_shape, rank, **settings),
        )
        for parity, part, where in polyphase_components(
            values, indices, spatial_shape, image_shape
        )
    ]
    return PolyphaseToeplitzKernel(
        components,
        image_shape,
        rank,
        truncation_bound=truncation_bound,
    )


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
        _compute_toeplitz_transfer(samples, image_shape, weights)
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
    flat = as_torch(indices).to(torch.int64)
    parity = torch.zeros_like(flat)
    stride = 1
    for size in reversed(spatial_shape):
        parity = parity + (flat // stride) % size
        stride *= size
    return torch.where(parity % 2 == 0, 1.0, -1.0).to(torch.complex64)


def _subspace_pair_transfers(
    blocks: Sequence[tuple[Any, Any, Any]],
    image_shape: tuple[int, ...],
    samples: Any,
    counts: Sequence[int],
    indices: Any,
    *,
    streaming: Any | None = None,
    keep_complex: bool = True,
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

    plan = psf_plan(spatial_shape, samples)
    axes = tuple(range(len(spatial_shape)))
    signs = _centring_signs(indices, spatial_shape)
    # The transform is raw, and the convolution runs on the doubled grid,
    # so that grid's size is the whole normalization.
    scale = 1.0 / prod(spatial_shape)
    n_pairs = int(blocks[0][2].numel())
    left_out = _complement_of(indices, prod(spatial_shape), samples.device)
    dropped = 0.0

    packed = None
    coefficients = coefficients.to(device=samples.device, dtype=torch.complex64)
    for pair in range(n_pairs):
        values = torch.repeat_interleave(coefficients[pair], repeats)
        if weights is not None:
            values = values * weights
        values_view = values.reshape(1, 1, -1)
        del values
        psf = plan.grid(values_view).reshape(spatial_shape)
        # Transformed in place, and the centring folded into a sign on the
        # locations kept: shifting the point-spread function by half the grid
        # is the same as alternating the sign of what comes out, and a copy of
        # the doubled grid is the largest thing a build holds after the plan.
        torch.fft.fftn(psf, dim=axes, out=psf)
        flat = psf.reshape(-1)
        if left_out is not None:
            stored = flat.real if not keep_complex else flat
            dropped = max(dropped, _largest_left_out(stored, left_out))
        selected = _selected_transfer(flat, indices, streaming=streaming)
        del psf, flat
        row = selected * signs * scale
        del selected
        if not keep_complex:
            row = row.real
        if packed is None:
            packed = torch.empty(
                (n_pairs, row.numel()),
                dtype=row.dtype,
                device="cpu",
            )
        packed[pair].copy_(row)
        del row
    assert packed is not None
    return packed, dropped


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
    indices = _support_locations(
        samples,
        spatial_shape,
        "cpu" if streaming is not None else samples.device,
        options["compress"],
    )
    packed, dropped = _subspace_pair_transfers(
        stripped,
        image_shape,
        samples,
        counts,
        indices,
        streaming=streaming,
        keep_complex=bool(coefficients.is_complex()),
    )
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
        truncation_bound=dropped,
    )


def _basis_as_rank_by_frames(basis: Any, n_frames: int | None) -> Any:
    """Return the basis as ``(rank, frames)``, whichever way round it came.

    A subspace has fewer components than the dimension it compresses, so where
    the trajectory does not settle the frame count the longer axis is the
    frames. A square basis compresses nothing and is read as ``(frames, rank)``.
    """
    torch = import_module("torch")
    basis = torch.as_tensor(basis)
    if basis.ndim != 2:
        raise ValueError(
            f"basis must be (frames, rank) or (rank, frames), got shape "
            f"{tuple(basis.shape)}"
        )
    if n_frames is not None:
        if basis.shape[0] == n_frames and basis.shape[1] != n_frames:
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
        ``(frames, rank)`` or ``(rank, frames)``, whichever way round. The
        frames axis is contrasts for a qMRI scan and time for a dynamic one;
        nothing here needs to know which.
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
        ``(frames, rank)`` or ``(rank, frames)``, whichever way round.
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
    )
