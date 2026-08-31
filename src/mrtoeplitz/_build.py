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

An operator here is anything exposing ``shape``, ``samples``, ``norm_factor``
and optionally ``density`` and ``backend`` -- an MRI-NUFFT operator satisfies
it, and so does a caller's own.
"""

from __future__ import annotations

__all__ = [
    "cartesian_subspace_kernel",
    "scalar_kernel",
    "subspace_kernel",
    "toeplitz_options",
]

from collections.abc import Sequence
from contextlib import contextmanager, suppress
from functools import wraps
from importlib import import_module
from math import isqrt, prod
from typing import Any

from ._backend import _base_fourier_operator, _mrinufft_norm_factor, _require_mrinufft
from ._kernel import (
    CompactToeplitzKernel,
    PolyphaseToeplitzKernel,
    as_torch,
    polyphase_components,
)
from ._options import _support_locations, _toeplitz_options

#: Public name for the options validator.
toeplitz_options = _toeplitz_options


_PSF_OPERATOR_SLOT: dict[tuple[Any, ...], Any] = {}

_PSF_TOLERANCE = 1e-4

_NARROW_PSF_TOLERANCE = 1e-3

_NARROW_PSF_UPSAMPLING = 1.25


def _psf_operator(
    samples: Any,
    backend: str,
    spatial_shape: tuple[int, ...],
) -> Any:
    """Return a NUFFT on the doubled grid, for gridding the transfer onto it.

    One plan is kept per (backend, grid, sample count) and retargeted at each
    trajectory it is asked for. Planning a NUFFT is the expensive part of
    building a kernel, and holding a second plan on the doubled grid is what
    makes a build run out of device memory.
    """
    mrinufft = _require_mrinufft()
    shape = tuple(int(size) for size in spatial_shape)
    key = (backend, shape, int(samples.shape[0]))
    operator = _PSF_OPERATOR_SLOT.get(key)
    if operator is not None:
        operator.update_samples(samples)
        return operator
    build = mrinufft.get_operator(backend)
    _yield_cached_device_memory(getattr(samples, "device", None))
    settings: dict[str, Any] = _psf_settings(shape, samples)
    try:
        operator = build(
            samples=samples,
            shape=shape,
            density=None,
            n_coils=1,
            squeeze_dims=False,
            **settings,
        )
    except TypeError:
        operator = build(
            samples=samples,
            shape=shape,
            density=None,
            n_coils=1,
            squeeze_dims=False,
        )
    # One slot: a plan on the doubled grid is the largest device allocation a
    # build makes, and holding a second one is what makes a build run out.
    _PSF_OPERATOR_SLOT.clear()
    _PSF_OPERATOR_SLOT[key] = operator
    return operator


def _yield_cached_device_memory(device: Any) -> None:
    """Hand the allocator's spare blocks back to the driver.

    A NUFFT plan is allocated outside Torch, so blocks Torch is holding for
    reuse are neither available to it nor counted as free -- and Torch does
    not release them when another library runs out. What a build measures and
    what it can take are both only true once these are returned.
    """
    torch = import_module("torch")
    if "cuda" not in str(device):
        return
    with suppress(RuntimeError):
        torch.cuda.empty_cache()


def _psf_settings(shape: tuple[int, ...], samples: Any) -> dict[str, Any]:
    """Return what to plan the gridding NUFFT with, for the room there is.

    A NUFFT spreads onto a grid of its own on the way to the one it answers
    on; that grid is internal and does not touch the transfer, so it is chosen
    for what it costs. The wide one is the default. On the doubled grid a
    kernel is built on it is eight times the transfer, so at these sizes it
    stops fitting, and the narrow one is asked for a looser tolerance -- which
    keeps its interpolation kernel the width the wide one has, and spends the
    difference on the transfer rather than on every point spread onto it.
    """
    torch = import_module("torch")
    narrow = {"eps": _NARROW_PSF_TOLERANCE, "upsampfac": _NARROW_PSF_UPSAMPLING}
    # NumPy answers `device` with a plain string, Torch with an object.
    device = getattr(samples, "device", None)
    if "cuda" not in str(device):
        return {"eps": _PSF_TOLERANCE}
    free, _ = torch.cuda.mem_get_info(device)
    spreading = 8 * (2 ** len(shape)) * prod(shape)
    wide = spreading + 8 * prod(shape) + 8 * int(samples.shape[0])
    return {"eps": _PSF_TOLERANCE} if wide < 0.6 * free else narrow


def _within_psf_plans(build: Any) -> Any:
    """Release the gridding plan a builder makes when its build ends."""

    @wraps(build)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with _psf_plans():
            return build(*args, **kwargs)

    return wrapper


@contextmanager
def _psf_plans() -> Any:
    """Hold one gridding plan for the length of a build, then release it.

    A plan on the doubled grid is the largest device allocation a build makes
    -- larger than the kernel it produces -- and the solve that follows needs
    that memory for its own transforms.
    """
    try:
        yield
    finally:
        _PSF_OPERATOR_SLOT.clear()
        with suppress(ImportError, AttributeError):
            torch = import_module("torch")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def _compute_toeplitz_transfer(
    native_operator: Any,
    weights: Any | None = None,
    *,
    complex_weights: bool = False,
) -> Any:
    """Return the transfer a Toeplitz normal operator multiplies by.

    The point-spread function is the adjoint of the sample weights taken on a
    grid twice the image in every dimension -- ones for a plain normal, the
    density for a compensated one, a basis product for a subspace frame or an
    off-resonance segment -- and the transfer is its transform.

    Gridding is what puts the weight where the trajectory is. The adjoint
    interpolates each sample onto the doubled grid with the backend's own
    kernel, so the transfer holds weight where the scan reached and in the rim
    that interpolation spreads into, and nowhere else. That is the same
    operator the forward NUFFT applies, so the normal is the Gram of the
    transform actually being inverted.
    """
    del complex_weights
    torch = import_module("torch")
    base = _base_fourier_operator(native_operator)
    image_shape = tuple(int(size) for size in base.shape)
    spatial_shape = tuple(2 * size for size in image_shape)
    operator = _psf_operator(
        base.samples,
        getattr(base, "backend", "finufft"),
        spatial_shape,
    )

    if weights is None:
        # The plain normal is weighted by whatever the operator itself carries:
        # its adjoint applies the density once, so the Gram does too.
        weights = getattr(base, "density", None)
    if weights is None:
        values = torch.ones(
            operator.n_samples,
            dtype=torch.complex64,
            device=as_torch(base.samples).device,
        )
    else:
        values = as_torch(weights).reshape(-1).to(torch.complex64)

    # Backends differ on whether they take a bare sample vector, so the
    # batch and coil axes are stated and dropped again.
    psf = as_torch(operator.adj_op(values.reshape(1, 1, -1))).reshape(spatial_shape)
    axes = tuple(range(len(spatial_shape)))
    # ``adj_op`` answers a centred image and divides by the doubled grid's own
    # normalization, while the normal operator this stands in for carries the
    # image grid's twice -- once in the forward and once in the adjoint.
    scale = float(operator.norm_factor) / float(base.norm_factor) ** 2
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


@_within_psf_plans
def scalar_kernel(
    native_operator: Any,
    options: dict[str, Any] | None = None,
    streaming: Any | None = None,
) -> CompactToeplitzKernel:
    """Build the rank-one normal operator for one trajectory.

    Parameters
    ----------
    native_operator
        A NUFFT exposing ``shape``, ``samples``, ``norm_factor`` and
        optionally ``density``. Its density, when it carries one, weights the
        transfer: the adjoint applies it once, so the Gram does too.
    options
        As :func:`toeplitz_options`.
    streaming
        Where a transfer too large for the device is staged.

    Returns
    -------
    CompactToeplitzKernel
        The normal operator, on the doubled grid.
    """
    options = _toeplitz_options() if options is None else options
    base = _base_fourier_operator(native_operator)
    image_shape = tuple(int(size) for size in base.shape)
    spatial_shape = tuple(2 * size for size in image_shape)
    transfer = as_torch(_compute_toeplitz_transfer(base)).flatten()
    indices = _support_locations(
        getattr(base, "samples", None),
        spatial_shape,
        "cpu" if streaming is not None else transfer.device,
        options["compress"],
    )
    left_out = _complement_of(indices, transfer.numel(), transfer.device)
    values = _selected_transfer(transfer, indices, streaming=streaming).real[None]
    kernel = _make_kernel(
        values,
        indices,
        spatial_shape,
        1,
        image_shape=image_shape,
        options=options,
        truncation_bound=_largest_left_out(transfer.real, left_out),
    )
    return kernel


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
    backend: str,
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

    operator = _psf_operator(samples, backend, spatial_shape)
    axes = tuple(range(len(spatial_shape)))
    signs = _centring_signs(indices, spatial_shape)
    scale = float(operator.norm_factor) / _mrinufft_norm_factor(image_shape) ** 2
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
        psf = as_torch(operator.adj_op(values_view)).reshape(spatial_shape)
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


def subspace_kernel(
    blocks: Sequence[tuple[Any, Any, Any]],
    image_shape: tuple[int, ...],
    *,
    backend: str = "finufft",
    options: dict[str, Any] | None = None,
    streaming: Any | None = None,
) -> CompactToeplitzKernel:
    """Grid one transfer per basis pair and pack them as coefficient matrices.

    A subspace normal costs ``rank (rank + 1) / 2`` gridding transforms, each
    the adjoint of one weight per sample over every frame's samples
    concatenated -- never one transform per frame, which for a real
    fingerprinting scan is a thousand of them.

    Parameters
    ----------
    blocks
        One entry per **distinct trajectory**, as
        ``(samples, weights, coefficients)``. ``samples`` is that trajectory,
        ``weights`` its density or ``None``, and ``coefficients`` the
        upper-triangular basis product ``conj(U[i, t]) * U[j, t]`` summed over
        the frames that share the trajectory, ordered as
        ``torch.triu_indices(rank, rank)`` gives the pairs. Grouping frames
        onto the trajectories they were acquired on is the caller's to do,
        because only the caller knows which frames share a plan.
    image_shape
        The image grid. The transfer is built on twice this in every dimension.
    backend
        MRI-NUFFT backend used to grid the point spread function.
    options
        As :func:`toeplitz_options`.
    streaming
        Where a transfer too large for the device is staged. ``None`` keeps it
        wherever it was built.

    Returns
    -------
    CompactToeplitzKernel
        The packed normal operator.

    Raises
    ------
    ValueError
        If ``blocks`` is empty, or the coefficient vectors do not all describe
        one packed upper triangle.
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
        backend,
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
        The subspace basis, ``(rank, frames)``.
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
    basis = torch.as_tensor(basis)
    rank, n_frames = basis.shape

    masks = as_torch(masks)
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
