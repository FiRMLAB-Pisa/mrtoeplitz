"""How a kernel is stored and executed, and the locations it is stored over."""

from __future__ import annotations

__all__ = ["_support_locations", "_toeplitz_options"]

from typing import Any

from ._kernel import occupancy_indices, support_indices


def _toeplitz_options(
    *,
    compress: bool = True,
    polyphase: bool | str = True,
    chunk_size: int = 65536,
    coil_batch_size: int = 1,
    cuda_mode: str = "compact",
    cuda_max_device_fraction: float = 0.85,
    cuda_transfer_precision: str = "auto",
    gridding_tolerance: float | None = None,
) -> dict[str, Any]:
    """Validate how a Toeplitz kernel is applied.

    Nothing here says how the kernel is built. It is gridded onto the doubled
    grid the way BART and MRISubspaceRecon.jl build theirs, and that is not a
    choice. What is left is what it is stored and executed on: whether the
    locations the trajectory never reached are dropped, how much is unpacked
    at a time, how many coils share a pass, and what a CUDA device holds.

    ``compress`` is BART's ``--compress-psf``: the transfer is kept where the
    gridded trajectory is non-zero -- the sampled region plus the rim the
    interpolation spreads into -- and dropped outside, which is what makes a
    large three-dimensional kernel fit. The transfer multiplies the spectrum
    pointwise, so what compression leaves out perturbs the normal operator by
    at most the largest value dropped, which the kernel records as its
    truncation bound. A conjugate-gradient solve that meets the resulting
    indefiniteness stops on its last valid iterate rather than diverging. A
    calibration solved over a small window keeps the whole transfer instead.

    ``cuda_mode`` chooses the CUDA lane. The default is ``"compact"``: the
    transfer is applied out of its packed form, so what the device holds is
    the transfer and one working buffer. ``"resident"`` materialises batched
    banks on the doubled grid instead, which is faster where there is room for
    them and impossible where there is not, and ``"auto"`` takes the banks
    whenever they fit. Compact is the default because memory is the scarce
    thing this package exists to save, on a large card as much as a small one.

    ``gridding_tolerance`` is what the transfer is gridded to. The default is
    loose on purpose: the transfer is then cut to the support the scan reached
    and encoded in bfloat16 on the device, and what those leave is an order of
    magnitude larger than what the gridding does. A build that keeps the whole
    transfer -- ``compress=False`` -- has no such margin and should ask for a
    tighter one.

    ``polyphase`` files the transfer as one component per parity of the
    doubled grid's coordinates, so the convolution runs on the image grid and
    the doubled one is never materialised. It is the same operator either way,
    and it is the default: the memory is won on every grid, and where the
    doubled form would not have fitted it is the difference between a solve
    that stays resident and one that falls back to a slower lane. Where the
    doubled form does fit, filing by parity costs a few percent of runtime,
    which is what ``False`` buys back. ``"auto"`` keeps the doubled grid until
    its banks no longer fit the device budget.
    """
    if gridding_tolerance is not None and not 0.0 < gridding_tolerance < 1.0:
        raise ValueError("Toeplitz gridding_tolerance must be in (0, 1)")
    if chunk_size <= 0:
        raise ValueError("Toeplitz chunk_size must be positive")
    if coil_batch_size <= 0:
        raise ValueError("Toeplitz coil_batch_size must be positive")
    if cuda_mode not in {"auto", "resident", "compact"}:
        raise ValueError("Toeplitz cuda_mode must be 'auto', 'resident', or 'compact'")
    if not 0.0 < cuda_max_device_fraction <= 1.0:
        raise ValueError("Toeplitz cuda_max_device_fraction must be in (0, 1]")
    if polyphase not in (True, False, "auto"):
        raise ValueError("Toeplitz polyphase must be True, False, or 'auto'")
    if cuda_transfer_precision not in {"auto", "float32", "bfloat16"}:
        raise ValueError(
            "Toeplitz cuda_transfer_precision must be 'auto', 'float32', or 'bfloat16'"
        )
    return {
        "compress": bool(compress),
        "polyphase": polyphase,
        "chunk_size": int(chunk_size),
        "coil_batch_size": int(coil_batch_size),
        "cuda_mode": cuda_mode,
        "cuda_max_device_fraction": float(cuda_max_device_fraction),
        "cuda_transfer_precision": cuda_transfer_precision,
        "gridding_tolerance": gridding_tolerance,
    }


_SPREAD_HALF_WIDTH = 4


def _support_locations(
    samples: Any,
    spatial_shape: tuple[int, ...],
    device: Any,
    compress: bool = True,
) -> Any:
    """Return the locations a gridded transfer holds weight at.

    Where the trajectory landed on the doubled grid, plus the neighbourhood the
    interpolation spread into. An encoding with no trajectory to read, which is
    what a Cartesian one leaves, keeps every location, and so does one that
    asks not to be compressed.
    """
    if samples is None or not compress:
        return support_indices(spatial_shape, support="full", radius=1.0, device=device)
    return occupancy_indices(samples, spatial_shape, width=_SPREAD_HALF_WIDTH).to(
        device
    )
