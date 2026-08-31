"""Sensitivities held as k-space kernels rather than as maps.

A sensitivity is smooth. That is why a calibration is solved on a
low-resolution ACS region at all, and it means the natural form of a
sensitivity is a small array of k-space coefficients rather than a map: what
NLINV solves for *is* that array, and the dense map is it zero-padded to the
image grid and transformed.

So forming the map is a choice, and an expensive one -- at 320 cubed with 48
channels, complex single-precision maps are 12.6 GB, which on most cards is
the reconstruction. Holding the coefficients instead and expanding one coil
when the apply asks for it defers the padding, and a SENSE normal reads a bank
one coil batch at a time anyway. Nothing is approximated by this: nothing has
been truncated.

This is riesling's low-memory mode. :meth:`CoilKernels.from_maps` covers the
other direction, for when the coefficients are gone and only maps remain.
"""

from __future__ import annotations

__all__ = ["CoilKernels"]

from importlib import import_module
from math import prod
from typing import Any


def _torch() -> Any:
    try:
        return import_module("torch")
    except ImportError as error:
        raise ImportError(
            "coil kernels require Torch: pip install mrtoeplitz"
        ) from error


def _resize_centered(value: Any, shape: tuple[int, ...]) -> Any:
    """Crop or zero-pad the trailing axes about the centre ``fftshift`` uses.

    The centre is index ``n // 2``, so a kernel cropped out of a centred
    spectrum and padded back lands where it came from, for even and odd sizes
    alike.
    """
    torch = _torch()
    spatial_ndim = len(shape)
    result = torch.zeros(
        (*value.shape[:-spatial_ndim], *shape),
        dtype=value.dtype,
        device=value.device,
    )
    source: list[Any] = [slice(None)] * value.ndim
    target: list[Any] = [slice(None)] * value.ndim
    for offset, size in enumerate(shape, start=value.ndim - spatial_ndim):
        have = value.shape[offset]
        count = min(have, size)
        source[offset] = slice(
            (have // 2) - count // 2, (have // 2) - count // 2 + count
        )
        target[offset] = slice(
            (size // 2) - count // 2, (size // 2) - count // 2 + count
        )
    result[tuple(target)] = value[tuple(source)]
    return result


def _smallest_kernel(
    spectrum: Any,
    image_shape: tuple[int, ...],
    tolerance: float,
) -> tuple[int, ...]:
    """Smallest centred cube whose truncation error is within ``tolerance``.

    Both transforms are the pair the class expands with, so by Parseval the
    error a truncation leaves is the square root of the energy it drops. That
    is read off the spectrum directly, which is why sizing costs one transform
    rather than one per candidate.
    """
    torch = _torch()
    power = spectrum.abs().square()
    total = power.sum()
    if total <= 0:
        return tuple(2 for _ in image_shape)

    spatial = tuple(range(power.ndim - len(image_shape), power.ndim))
    largest = max(image_shape)

    def dropped(side: int) -> tuple[tuple[int, ...], float]:
        shape = tuple(min(side, size) for size in image_shape)
        kept = power
        for axis, size in zip(spatial, shape, strict=True):
            start = (kept.shape[axis] // 2) - size // 2
            kept = kept.narrow(axis, start, size)
        return shape, float(torch.sqrt(torch.clamp(1.0 - kept.sum() / total, min=0.0)))

    # The full grid is not a candidate: it is the dense bank under another
    # name, and answering with it would report a saving of one.
    best = 1.0
    for side in range(2, largest, 2):
        shape, error = dropped(side)
        best = min(best, error)
        if error <= tolerance:
            return shape
    raise ValueError(
        f"no kernel smaller than {image_shape} holds these maps to "
        f"{tolerance:g}; the closest gets to {best:.2e}. A map that is not "
        f"band-limited cannot be truncated -- keep the calibration kernels "
        f"and never form it."
    )


class CoilKernels:
    """Coil sensitivities stored as truncated centred k-space kernels.

    Presents itself as the map bank it stands for -- ``shape``, ``ndim``,
    ``dtype``, ``device`` and slicing on the leading axes all answer as the
    dense tensor would -- and materialises only the coils that are asked for.
    Every consumer of a sensitivity bank in this package reads it one coil
    batch at a time, so nothing else has to change to use one.

    Indexing acts on the leading axes and materialising acts on the spatial
    ones, so the two commute: ``coil_kernels[i]`` is exactly the ``i``-th map of
    ``coil_kernels.materialize()``, and neither ever forms the coils it was not
    asked for.

    Parameters
    ----------
    kernels
        Centred k-space kernels, ``(coils, *kernel_shape)`` or with a leading
        batch axis. Complex.
    image_shape
        The grid a coil is expanded onto. Must have as many entries as the
        kernel has trailing axes.

    Attributes
    ----------
    kernel_shape : tuple of int
        What is actually stored, per coil.

    Raises
    ------
    ValueError
        If ``image_shape`` does not fit the kernels, or a kernel side exceeds
        the image side it expands onto.

    Examples
    --------
    >>> import torch
    >>> import mrtoeplitz as mt
    >>> maps = torch.ones(8, 64, 64, dtype=torch.complex64)
    >>> coil_kernels = mt.CoilKernels.from_maps(maps, (12, 12))

    It stands in for the bank it came from, and holds a fraction of it:

    >>> coil_kernels.shape
    (8, 64, 64)
    >>> coil_kernels[0:2].shape
    torch.Size([2, 64, 64])
    >>> round(coil_kernels.compression_ratio, 1)
    28.4
    """

    def __init__(self, kernels: Any, image_shape: tuple[int, ...]) -> None:
        torch = _torch()
        kernels = torch.as_tensor(kernels)
        if not kernels.is_complex():
            kernels = kernels.to(torch.complex64)
        image_shape = tuple(int(size) for size in image_shape)
        if kernels.ndim < len(image_shape) + 1:
            raise ValueError(
                f"kernels of shape {tuple(kernels.shape)} cannot carry a "
                f"{len(image_shape)}-dimensional image plus a coil axis"
            )
        kernel_shape = tuple(int(size) for size in kernels.shape[-len(image_shape) :])
        too_wide = [
            (kernel, image)
            for kernel, image in zip(kernel_shape, image_shape, strict=True)
            if kernel > image
        ]
        if too_wide:
            raise ValueError(
                f"kernel shape {kernel_shape} exceeds the image grid {image_shape}; "
                f"a kernel is a truncation, not an interpolation"
            )
        self._kernels = kernels
        self.image_shape = image_shape
        self.kernel_shape = kernel_shape

    @classmethod
    def from_maps(
        cls,
        maps: Any,
        kernel_shape: tuple[int, ...] | None = None,
        *,
        tolerance: float | None = None,
        spatial_ndim: int | None = None,
    ) -> CoilKernels:
        """Reduce a dense map bank to central k-space kernels.

        The fallback, not the path: where a calibration's coefficients are to
        hand, pass them to :class:`CoilKernels` directly and no truncation
        arises. This is for a bank that reaches you already padded.

        **Only sound for maps that are still band-limited.** NLINV's are, by
        construction: its Sobolev weighting is a band limit, and BART already
        stores those maps as k-space coefficients. ESPIRiT's are not, once the
        eigenvector normalisation has been applied -- the ``|m| = 1`` mask puts
        a hard edge in the image, whose k-space support is broad, so truncating
        it is not band-limiting but ringing. For ESPIRiT, keep the calibration
        kernels and never form the map.

        :meth:`truncation_error` says which case you are in, so this does not
        have to be taken on trust -- and ``tolerance`` does not require you to
        ask, because it sizes the kernel by that same measurement.

        Parameters
        ----------
        maps
            Dense sensitivities, ``(coils, *image_shape)`` or with a leading
            batch axis.
        kernel_shape
            Sides of the kernel to keep, centred on k-space DC. Give this or
            ``tolerance``, not both.
        tolerance
            Size the kernel instead: the smallest cube whose truncation error
            is at or below this. The error is read off the spectrum by
            Parseval rather than by expanding a candidate, so this costs one
            transform however many sizes it considers.
        spatial_ndim
            How many trailing axes are the image, when ``tolerance`` is used.
            ``None`` treats every axis after the first as spatial.

        Returns
        -------
        CoilKernels
            The truncated bank.

        Raises
        ------
        ValueError
            If neither or both of ``kernel_shape`` and ``tolerance`` are
            given, or if no kernel short of the full grid meets the tolerance
            -- which is what a map that is not band-limited looks like.

        Examples
        --------
        >>> import torch
        >>> import mrtoeplitz as mt
        >>> maps = torch.ones(4, 32, 32, dtype=torch.complex64)
        >>> mt.CoilKernels.from_maps(maps, (8, 8)).kernel_shape
        (8, 8)

        Sized by measurement instead. A constant map is one k-space cell, so
        the smallest kernel that carries it is the smallest there is:

        >>> mt.CoilKernels.from_maps(maps, tolerance=1e-6).kernel_shape
        (2, 2)
        """
        torch = _torch()
        maps = torch.as_tensor(maps)
        if not maps.is_complex():
            maps = maps.to(torch.complex64)
        if (kernel_shape is None) == (tolerance is None):
            raise ValueError("give from_maps a kernel_shape or a tolerance, not both")

        if kernel_shape is not None:
            kernel_shape = tuple(int(size) for size in kernel_shape)
            image_shape = tuple(int(size) for size in maps.shape[-len(kernel_shape) :])
            spectrum = cls._forward(maps, len(kernel_shape))
            return cls(_resize_centered(spectrum, kernel_shape), image_shape)

        ndim = maps.ndim - 1 if spatial_ndim is None else int(spatial_ndim)
        image_shape = tuple(int(size) for size in maps.shape[-ndim:])
        spectrum = cls._forward(maps, ndim)
        kernel_shape = _smallest_kernel(spectrum, image_shape, float(tolerance))
        return cls(_resize_centered(spectrum, kernel_shape), image_shape)

    @staticmethod
    def _forward(maps: Any, spatial_ndim: int) -> Any:
        """Centred k-space of a map bank, unnormalised."""
        torch = _torch()
        axes = tuple(range(-spatial_ndim, 0))
        return torch.fft.fftshift(
            torch.fft.fftn(torch.fft.ifftshift(maps, dim=axes), dim=axes), dim=axes
        )

    def _expand(self, kernels: Any) -> Any:
        """Zero-pad to the image grid and transform back.

        The forward transform is unnormalised and this one divides by the image
        grid, which is the same count the kernels were cut from, so a bank
        truncated and expanded again differs from the original only by what the
        truncation dropped.
        """
        torch = _torch()
        axes = tuple(range(-len(self.image_shape), 0))
        padded = _resize_centered(kernels, self.image_shape)
        return torch.fft.fftshift(
            torch.fft.ifftn(torch.fft.ifftshift(padded, dim=axes), dim=axes), dim=axes
        )

    def __getitem__(self, index: Any) -> Any:
        """Materialise the coils ``index`` selects, and only those."""
        return self._expand(self._kernels[index])

    def materialize(self) -> Any:
        """Expand every coil at once, as the dense bank this stands in for."""
        return self._expand(self._kernels)

    def truncation_error(self, maps: Any) -> float:
        """Relative energy the truncation dropped, against the maps it came from.

        Small for a band-limited bank and not small for a masked ESPIRiT one,
        which is the distinction :meth:`from_maps` rests on.

        Parameters
        ----------
        maps
            The dense sensitivities this bank was truncated from.

        Returns
        -------
        float
            ``||maps - materialize()|| / ||maps||``.

        Examples
        --------
        A map that is band-limited by construction loses nothing:

        >>> import torch
        >>> import mrtoeplitz as mt
        >>> _ = torch.manual_seed(0)
        >>> seed = torch.randn(4, 12, 12, dtype=torch.complex64)
        >>> maps = mt.CoilKernels(seed, (64, 64)).materialize()
        >>> mt.CoilKernels.from_maps(maps, (12, 12)).truncation_error(maps) < 1e-5
        True

        A map that does not vanish at the edge of the field of view wraps, and
        its spectrum falls off like one over frequency, so it does not:

        >>> ramp = torch.linspace(0.0, 1.0, 64)
        >>> sloped = (ramp[:, None] * torch.ones(64))[None].to(torch.complex64)
        >>> bool(mt.CoilKernels.from_maps(sloped, (12, 12)).truncation_error(sloped) > 1e-2)
        True
        """
        torch = _torch()
        maps = torch.as_tensor(maps).to(self._kernels.dtype)
        difference = torch.linalg.vector_norm(maps - self.materialize().to(maps.device))
        return float(difference / torch.linalg.vector_norm(maps))

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of the dense bank this stands in for."""
        return (
            *tuple(int(size) for size in self._kernels.shape[: -len(self.image_shape)]),
            *self.image_shape,
        )

    @property
    def ndim(self) -> int:
        """Rank of the dense bank this stands in for."""
        return self._kernels.ndim

    @property
    def dtype(self) -> Any:
        """Dtype a materialised coil comes back in."""
        return self._kernels.dtype

    @property
    def device(self) -> Any:
        """Where the kernels are held, and where a coil is materialised."""
        return self._kernels.device

    @property
    def kernels(self) -> Any:
        """The stored kernels themselves."""
        return self._kernels

    @property
    def storage_nbytes(self) -> int:
        """Bytes the kernels occupy."""
        return int(self._kernels.numel() * self._kernels.element_size())

    @property
    def dense_nbytes(self) -> int:
        """Bytes the map bank this stands in for would occupy."""
        return int(prod(self.shape) * self._kernels.element_size())

    @property
    def compression_ratio(self) -> float:
        """How many times smaller the kernels are than the bank."""
        return self.dense_nbytes / self.storage_nbytes

    def to(self, device: Any) -> CoilKernels:
        """Move the kernels to ``device``, or return self when already there."""
        moved = self._kernels.to(device)
        if moved is self._kernels:
            return self
        return CoilKernels(moved, self.image_shape)

    def __len__(self) -> int:
        """Return the leading-axis length, as the dense bank would."""
        return int(self._kernels.shape[0])

    def __repr__(self) -> str:
        return (
            f"CoilKernels(shape={self.shape}, kernel_shape={self.kernel_shape}, "
            f"compression={self.compression_ratio:.1f}x)"
        )
