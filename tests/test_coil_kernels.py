"""Sensitivities held as k-space kernels.

The representation is exact for a band-limited map and lossy for one that is
not, and which of those a caller has is measurable rather than a matter of
faith. These tests say where the line falls.
"""

import numpy as np
import pytest
import torch

import mrtoeplitz as mt

SINGLE = mt.toeplitz_options(compress=False, cuda_transfer_precision="float32")


@pytest.fixture
def band_limited():
    """A map bank that is band-limited by construction."""

    def build(n_coils=4, kernel_side=12, image_side=64, seed=0):
        torch.manual_seed(seed)
        seed_kernels = torch.randn(
            n_coils, kernel_side, kernel_side, dtype=torch.complex64
        )
        return seed_kernels, mt.CoilKernels(
            seed_kernels, (image_side, image_side)
        ).materialize()

    return build


@pytest.fixture
def surface_maps():
    """A Biot-Savart loop array: a real sensitivity, which does not vanish at
    the edge of the field of view."""

    def build(n_coils=4, image_side=64):
        grid = torch.linspace(-0.5, 0.5, image_side)
        yy, xx = torch.meshgrid(grid, grid, indexing="ij")
        maps = []
        for angle in torch.linspace(0, 2 * torch.pi, n_coils + 1)[:n_coils]:
            cx, cy = 0.6 * torch.cos(angle), 0.6 * torch.sin(angle)
            r2 = (xx - cx) ** 2 + (yy - cy) ** 2 + 0.04
            maps.append(((xx - cx) + 1j * (yy - cy)) / r2)
        return torch.stack(maps).to(torch.complex64)

    return build


def test_the_round_trip_is_exact_for_a_band_limited_map(band_limited):
    seed, maps = band_limited()
    again = mt.CoilKernels.from_maps(maps, seed.shape[-2:])
    assert again.truncation_error(maps) < 1e-5
    recovered = torch.linalg.vector_norm(again.kernels - seed)
    assert float(recovered / torch.linalg.vector_norm(seed)) < 1e-5


def test_a_kernel_wider_than_the_band_loses_nothing(band_limited):
    _, maps = band_limited(kernel_side=12)
    assert mt.CoilKernels.from_maps(maps, (24, 24)).truncation_error(maps) < 1e-5


def test_indexing_commutes_with_materialising(band_limited):
    """The invariant the whole design rests on.

    Slicing acts on the leading axes and expanding acts on the spatial ones,
    so a coil taken from the kernels is the same coil taken from the bank --
    and taking it never forms the coils it did not ask for.
    """
    _, maps = band_limited(n_coils=6)
    kernels = mt.CoilKernels.from_maps(maps, (12, 12))
    whole = kernels.materialize()
    assert torch.allclose(kernels[2:5], whole[2:5], atol=1e-6)
    assert torch.allclose(kernels[0], whole[0], atol=1e-6)


def test_it_stands_in_for_the_dense_bank(band_limited):
    _, maps = band_limited(n_coils=8, image_side=64)
    kernels = mt.CoilKernels.from_maps(maps, (12, 12))
    assert kernels.shape == (8, 64, 64)
    assert kernels.ndim == maps.ndim
    assert kernels.dtype == maps.dtype
    assert kernels.device == maps.device
    assert len(kernels) == 8


def test_the_compression_ratio_is_the_size_it_saves(band_limited):
    _, maps = band_limited(n_coils=8, image_side=64)
    kernels = mt.CoilKernels.from_maps(maps, (16, 16))
    assert kernels.compression_ratio == pytest.approx((64 * 64) / (16 * 16))
    assert kernels.storage_nbytes * kernels.compression_ratio == pytest.approx(
        kernels.dense_nbytes
    )


def test_a_kernel_wider_than_the_image_is_refused():
    with pytest.raises(ValueError, match="a truncation, not an interpolation"):
        mt.CoilKernels(torch.zeros(2, 32, 32, dtype=torch.complex64), (16, 16))


def test_kernels_that_cannot_carry_the_image_are_refused():
    with pytest.raises(ValueError, match="cannot carry"):
        mt.CoilKernels(torch.zeros(8, 8, dtype=torch.complex64), (16, 16, 16))


def test_to_moves_the_kernels(band_limited, device):
    _, maps = band_limited()
    kernels = mt.CoilKernels.from_maps(maps, (12, 12)).to(device)
    assert kernels.device.type == device
    assert kernels[0:2].device.type == device


def test_a_surface_map_does_not_truncate_cleanly(surface_maps):
    """A real sensitivity does not vanish at the edge of the field of view, so
    it wraps and its spectrum falls off like one over frequency. Truncating a
    finished map of one is lossy at the percent level, and staying inside the
    object does not rescue it -- which is why the kernels should come from the
    calibration rather than from a map that was formed first."""
    maps = surface_maps()
    kernels = mt.CoilKernels.from_maps(maps, (16, 16))
    assert kernels.truncation_error(maps) > 1e-2

    grid = torch.linspace(-0.5, 0.5, 64)
    yy, xx = torch.meshgrid(grid, grid, indexing="ij")
    inside = (torch.sqrt(xx**2 + yy**2) < 0.3)[None]
    error = torch.linalg.vector_norm(
        (kernels.materialize() - maps) * inside
    ) / torch.linalg.vector_norm(maps * inside)
    assert float(error) > 1e-2


def test_tapering_the_field_of_view_edge_makes_truncation_viable(surface_maps):
    """The same map, brought to zero at the boundary, truncates two orders
    better -- the error was the wrap, not the coil geometry."""
    maps = surface_maps()
    window = torch.hann_window(64)
    taper = (window[:, None] * window[None])[None].to(torch.complex64)
    tapered = maps * taper

    raw = mt.CoilKernels.from_maps(maps, (16, 16)).truncation_error(maps)
    smooth = mt.CoilKernels.from_maps(tapered, (16, 16)).truncation_error(tapered)
    assert smooth < 0.05 * raw


def test_a_sense_normal_reads_kernels_as_it_reads_a_dense_bank(
    radial, image, band_limited, device
):
    """The point of the whole thing: nothing in the apply changes."""
    _, maps = band_limited(n_coils=4, image_side=32)
    x = torch.as_tensor(image(shape=(32, 32)))[None, None].to(device)
    kernel = mt.scalar_kernel(radial(), (32, 32), options=SINGLE).to(device)

    dense = mt.apply_sense(kernel, x, maps.to(device))
    compact = mt.apply_sense(
        kernel, x, mt.CoilKernels.from_maps(maps, (12, 12)).to(device)
    )

    error = torch.linalg.vector_norm(compact - dense) / torch.linalg.vector_norm(dense)
    assert float(error) < 1e-4
    assert np.isfinite(float(torch.linalg.vector_norm(compact)))


def _band_limited(coils=4, side=10, grid=64, seed=0):
    """A bank that is band-limited by construction, as an NLINV map is."""
    rng = np.random.default_rng(seed)
    seed_kernels = torch.as_tensor(
        rng.normal(size=(coils, side, side))
        + 1j * rng.normal(size=(coils, side, side))
    ).to(torch.complex64)
    return mt.CoilKernels(seed_kernels, (grid, grid)).materialize()


def test_a_tolerance_sizes_the_kernel_and_is_met():
    """Sizing is by Parseval on the spectrum, so it has to hold when expanded."""
    maps = _band_limited()
    for tolerance in (1e-2, 1e-4, 1e-6):
        kernels = mt.CoilKernels.from_maps(maps, tolerance=tolerance)
        assert kernels.truncation_error(maps) <= tolerance


def test_a_tolerance_picks_a_smaller_kernel_than_a_looser_one_would():
    maps = _band_limited()
    loose = mt.CoilKernels.from_maps(maps, tolerance=1e-2)
    tight = mt.CoilKernels.from_maps(maps, tolerance=1e-8)
    assert loose.kernel_shape <= tight.kernel_shape
    assert loose.compression_ratio >= tight.compression_ratio


def test_maps_that_are_not_band_limited_are_refused_rather_than_truncated():
    """The one case where the saving is not worth a couple of decimal digits.

    A hard edge in the image has broad k-space support, so there is no kernel
    short of the whole grid, and answering with the whole grid would report a
    saving that is not one.
    """
    grid = 64
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, grid), torch.linspace(-1, 1, grid), indexing="ij"
    )
    masked = (_band_limited(grid=grid) * ((x**2 + y**2) < 0.6**2)).to(torch.complex64)
    with pytest.raises(ValueError, match="not band-limited"):
        mt.CoilKernels.from_maps(masked, tolerance=1e-4)


def test_from_maps_wants_a_shape_or_a_tolerance_and_not_both():
    maps = _band_limited()
    with pytest.raises(ValueError, match="not both"):
        mt.CoilKernels.from_maps(maps)
    with pytest.raises(ValueError, match="not both"):
        mt.CoilKernels.from_maps(maps, (8, 8), tolerance=1e-3)
