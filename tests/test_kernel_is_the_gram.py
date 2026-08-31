"""The only evidence that counts for a Toeplitz variant.

Every check here is ``fast(x)`` against ``A_adjoint(A(x))`` on a real
trajectory. A variant that merely *carries* the right support, packing and
bound can still compute the wrong operator; only this says it does not.
"""

import math

import numpy as np
import pytest
import torch

import mrtoeplitz as mt


def _apply(kernel, x, device):
    tensor = torch.as_tensor(x)[None, None].to(device)
    return np.asarray(kernel.to(device).apply(tensor).detach().cpu()).reshape(x.shape)


#: What a kernel agrees with the exact Gram to, in single precision. The CUDA
#: default is bfloat16, which is two decimal digits coarser -- see
#: ``test_the_cuda_transfer_defaults_to_bfloat16_on_a_capable_device``. Every
#: check of *what operator is computed* pins float32 so the two devices are
#: held to one number.
SINGLE = mt.toeplitz_options(compress=False, cuda_transfer_precision="float32")


def test_the_scalar_kernel_reproduces_the_exact_gram(
    radial, operator, image, exact_gram, relative_error, device
):
    op = operator()
    x = image()
    kernel = mt.scalar_kernel(radial(), (32, 32), options=SINGLE)
    assert relative_error(_apply(kernel, x, device), exact_gram(op, x)) < 1e-4


def test_a_density_weighted_kernel_reproduces_its_own_gram(
    operator, radial, image, exact_gram, relative_error
):
    # Density inside the normal is the intended acceleration (Pruessmann
    # CG-SENSE), not a defect: the adjoint applies it once, so the Gram does.
    trajectory = radial()
    density = (np.linalg.norm(trajectory, axis=-1) + 1e-3).astype(np.float32)
    op = operator(density=density.reshape(-1))
    x = image()
    kernel = mt.scalar_kernel(trajectory, (32, 32), density=density, options=SINGLE)
    assert relative_error(_apply(kernel, x, "cpu"), exact_gram(op, x)) < 1e-4


def test_the_transfer_is_built_on_the_doubled_grid(radial):
    """A point spread function on N covers displacements to +-N/2; the Gram
    needs +-(N - 1), so the transfer lives on 2N."""
    kernel = mt.scalar_kernel(radial(), (32, 32))
    assert kernel.spatial_shape == (64, 64)
    assert kernel.image_shape == (32, 32)


def test_a_fully_sampled_radial_disk_keeps_pi_over_four_of_the_grid(radial):
    """What compression saves on a radial scan is geometry, not tuning.

    A fully sampled 2D radial acquisition fills the disk inscribed in the
    square that encloses it, so the kept fraction is the ratio of their areas,
    pi/4. The measured fraction sits above that by the interpolation rim --
    a fixed number of cells on a disk whose radius is not fixed -- so it
    approaches pi/4 from above as the grid grows.
    """
    kept = {}
    for n in (64, 128):
        kernel = mt.scalar_kernel(
            radial(n_spokes=round(math.pi / 2 * n), n_samples=2 * n),
            (n, n),
            options=mt.toeplitz_options(compress=True),
        )
        kept[n] = kernel.n_locations / (2 * n) ** 2

    assert all(fraction > math.pi / 4 for fraction in kept.values())
    assert kept[128] < kept[64]
    # The rim's share halves as the grid doubles, so the excess does too.
    assert (kept[128] - math.pi / 4) < 0.6 * (kept[64] - math.pi / 4)
    assert kept[128] < 0.85


def test_compression_costs_accuracy_that_widening_the_rim_does_not_recover(
    radial, operator, image, exact_gram, relative_error
):
    """What compression drops is not the interpolation rim.

    The transfer is not the gridded trajectory: it carries genuine weight
    outside the sampled region, falling off like one over distance. So a
    compressed kernel's error reaches a floor as the rim widens rather than
    returning to the uncompressed answer.
    """
    trajectory = radial()
    op = operator()
    x = image()
    truth = exact_gram(op, x)

    whole = mt.scalar_kernel(trajectory, (32, 32), options=SINGLE)
    cut = mt.scalar_kernel(
        trajectory, (32, 32), options=mt.toeplitz_options(compress=True)
    )

    assert relative_error(_apply(whole, x, "cpu"), truth) < 1e-4
    assert relative_error(_apply(cut, x, "cpu"), truth) > 1e-4
    assert cut.n_locations < whole.n_locations


def test_compression_records_what_it_left_out(radial):
    cut = mt.scalar_kernel(
        radial(), (32, 32), options=mt.toeplitz_options(compress=True)
    )
    assert cut.truncation_bound > 0.0
    whole = mt.scalar_kernel(
        radial(), (32, 32), options=mt.toeplitz_options(compress=False)
    )
    assert whole.truncation_bound == 0.0


def test_the_polyphase_layout_computes_the_same_operator(
    radial, operator, image, exact_gram, relative_error, device
):
    """The layout check that must run on both devices.

    Sharing scratch between polyphase components is the memory win; sharing
    the components' own transfers by mistake gives an operator that is 1.47
    relative error wrong on CUDA and right on CPU.
    """
    op = operator()
    x = image()
    truth = exact_gram(op, x)
    padded = mt.scalar_kernel(
        radial(),
        (32, 32),
        options=mt.toeplitz_options(**{**SINGLE, "polyphase": False}),
    )
    parity = mt.scalar_kernel(
        radial(), (32, 32), options=mt.toeplitz_options(**{**SINGLE, "polyphase": True})
    )

    assert isinstance(parity, mt.PolyphaseToeplitzKernel)
    assert relative_error(_apply(padded, x, device), truth) < 1e-4
    assert relative_error(_apply(parity, x, device), truth) < 1e-4
    assert relative_error(_apply(parity, x, device), _apply(padded, x, device)) < 1e-5


def test_a_polyphase_kernel_files_one_component_per_parity(radial):
    parity = mt.scalar_kernel(
        radial(), (32, 32), options=mt.toeplitz_options(compress=False, polyphase=True)
    )
    # Two dimensions, so four parities, each on the image grid rather than
    # the doubled one: that is what keeps the doubled grid unmaterialised.
    assert len(parity.components) == 4
    for _parity, component in parity.components:
        assert component.spatial_shape == (32, 32)


@pytest.mark.cuda
def test_the_cuda_transfer_defaults_to_bfloat16_on_a_capable_device(
    radial, operator, image, exact_gram, relative_error
):
    """The default costs two decimal digits, and is worth knowing about.

    ``cuda_transfer_precision="auto"`` narrows the transfer to bfloat16 wherever
    the device supports it natively, which halves what a large kernel occupies.
    Anyone comparing a CUDA result against a CPU one is comparing bfloat16
    against float32 unless they say otherwise.
    """
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("no native bfloat16")

    op = operator()
    x = image()
    truth = exact_gram(op, x)

    default = mt.scalar_kernel(
        radial(), (32, 32), options=mt.toeplitz_options(compress=False)
    )
    single = mt.scalar_kernel(radial(), (32, 32), options=SINGLE)

    coarse = relative_error(_apply(default, x, "cuda"), truth)
    fine = relative_error(_apply(single, x, "cuda"), truth)

    assert fine < 1e-4
    assert 1e-4 < coarse < 1e-2
    assert default.last_cuda_mode is not None


def test_samples_are_read_on_the_half_open_minus_half_to_half_scale():
    """The convention every entry point of this package takes.

    ``occupancy_indices`` is public and reads a trajectory onto the transfer
    grid. Reading it on MRI-NUFFT's internal ``[-pi, pi)`` scale instead would
    shrink the support by ``(2 pi)`` per axis, and a compressed kernel would
    silently keep a fortieth of what it should while still passing every
    uncompressed accuracy check.
    """
    grid = (32, 32)
    edge = np.linspace(-0.5, 0.5, 64, endpoint=False, dtype=np.float32)
    whole = np.stack(np.meshgrid(edge, edge, indexing="ij"), axis=-1).reshape(-1, 2)
    assert mt.occupancy_indices(whole, grid, width=0).numel() == 32 * 32

    quarter = np.stack(np.meshgrid(edge / 2, edge / 2, indexing="ij"), axis=-1).reshape(
        -1, 2
    )
    kept = mt.occupancy_indices(quarter, grid, width=0).numel() / (32 * 32)
    assert 0.2 < kept < 0.3
