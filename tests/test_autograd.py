"""The normal operator as a differentiable function of the image.

A normal is a Gram, so it is Hermitian and its own adjoint: the gradient is
one more application and nothing from the forward pass is kept. That is not a
cheaper alternative to tracing the forward pass -- the lanes that make a large
transfer fit transform into a reused buffer with ``out=``, which Torch cannot
differentiate through at all.
"""

import numpy as np
import pytest
import torch

import mrtoeplitz as mt

WHOLE = mt.toeplitz_options(compress=False, cuda_transfer_precision="float32")


def _dense(operator, shape):
    """The operator as an explicit matrix, one basis vector at a time."""
    n = int(np.prod(shape))
    matrix = torch.zeros(n, n, dtype=torch.complex64)
    for column in range(n):
        unit = torch.zeros(n, dtype=torch.complex64)
        unit[column] = 1.0
        matrix[:, column] = operator(unit.reshape(1, 1, *shape)).reshape(-1)
    return matrix


@pytest.fixture
def small(radial):
    """A transfer small enough to write out as a matrix."""
    return mt.scalar_kernel(radial(n_spokes=12, n_samples=16), (8, 8), options=WHOLE)


def test_a_kernel_is_called_not_applied(small):
    """``apply`` is taken by ``torch.nn.Module``; calling is the entry point."""
    image = torch.randn(1, 1, 8, 8, dtype=torch.complex64)
    assert callable(small)
    assert small(image).shape == image.shape
    assert not hasattr(small, "apply")


def test_the_normal_operator_is_hermitian(small):
    matrix = _dense(small, (8, 8))
    asymmetry = torch.linalg.vector_norm(matrix - matrix.conj().T)
    assert float(asymmetry / torch.linalg.vector_norm(matrix)) < 1e-6


def test_the_gradient_is_the_adjoint_applied_to_it(small):
    """What the backward pass must equal, checked against the matrix itself."""
    matrix = _dense(small, (8, 8))
    image = torch.randn(1, 1, 8, 8, dtype=torch.complex64, requires_grad=True)
    seed = torch.randn(1, 1, 8, 8, dtype=torch.complex64)

    small(image).backward(seed)
    expected = (matrix.conj().T @ seed.reshape(-1)).reshape(1, 1, 8, 8)
    error = torch.linalg.vector_norm(image.grad - expected)
    assert float(error / torch.linalg.vector_norm(expected)) < 1e-5


def test_the_backward_pass_keeps_nothing_from_the_forward_one(small):
    """The saving: no intermediate of the forward pass is held for backward."""
    image = torch.randn(1, 1, 8, 8, dtype=torch.complex64, requires_grad=True)
    result = small(image)
    saved = [name for name in dir(result.grad_fn) if name.startswith("_saved_")]
    assert saved == []


def test_a_polyphase_kernel_is_differentiable_too(radial):
    kernel = mt.scalar_kernel(
        radial(n_spokes=12, n_samples=16),
        (8, 8),
        options=mt.toeplitz_options(**{**WHOLE, "polyphase": True}),
    )
    assert isinstance(kernel, mt.PolyphaseToeplitzKernel)
    image = torch.randn(1, 1, 8, 8, dtype=torch.complex64, requires_grad=True)
    kernel(image).abs().sum().backward()
    assert image.grad is not None
    assert torch.isfinite(image.grad.abs()).all()


def test_a_subspace_kernel_is_differentiable(radial):
    trajectory = radial(n_spokes=12, n_samples=16)
    basis = np.random.default_rng(0).normal(size=(2, 4)).astype(np.float32)
    kernel = mt.subspace_kernel(trajectory, basis, (8, 8), options=WHOLE)
    image = torch.randn(1, 2, 8, 8, dtype=torch.complex64, requires_grad=True)
    kernel(image).abs().sum().backward()
    assert image.grad.shape == image.shape


def test_the_sense_normal_is_differentiable(small):
    maps = torch.randn(4, 8, 8, dtype=torch.complex64)
    image = torch.randn(1, 1, 8, 8, dtype=torch.complex64, requires_grad=True)
    seed = torch.randn(1, 1, 8, 8, dtype=torch.complex64)

    mt.apply_sense(small, image, maps).backward(seed)
    grad = image.grad.clone()

    # The SENSE normal is Hermitian too, so applying it to the seed is the
    # gradient -- computed here the other way round as the check.
    with torch.no_grad():
        expected = mt.apply_sense(small, seed, maps)
    error = torch.linalg.vector_norm(grad - expected)
    assert float(error / torch.linalg.vector_norm(expected)) < 1e-5


def test_explicit_factors_refuse_a_gradient_rather_than_return_a_wrong_one(small):
    factor = torch.ones(1, 1, 8, 8, dtype=torch.complex64)
    image = torch.randn(1, 1, 8, 8, dtype=torch.complex64, requires_grad=True)
    with pytest.raises(ValueError, match="own adjoint"):
        mt.apply_sense(small, image, None, right_factors=factor)


def test_a_gradient_flows_through_a_coil_kernel_bank(small, radial):
    torch.manual_seed(0)
    seed = torch.randn(4, 6, 6, dtype=torch.complex64)
    kernels = mt.CoilKernels(seed, (8, 8))
    image = torch.randn(1, 1, 8, 8, dtype=torch.complex64, requires_grad=True)
    mt.apply_sense(small, image, kernels).abs().sum().backward()
    assert image.grad is not None
    assert torch.isfinite(image.grad.abs()).all()
