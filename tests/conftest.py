"""Shared fixtures.

Anything that decides a kernel's *layout* is parametrised over CPU and CUDA.
A layout check that ran on CPU only has, in this code's history, passed while
the CUDA path was completely wrong.
"""

import numpy as np
import pytest


def _cuda_available():
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


@pytest.fixture(
    params=[
        "cpu",
        pytest.param(
            "cuda",
            marks=[
                pytest.mark.cuda,
                pytest.mark.skipif(not _cuda_available(), reason="no CUDA device"),
            ],
        ),
    ]
)
def device(request):
    """Run the test on each device this machine actually has."""
    return request.param


@pytest.fixture
def radial():
    """A 2D radial trajectory as (shots, points, axes).

    Samples are in normalized k-space: -0.5 is grid location -kN/2 of a grid
    of size kN, and 0.5 is +kN/2. The same numbers therefore describe the
    image grid and the doubled grid the transfer lives on.
    """

    def build(n_spokes=48, n_samples=64):
        angles = np.linspace(0, np.pi, n_spokes, endpoint=False)
        radius = np.linspace(-0.5, 0.5, n_samples, endpoint=False)
        return np.stack(
            [np.outer(np.cos(angles), radius), np.outer(np.sin(angles), radius)],
            axis=-1,
        ).astype(np.float32)

    return build


@pytest.fixture
def operator(radial):
    """An MRI-NUFFT operator, for computing the reference Gram the slow way.

    Nothing in the package takes one of these any more; it is the yardstick.
    """
    mrinufft = pytest.importorskip("mrinufft")

    def build(shape=(32, 32), n_spokes=48, n_samples=64, density=None):
        return mrinufft.get_operator("finufft")(
            samples=radial(n_spokes, n_samples).reshape(-1, 2),
            shape=shape,
            density=density,
            n_coils=1,
            squeeze_dims=False,
        )

    return build


@pytest.fixture
def image():
    """A reproducible complex image."""

    def build(shape=(32, 32), seed=0):
        rng = np.random.default_rng(seed)
        return (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(
            np.complex64
        )

    return build


@pytest.fixture
def exact_gram():
    """``A^H A x`` computed the slow, unarguable way."""

    def build(op, x):
        shape = tuple(int(size) for size in op.shape)
        return np.asarray(op.adj_op(op.op(x.reshape(1, 1, *shape)))).reshape(shape)

    return build


@pytest.fixture
def relative_error():
    def measure(got, want):
        return float(np.linalg.norm(got - want) / np.linalg.norm(want))

    return measure
