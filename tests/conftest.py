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
def exact_gram():
    """``A^H A x`` computed with raw FINUFFT, as the yardstick.

    ``A`` is a type-2 transform with ``isign=-1`` and ``A^H`` a type-1 with
    ``isign=+1`` -- the standard unnormalised adjoint pair -- divided by the
    size of the doubled grid the convolution runs on, which is the whole
    normalization the package uses. Nothing here comes from the package, so a
    kernel agreeing with it is evidence rather than self-consistency.
    """
    finufft = pytest.importorskip("finufft")

    def build(trajectory, image_shape, image, density=None):
        points = np.ascontiguousarray(
            np.asarray(trajectory, dtype=np.float64).reshape(-1, len(image_shape))
            * 2
            * np.pi
        )
        columns = [
            np.ascontiguousarray(points[:, axis]) for axis in range(points.shape[1])
        ]

        forward = finufft.Plan(2, image_shape, isign=-1, eps=1e-9, dtype="complex128")
        forward.setpts(*columns)
        adjoint = finufft.Plan(1, image_shape, isign=+1, eps=1e-9, dtype="complex128")
        adjoint.setpts(*columns)

        measured = forward.execute(np.asarray(image, dtype=np.complex128))
        if density is not None:
            measured = measured * np.asarray(density, dtype=np.float64).reshape(-1)
        gridded = adjoint.execute(measured)
        return gridded / np.prod([2 * size for size in image_shape])

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
def relative_error():
    def measure(got, want):
        return float(np.linalg.norm(got - want) / np.linalg.norm(want))

    return measure
