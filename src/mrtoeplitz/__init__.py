"""Memory-efficient Toeplitz normal operators for MRI reconstruction.

A normal operator ``A^H A`` for a non-Cartesian encoding is a convolution, so
it can be applied as a pointwise multiply between two FFTs instead of a
forward and adjoint NUFFT. What it costs is memory: the transfer lives on a
grid twice the image in every dimension, and for a high-resolution subspace
reconstruction that is the whole device.

This package is about that cost. The transfer is stored only where the scan
put weight, filed by coordinate parity so the doubled grid is never
materialised, and applied out of a bank that is reused rather than reallocated.

Building a kernel needs a NUFFT (the ``nufft`` extra). Applying one needs
nothing but Torch.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _distribution_version

from ._build import (
    cartesian_subspace_kernel,
    scalar_kernel,
    subspace_kernel,
    toeplitz_options,
)
from ._coils import CoilKernels
from ._kernel import (
    CompactToeplitzKernel,
    PolyphaseToeplitzKernel,
    as_torch,
    occupancy_indices,
    polyphase_components,
    significant_indices,
    support_indices,
)
from ._sense import apply_sense
from ._streaming import CudaStreaming

try:
    __version__ = _distribution_version(__name__)
except PackageNotFoundError:  # a source tree that was never installed
    __version__ = "0.0.0.dev0"

__all__ = [
    "CoilKernels",
    "CompactToeplitzKernel",
    "CudaStreaming",
    "PolyphaseToeplitzKernel",
    "__version__",
    "apply_sense",
    "as_torch",
    "cartesian_subspace_kernel",
    "occupancy_indices",
    "polyphase_components",
    "scalar_kernel",
    "significant_indices",
    "subspace_kernel",
    "support_indices",
    "toeplitz_options",
]
