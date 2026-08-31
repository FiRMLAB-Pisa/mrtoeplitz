# mrtoeplitz

Memory-efficient Toeplitz normal operators for MRI reconstruction: scalar and
subspace, on CPU and CUDA.

[![Tests](https://github.com/FiRMLAB-Pisa/mrtoeplitz/actions/workflows/test-ci.yml/badge.svg)](https://github.com/FiRMLAB-Pisa/mrtoeplitz/actions/workflows/test-ci.yml)
[![codecov](https://codecov.io/gh/FiRMLAB-Pisa/mrtoeplitz/branch/main/graph/badge.svg)](https://codecov.io/gh/FiRMLAB-Pisa/mrtoeplitz)
[![PyPI](https://img.shields.io/pypi/v/mrtoeplitz.svg)](https://pypi.org/project/mrtoeplitz/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A normal operator `AᴴA` for a non-Cartesian encoding is a convolution, so it
can be applied as a pointwise multiply between two FFTs instead of a forward
and adjoint NUFFT. What it costs is memory: the transfer lives on a grid twice
the image in every dimension, and for a high-resolution subspace
reconstruction that is the whole device.

This package is about that cost. The transfer is stored only where the scan
put weight, filed by coordinate parity so the doubled grid is never
materialised, and applied out of a bank that is reused rather than
reallocated.

## Install

```bash
pip install mrtoeplitz          # applying a kernel: Torch only
pip install mrtoeplitz[nufft]   # building one: grids a PSF, so needs a NUFFT
pip install mrtoeplitz[cuda]    # CUFINUFFT, for building on the GPU
```

## Use

```python
import mrinufft
import torch

import mrtoeplitz as mt

operator = mrinufft.get_operator("finufft")(
    samples=trajectory, shape=(256, 256), density=None, n_coils=1, squeeze_dims=False
)

kernel = mt.scalar_kernel(operator)
normal = kernel.apply(image[None, None])  # (batch, rank, *image_shape)
```

`kernel.apply` is `AᴴA`, and that is the only claim worth making about it: the
test suite checks it against `A_adjoint(A(x))` on real trajectories, on both
devices, because a variant that merely *carries* the right support, packing and
bound can still compute the wrong operator.

### Subspace

A subspace normal costs one gridding transform per basis **pair** —
`rank (rank + 1) / 2` of them — over every frame's samples concatenated. Not
one per frame, which for a fingerprinting scan is a thousand.

You group the frames onto the distinct trajectories they were acquired on,
because only you know which frames share a plan:

```python
rows, columns = torch.triu_indices(rank, rank)
blocks = [
    # (samples, density, upper-triangular basis products summed over the
    #  frames that share this trajectory)
    (samples_t, None, basis[rows, t] * basis[columns, t].conj())
    for t, samples_t in enumerate(trajectories)
]
kernel = mt.subspace_kernel(blocks, image_shape=(256, 256))
```

A Cartesian encoding needs no gridding and no doubled grid — the normal is the
sampling mask itself:

```python
kernel = mt.cartesian_subspace_kernel(masks, basis)
```

## What the options actually change

Nothing decides how the kernel is *built*. It is gridded onto the doubled grid
the way BART's `compute_psf_int` and MRFingerprintingRecon.jl's
`calculate_kernel_noncartesian` build theirs, and that is not a choice: the
exact transfer — the transform of the analytic point spread function — is
dense, with Dirichlet tails putting 84% of peak in the corners of the cube even
for a strictly ball-supported trajectory, and cannot be truncated.

What is left is what it is stored and executed on:

| option | what it does |
|---|---|
| `compress` | BART's `--compress-psf`. Keeps the transfer where the gridded trajectory is non-zero and drops it outside. What that saves is geometry: a fully sampled 2D radial disk fills the disk inscribed in the square that encloses it, so it keeps **π/4 ≈ 79%** and there is no more to be had. A 3D koosh ball keeps 27.7%, which is where compression earns its keep. |
| `polyphase` | Files the transfer as one component per parity of the doubled grid, so the convolution runs on the image grid. `"auto"` picks by what the device can hold. |
| `cuda_transfer_precision` | `"auto"` narrows to **bfloat16** wherever the device supports it natively, which halves what a large kernel occupies and costs about two decimal digits. Pass `"float32"` when comparing against a CPU result. |
| `coil_batch_size`, `chunk_size`, `cuda_mode` | How much is unpacked and how many coils share a pass. |

Compression is not free: it makes an already-indefinite normal considerably
more so, and a conjugate-gradient solve over a compressed kernel must be able
to step through negative curvature — as BART's `italgos.c` does — or it
freezes.

Nor is what it drops the interpolation rim. The transfer is not the gridded
trajectory: it has genuine weight outside the sampled region, falling off like
one over distance, so widening the rim reaches a floor rather than the
uncompressed answer. Measured on a fully sampled 2D radial disk at N=64, where
the uncompressed kernel agrees with `AᴴA` to 8.6e-06:

| rim `w` | kept | relative error |
|---|---|---|
| 1 | 0.809 | 5.3e-03 |
| 2 | 0.833 | 3.8e-03 |
| 4 (default) | 0.874 | 2.8e-03 |
| 6 | 0.908 | 2.4e-03 |

The kept fraction approaches π/4 from above as the grid grows, because the rim
is a fixed number of cells on a disk whose radius is not: 0.932 at N=32, 0.874
at N=64, 0.835 at N=128, 0.812 at N=256.

## Development

```bash
pip install -e .[dev]
bash scripts/format_and_lint.sh
pytest -q
```

See [CONTRIBUTING.md](CONTRIBUTING.md).
