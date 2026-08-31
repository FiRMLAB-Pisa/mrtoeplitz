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

Every builder takes a trajectory and an image shape. Nothing takes an MRI-NUFFT
operator: a NUFFT is needed to *grid* the point spread function, and which one
is the `backend` argument.

```python
import mrtoeplitz as mt

# trajectory: (shots, points, axes), samples in [-0.5, 0.5), unscaled by the grid
kernel = mt.scalar_kernel(trajectory, image_shape=(256, 256))
normal = kernel.apply(image[None, None])  # (batch, rank, *image_shape)
```

`kernel.apply` is `AᴴA`, and that is the only claim worth making about it: the
test suite checks it against `A_adjoint(A(x))` on real trajectories, on both
devices, because a variant that merely *carries* the right support, packing and
bound can still compute the wrong operator.

Through coil sensitivities:

```python
normal = mt.apply_sense(kernel, image, maps)  # maps or a CoilKernels bank
```

### Subspace

A subspace normal costs one gridding transform per basis **pair** —
`rank (rank + 1) / 2` of them — over every frame's samples at once. Not one per
frame, which for a fingerprinting scan is a thousand.

```python
kernel = mt.subspace_kernel(trajectory, basis, image_shape=(256, 256))
```

- `trajectory` is `(shots, points, axes)` when every frame shares one, or
  `(frames, shots, points, axes)` when they differ. Samples in the
  `[-0.5, 0.5)` units MRI-NUFFT expects, unscaled by the grid.
- `basis` is `(frames, rank)` or `(rank, frames)`, whichever way round. That
  axis is contrasts for a qMRI scan (MRF, FSE) and time for a dynamic one
  (cardiac); nothing here needs to know which.
- `density` is optional and broadcasts: `(points,)`, `(shots, points)` or
  `(frames, shots, points)`.

Frames acquired on the same trajectory are grouped before anything is gridded,
so a scan whose 1000 frames cycle through 8 rotations grids those 8 — not 1000
copies of them.

A Cartesian encoding needs no gridding and no doubled grid — the normal is the
sampling mask itself:

```python
kernel = mt.cartesian_subspace_kernel(masks, basis)
```

## Coil sensitivities as k-space kernels

A SENSE normal reads one coil at a time and never needs the whole bank, but
the bank is what gets stored: 48 channels at 320³ in complex single precision
is 12.6 GB. `CoilKernels` holds the centred k-space kernels instead and
expands one coil when the apply asks for it, so what is in flight is a single
map — which the apply already held.

```python
kernels = mt.CoilKernels.from_maps(maps, kernel_shape=(16, 16, 16))
kernels.compression_ratio  # 8000x for 320**3 -> 16**3
```

It stands in for the dense bank — `shape`, `ndim`, `dtype`, `device` and
leading-axis slicing all answer as the tensor would — so nothing in the apply
changes. Pass it where sensitivities go, on a holder with `shape` and `smaps`
(MRI-NUFFT's own operator validates that attribute as a NumPy array, so a
kernel bank travels beside it rather than through it).

### What it costs, measured

The representation is **exact for a band-limited map**: expand a random 12²
kernel to 64², truncate it back, and you recover the kernels to 1.0e-07 with a
round-trip error of 1.6e-07.

It is **not exact for a map that was formed first**. A real surface-coil
sensitivity does not vanish at the edge of the field of view, so under the
periodic transform it wraps, its spectrum falls off like one over frequency,
and truncation rings. Measured on a Biot-Savart four-loop array at 64²:

| kernel | whole FOV | inside r<0.3 | compression |
|---|---|---|---|
| 8² | 9.4e-02 | 5.4e-02 | 64× |
| 12² | 6.4e-02 | 3.5e-02 | 28× |
| 16² | 4.8e-02 | 2.6e-02 | 16× |
| 24² | 3.3e-02 | 1.8e-02 | 7× |

Staying inside the object only buys a factor of about 1.8 — the ringing
reaches it. Two ways to be in the exact case instead:

- **Take the kernels from the calibration and never form the map.** NLINV's
  maps are band-limited by construction — the Sobolev weighting *is* a band
  limit, and BART already stores them as k-space coefficients. For ESPIRiT,
  keep the calibration kernels and skip the eigenvector normalisation: the
  `|m| = 1` mask puts a hard edge in the image, which is the worst case in the
  table above.
- **Bring the map to zero at the boundary first.** The same Biot-Savart bank
  under a Hann taper truncates to 7.9e-04 at 16² — two orders better, because
  the error was the wrap and not the coil geometry.

`truncation_error(maps)` reports which case you are in, so none of this has to
be taken on trust.

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
