# Gridding the transfer without the doubled grid

The parity layout keeps the *apply* off the doubled grid, but the build still
grids the point spread function onto `(2N)^d` and decomposes afterwards. At
256³ that is where the memory goes: Torch's peak live allocation during a
build is 6722 MiB, and each doubled-grid buffer is `512³ × 8 = 1073 MiB`.

## What the build does now

```
psf      = type1(samples, weights, modes = 2N centred)     # cufinufft
transfer = fftn(ifftshift(psf)) / (2N)^d
```

## The decomposition

Write the transfer's index as `k = 2m + p`, with `p ∈ {0,1}^d` the parity.
Splitting the sum over the shifted PSF's two halves,

    H[2m + p] = (1 / (2N)^d) · FFT_N( e^{-iπ p·j/N} · (psf[j] + (-1)^p psf[j-N]) )

for `j ∈ [0, N)`, where `psf` is indexed on the centred doubled grid. The fold
never needs the doubled array, because it can be gridded directly:

    psf[j] + (-1)^p psf[j-N] = Σ_s w_s · e^{i j·2π x_s} · (1 + (-1)^p e^{-2πi N x_s})

In `d` dimensions the trailing factor is a product over axes, one factor per
axis, each depending only on that axis's parity bit and coordinate.

## The wrinkle

CUFINUFFT answers on *centred* modes, `[-N/2, N/2)`, and the fold above wants
`[0, N)`. A shift of the mode index by `N/2` is a modulation of the samples,
so the weights carry one more factor:

    w_s  ->  w_s · e^{iπ N x_s} · Π_a (1 + (-1)^{p_a} e^{-2πi N x_{s,a}})

Each component is then one type-1 onto `N^d`, a separable phase ramp
`e^{-iπ p·j/N}` on the result, and `fftn` on `N^d`.

## What it costs and buys

Buys: the working grid drops from `(2N)^d` to `N^d`, a factor of `2^d` -- 8 in
three dimensions. The 1073 MiB buffers become 134 MiB, and CUFINUFFT's own
upsampled grid shrinks with them.

Costs: `2^d` gridding passes instead of one, each spreading every sample.
Spreading is the dominant cost of a type-1 and scales with the sample count
and the kernel width rather than with the grid, so the gridding time is
expected to rise by close to `2^d`. Whether that lands is a measurement, not a
prediction: at 192³ gridding is about 20% of a 7.3 s build, so an eightfold
rise would roughly double it.

The obvious way to claw some back is CUFINUFFT's `gpu_stream` option, which
binds a plan to a caller-owned CUDA stream. Two plans on two streams let one
component's spreading overlap the next one's transform, in the same way the
apply already overlaps its host staging.

## Before implementing

- The components must reproduce the current build exactly, on a trajectory
  where the exact Gram is computable. That is the acceptance test, and it
  should be written first.
- `_centring_signs` and the support selection both currently work in
  doubled-grid coordinates and will need their per-component equivalents.
- Two plans double the largest allocation a build makes, so the streaming
  variant only pays where the staging is to host.
