# Benchmarks

What one application of a multicoil subspace normal costs, against the
transform work it cannot avoid.

## What is measured

Three phases, on real data:

| phase | |
|---|---|
| `A^H y` | the zero-filled reconstruction a solve starts from |
| create | building the Toeplitz transfer |
| apply | one application of the normal operator |

for runtime, peak host memory and peak device memory. Each measurement runs in
its own process, so no phase is charged another's allocator. Host memory is the
kernel's `VmHWM`; device memory is total device occupancy against a baseline
taken before the process starts, because WSL2 reports nothing per process --
so it is what the process takes *from the card*, not its live working set.

Every lane is warmed before it is measured. The first application of any of
them compiles and autotunes CUDA kernels, which allocates hundreds of
megabytes that have nothing to do with the operator.

## The floor

The normal has to transform every coil's every coefficient onto the grid the
convolution runs on and back: `8 coils x rank 4 = 32` volumes, one forward and
one inverse each. The SENSE multiply, the transfer multiply and every copy are
counted as free. Nothing can beat this.

There are two ways to do that transform. The padded layout runs one pair on
the doubled grid; the parity decomposition never materialises that grid and
runs eight pairs on image-grid volumes instead -- the same cells, a smaller
transform each. The floor is the cheaper, and it is the parity one on both
devices:

| | one unit | x 32 volumes |
|---|---|---|
| CUDA, padded 512³ | 82 ms | 2636 ms |
| CUDA, parity 8x256³ | **75 ms** | **2410 ms** |
| CPU, padded 512³ | 1633 ms | 52241 ms |
| CPU, parity 8x256³ | **826 ms** | **26437 ms** |

## Results

Deli-CS six-minute acquisition: 500 frames x 48 shots x 1688 points, 48
channels compressed to 8, rank 4, on a 256³ grid, in low-memory mode. RTX 4060
Laptop, 8 GiB.

| | RAM | VRAM | `A^H y` | create | apply | vs floor |
|---|---|---|---|---|---|---|
| CUDA, floor | 692 MiB | 3185 MiB | — | — | 2410 ms | |
| CUDA, mrtoeplitz | 15074 MiB | 7913 MiB | 14521 ms | 31610 ms | **6898 ms** | **2.86x** |
| CPU, floor | 3577 MiB | — | — | — | 26437 ms | |
| CPU, mrtoeplitz | 18068 MiB | — | 52805 ms | 89131 ms | **61131 ms** | **2.31x** |

The transfer is 3.09 GiB and never resident: it is built onto the host and
streamed for every application, which is what keeps the device figure below
what the card holds.

The build's own working set is 3.9 GiB of Torch allocations, which matters
because the card is 8 GiB and cufinufft's plan takes about 2 GiB more outside
Torch's accounting. It was 8.35 GiB until the index arithmetic around the
transfer was walked in pieces and held in the narrowest types that carry it --
the coordinates a support location decodes to are int64 and there is one per
axis, which is several times the transfer they describe. Over the card the
WSL2 driver does not fail: it satisfies the overflow from host memory, and one
512³ FFT goes from 0.09 s to 7.3 s while nothing reports an error.

Runtimes on this machine are not reproducible to better than about a factor of
two for the build -- the same build has measured 16 to 45 seconds at a smaller
sample count -- so the memory is the part to trust. The apply is the stable
one, and it is the phase a solve repeats.

`A^H y` is not part of the package. It lives in `lane_mrtoeplitz.py` because a
benchmark needs somewhere to start, and it will move into a reconstruction API
rather than this one.

## Running it

```bash
python benchmarks/run.py          # the table
python benchmarks/figure.py       # the figure the top-level README carries
```

The data is fetched from Zenodo on first use (see `delics.py`) and prepared
once by `prepare.py`. Coil sensitivities are estimated separately with BART's
NLINV over the k-space centre and expanded onto the reconstruction grid; none
of that is inside a measurement.

## Reading the raw acquisition

`raw_mrf.npy` is not published with its layout written down, and two things
about it have to be established rather than assumed. `prepare.py` derives both
and refuses to continue if they do not hold:

- **Which arm is which.** The flip-angle train varies over the 500 frames and
  repeats for each of the 48 groups, so the arm axis is periodic in the frame
  with period 500. Folded that way the average is the train; folded the other
  way it is flat -- 0.456 against 0.003.
- **Where the readout starts.** Every arm is centre-out and the trajectory's
  first point is `k = 0`, so the largest sample of a readout is its first.

Neither can be settled by scoring a reconstruction. A misaligned gridding puts
every arm's centre sample at `k = 0` and scatters the rest, which is a bright
point at the origin surrounded by streaks -- more concentrated, and sharper by
any edge measure, than a real image. Every metric that rewards concentration
picks it, and picking it produces an array of the right shape that is not an
image of anything.

## What is not here

Lanes for BART and MRISubspaceRecon.jl exist (`lane_bart.py`,
`julia/lane.jl`) and neither is reported. BART needs hour-scale budgets per
phase at 256³ here, and MRISubspaceRecon's CUDA extension does not compile for
a complex basis (`kernel_mul!` in `ext/MRISubspaceReconCUDAExt/NFFTNormalOp.jl`
reads undefined names).

Any comparison made through `lane_bart.py` before its `write_cfl` was fixed is
void: it wrote C-ordered bytes under a header stating BART's own order, so
every array reached BART transposed. That includes an earlier reading of what
`compress-psf` retains against what this package retains, which is withdrawn.
