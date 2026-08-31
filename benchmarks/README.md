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
transform each. The floor is the cheaper, and which is cheaper depends on the
device:

| | one unit | x 32 volumes |
|---|---|---|
| CUDA, padded 512³ | 69.9 ms | 2237 ms |
| CUDA, parity 8x256³ | 67.2 ms | **2149 ms** |
| CPU, padded 512³ | 1093 ms | 34988 ms |
| CPU, parity 8x256³ | 657 ms | **21020 ms** |

The parity layout is 4% cheaper on CUDA and **1.66x** cheaper on CPU. Its
memory advantage is the same on both; its runtime advantage is not.

## Results

Deli-CS, 500 frames x 8 shots x 1688 points, 48 channels compressed to 8,
rank 4, 256³, in low-memory mode. RTX 4060 Laptop, 8 GiB.

### CUDA

| | RAM | VRAM | `A^H y` | create | apply |
|---|---|---|---|---|---|
| floor | 643 MiB | 3185 MiB | — | — | 2149 ms |
| mrtoeplitz | 12828 MiB | 7759 MiB | 13277 ms | 88831 ms | **5143 ms** |

**The apply runs at 2.39x the floor.** The transfer is built once and applied
every iteration, so the 89 s build is amortised over a solve; what a solve
feels is the 5.1 s.

### CPU

| | RAM | VRAM | `A^H y` | create | apply |
|---|---|---|---|---|---|
| floor | 3578 MiB | — | — | — | 21020 ms |
| mrtoeplitz | 15462 MiB | — | 42668 ms | 83986 ms | **61506 ms** |

**2.93x the floor.**

Device memory is the number to watch: 7759 MiB of an 8188 MiB card, against a
floor that already needs 3185 MiB for the transforms alone. The operator fits,
and it fits with little to spare.

## Running it

```bash
python benchmarks/run.py --sizes 256 --devices cuda,cpu --lanes floor,mrtoeplitz
```

The data is fetched from Zenodo on first use (see `delics.py`), prepared once
by `prepare.py`, and the coil maps are estimated once at the reconstruction
size. None of that is inside a measurement.

## What is not here

Lanes for BART and MRISubspaceRecon.jl are written and working
(`lane_bart.py`, `julia/lane.jl`), configured to match: BART with
`--nufft-conf real-psf,compress-psf,decomposed-psf,upper-triag-psf` and
`-U`, and Julia with the real basis that selects its real-only NUFFT. They are
not reported because neither produced a defensible timing here -- BART needs
well over four minutes per phase at 64³ and hour-scale budgets at 256³, and
MRISubspaceRecon's CUDA extension does not compile for a complex basis
(`kernel_mul!` in `ext/MRISubspaceReconCUDAExt/NFFTNormalOp.jl` reads
undefined names). Reporting numbers from a lane that was still being debugged
would be worse than reporting none.

One measurement from that work does stand, because file sizes do not depend on
contention. At 64³ with rank 4, where a dense transfer is 268,435,456 bytes:

| | transfer |
|---|---|
| BART, default | 268,435,456 B |
| BART, `upper-triag-psf` | 167,772,160 B |
| BART, `compress-psf` | **38,515,712 B** |
| mrtoeplitz, low-memory default | 63,132,608 B |

BART's `upper-triag-psf` figure is exactly `128³ x 10 x 8`, so it packs the
same `rank (rank + 1) / 2` pairs. Its compression keeps less than ours does on
this trajectory -- 36.7 MiB against 60.2 MiB -- which is worth understanding
before claiming anything about relative footprint.
