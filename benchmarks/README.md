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
| CUDA, padded 384³ | 1334–1357 ms, or 1020 ms | |
| CUDA, parity 8x192³ | **1017–1020 ms** | |
| CPU, padded 512³ | 1093 ms | 34988 ms |
| CPU, parity 8x256³ | **657 ms** | 21020 ms |

At 192³ on CUDA the parity layout is steady at 1019 ms every time, while the
padded one is usually 1.32x slower and occasionally lands at exactly parity's
speed -- four runs gave 1.31x, 1.32x, 1.33x and 1.00x. That is cuFFT choosing
differently for the awkward 384³ depending on the workspace it finds. So the
decomposition is not only cheaper on average but predictable, which is the
better property of the two. On CPU it is **1.66x** cheaper outright.

## Results

Deli-CS, 500 frames x 8 shots x 1688 points, 48 channels compressed to 8,
rank 4, 256³, in low-memory mode. RTX 4060 Laptop, 8 GiB.

### CUDA

| size | | VRAM | `A^H y` | create | apply | vs floor |
|---|---|---|---|---|---|---|
| 192³ | floor | 1409 MiB | — | — | 1019 ms | |
| 192³ | mrtoeplitz | 4618 MiB | 6240 ms | 7347 ms | **2371 ms** | **2.33x** |
| 256³ | floor | 3185 MiB | — | — | 2103 ms | |
| 256³ | mrtoeplitz | 7925 MiB | 13971 ms | 19097 ms | **5609 ms** | **2.67x** |

The apply replicates at both sizes: 2365.1 and 2365.7 ms at 192³, and 5609.2,
5621.0 and 5608.9 ms at 256³. The build does not, ranging 19 to 40 seconds at
256³ across the same three runs; it is paid once and amortised over a solve,
but no single figure for it is quoted.

That 256³ column only became measurable when the gridding moved to a 1.25
working grid. Before it, the same work ranged 5143 to 55828 ms across runs
differing only in where the transfer was built and whether it was streamed:
CUFINUFFT's default upsampling put 3962 MiB in its own working grid, the card
had nothing left to give, and what the apply measured was how much room it
found. At 1.25 the plan is 1338 MiB and the spread is gone.

VRAM here is peak *occupancy*, not the live set. Most of what is above the
transfer and the volumes is Torch's caching allocator holding what the build
grew, which it returns when asked, and a workspace CUFINUFFT keeps for the
life of the process -- one-time, identical across four repeated builds, and
not reclaimed by destroying the plan.

### CPU

| | RAM | VRAM | `A^H y` | create | apply |
|---|---|---|---|---|---|
| floor | 3578 MiB | — | — | — | 21020 ms |
| mrtoeplitz | 15462 MiB | — | 42668 ms | 83986 ms | **61506 ms** |

**2.93x the floor**, and stable: nothing here is near a limit.

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

One measurement from that work looked comparable and is not. At 64³ BART's
`compress-psf` keeps 38,515,712 bytes where our low-memory default keeps
63,132,608, but the two are compressing different regions: a sample at 0.5
lands on the last cell of the transfer grid here and on the middle one in the
BART staging, so ours retains the ball the trajectory fills -- 53.7% of the
grid, against the 52.4% a sphere occupies in its cube -- and BART's retains a
ball of half that radius. Ours is right for its own convention and cannot be
tightened: a koosh-ball genuinely reaches half the doubled grid. What BART's
number means depends on how it maps `-t` units onto its internal grid, which
is not established here, so no conclusion is drawn from the pair.
