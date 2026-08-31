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
| floor | 643 MiB | 3185 MiB | — | — | 2103–2149 ms |
| mrtoeplitz | 10733–12828 MiB | 7759–7879 MiB | 12090–13277 ms | 16786–88831 ms | **5143–27949 ms** |

**The apply is not reproducible at this size and no ratio to the floor is
quoted from it.** Three runs of the same work gave 5143, 17992 and 27949 ms
while device memory crept from 7759 to 7879 MiB of an 8188 MiB card. The
operator is at the edge of the card, so what varies is how much room the apply
finds, not what it computes. Building the transfer on the device rather than
on the host and moving it makes creation 4.6x faster (88.8 s to 19.5 s) and is
the right way to build it; it does not account for the spread in the apply,
and neither does releasing the allocator between the two, which was tried and
made no difference.

At 128³, where there is room, the same comparison is stable and the two ways
of building are indistinguishable in the apply: 428 ms host-built against
407 ms device-built, identical storage, dtype and lane.

A trustworthy number at 256³ needs either a card this configuration fits
inside with room to spare, or less pressure on this one -- fewer coils
resident, or sensitivities held as kernels rather than as maps.

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
