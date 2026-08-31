# Benchmarks

Peak memory and runtime for the subspace normal, against:

1. **FFT floor** — pure cuFFT of the same grids, the irreducible cost.
2. **BART** — `pics --nufft-conf compress-psf`.
3. **MRFingerprintingRecon.jl** — allocates the doubled grid whole, with no
   parity decomposition.
4. **torchkbnufft** — the Torch-native reference point.

## The data

[`delics.py`](delics.py) fetches the Deli-CS dataset from Zenodo the first
time a benchmark asks for it, and never again. Nothing is committed here.

```bash
python benchmarks/delics.py            # fetch everything the benchmark uses
python benchmarks/delics.py shared     # or just one dataset
```

| name | record | file | size |
|---|---|---|---|
| `shared` | [7734431](https://zenodo.org/records/7734431) | `shared.tar.gz` | 0.98 GiB |
| `raw` | [7697373](https://zenodo.org/records/7697373) | `val_case000.tar.gz` | 2.89 GiB |
| `bart` | [7734431](https://zenodo.org/records/7734431) | `bartcompare.tar.gz` | 1.17 GiB |

`shared` carries the trajectories, density weights, subspace basis and
dictionary; `raw` is one acquisition's k-space; `bart` is BART's own
reconstruction of the two-minute scan, as a correctness anchor.

Sizes and checksums are read from the record at run time rather than written
down here, so a revised record is noticed rather than silently accepted. A
download that is interrupted resumes; one that fails its MD5 is deleted, so
the next run refetches instead of reading a corrupt archive.

Files land in `~/.cache/mrtoeplitz/delics`, or wherever `MRTOEPLITZ_DATA`
points.

Deli-CS is Iyer, Schauman, Sandino et al., *Deep Learning Initialized
Compressed Sensing in Volumetric Spatio-Temporal Subspace Reconstruction* —
3D spiral-projection MRF at 3T on a GE Premier with a 48-channel head coil,
released under the BSD licence.

## Reading the numbers

Peak memory is deterministic and reproducible; absolute times are not, and are
reported with the card and its clock state named.

Warm every lane before measuring it. The first application of any lane
compiles and autotunes its Triton kernels and allocates hundreds of megabytes
doing so, which has nothing to do with the transfer and will otherwise be
reported as the cost of the lane.

*Not yet run.*
