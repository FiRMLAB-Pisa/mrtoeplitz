# Benchmarks

Peak memory and runtime for the subspace normal, against:

1. **FFT floor** — pure cuFFT of the same grids, the irreducible cost.
2. **BART** — `pics --nufft-conf compress-psf`.
3. **MRFingerprintingRecon.jl** — allocates the doubled grid whole, with no
   parity decomposition.
4. **torchkbnufft** — the Torch-native reference point.

Dataset: Deli-CS (Zenodo [7697373](https://zenodo.org/records/7697373),
[7703200](https://zenodo.org/records/7703200),
[7734431](https://zenodo.org/records/7734431)), 3D spiral-projection MRF at 3T
on a 48-channel head coil. Fetched hash-pinned; too large for CI, which runs
the same code paths on a synthetic phantom.

Peak memory is deterministic and reproducible; absolute times are not, and are
reported with the card and its clock state named.

Not yet run.
