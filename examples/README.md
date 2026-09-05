# Examples

One example per usage case, each checking the operator against the thing it
replaces. The `.py` is the source: it runs as a script, lints with the rest of
the package, and reads as a diff. The `.ipynb` beside it is generated from it,
executed, and committed with its outputs, so it opens in Colab and runs top to
bottom on synthetic data — there is nothing to download.

| notebook | shows | checked against |
|---|---|---|
| [`01-scalar`](01-scalar.ipynb) | `scalar_kernel`, and a CG solve driven by it | raw FINUFFT, `A_adjoint(A(x))` |
| [`02-subspace`](02-subspace.ipynb) | `subspace_kernel` on a 500-frame fingerprinting scan, real dictionary basis | the definition, all 500 frames one at a time |
| [`03-cartesian`](03-cartesian.ipynb) | `cartesian_subspace_kernel` | the masked transform, exactly |
| [`04-coil_kernels`](04-coil_kernels.ipynb) | `CoilKernels`, `apply_sense`, against BART's nlinv / ecalib / coils maps | the dense map banks they stand for |
| [`05-streaming`](05-streaming.ipynb) | `CudaStreaming` | the same operator, resident |
| [`06-unrolled`](06-unrolled.ipynb) | gradients through the normal | a second application of the operator |

`01-scalar`, `02-subspace`, `03-cartesian`, `04-coil_kernels` and `06-unrolled` run on a host
alone and need the `nufft` extra. `05-streaming` needs a CUDA device, and says so
rather than failing.

`coil_maps.npz` holds three sensitivity banks from BART and `mrf_basis.npz`
the Deli-CS temporal basis; [`make_coil_maps.sh`](make_coil_maps.sh) records
the BART commands that produced the first.

## Rebuilding

```bash
pip install -e .[dev] jupytext nbclient ipykernel
bash scripts/build_examples.sh
```

Every notebook is regenerated from its script and executed against the
interpreter the package is installed into, which is also what writes the
figures under [`figures/`](figures/). `--check` verifies the notebooks are
current without running them.
