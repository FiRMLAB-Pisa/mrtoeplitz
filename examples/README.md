# Examples

One notebook per usage case, each checking the operator against the thing it
replaces. Every notebook opens in Colab and runs top to bottom on synthetic
data — there is nothing to download.

| notebook | shows | checked against |
|---|---|---|
| [`scalar.ipynb`](scalar.ipynb) | `scalar_kernel`, and a CG solve driven by it | raw FINUFFT, `A_adjoint(A(x))` |
| [`subspace.ipynb`](subspace.ipynb) | `subspace_kernel` on a 500-frame fingerprinting scan, real dictionary basis | the definition, all 500 frames one at a time |
| [`cartesian.ipynb`](cartesian.ipynb) | `cartesian_subspace_kernel` | the masked transform, exactly |
| [`coil_kernels.ipynb`](coil_kernels.ipynb) | `CoilKernels`, `apply_sense`, against BART's nlinv / ecalib / coils maps | the dense map banks they stand for |
| [`streaming.ipynb`](streaming.ipynb) | `CudaStreaming` | the same operator, resident |
| [`unrolled.ipynb`](unrolled.ipynb) | gradients through the normal | a second application of the operator |

`scalar`, `subspace`, `cartesian`, `coil_kernels` and `unrolled` run on a host
alone and need the `nufft` extra. `streaming` needs a CUDA device, and says so
rather than failing.

`coil_maps.npz` holds three sensitivity banks from BART and `mrf_basis.npz`
the Deli-CS temporal basis; [`make_coil_maps.sh`](make_coil_maps.sh) records
the BART commands that produced the first.

The figures under [`figures/`](figures/) are what the README shows, and each is
written by the notebook of the same name. Regenerating them is running the
notebooks:

```bash
pip install -e .[dev] jupytext nbclient
jupyter nbconvert --to notebook --execute --inplace examples/*.ipynb
```
