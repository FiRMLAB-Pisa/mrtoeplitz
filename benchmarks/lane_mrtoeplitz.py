"""The mrtoeplitz lane: A^H y, kernel creation, kernel application.

Run as a subprocess by ``run.py``; prints its phase timings for the harness to
read. Everything is in low-memory mode, which is the default: the compact
apply out of the packed transfer, filed by coordinate parity so the doubled
grid is never materialised.

    python benchmarks/lane_mrtoeplitz.py --device cuda --size 128
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare import prepare

import mrtoeplitz as mt


def adjoint(acquisition, maps, shape, device):
    """A^H y: the zero-filled reconstruction the solve starts from.

    One type-1 transform per basis coefficient and coil, over every frame's
    samples at once, with the density folded in, combined through the
    sensitivities as it goes -- so what comes back is one coefficient volume
    rather than one per coil.
    """
    library = "cufinufft" if device == "cuda" else "finufft"
    module = __import__(library)

    coils = acquisition.kspace.shape[0]
    samples = acquisition.trajectory.reshape(-1, 3) * 2 * np.pi
    weighted = acquisition.kspace * acquisition.density[None]
    basis = acquisition.basis

    if device == "cuda":
        columns = [
            torch.as_tensor(np.ascontiguousarray(samples[:, axis])).cuda()
            for axis in range(3)
        ]
    else:
        columns = [np.ascontiguousarray(samples[:, axis]) for axis in range(3)]
    out = np.zeros((basis.shape[1], *shape), dtype=np.complex64)

    plan = module.Plan(1, shape, isign=1, eps=1e-4, dtype="complex64")
    plan.setpts(*columns)
    for coefficient in range(basis.shape[1]):
        # Project onto the basis first: one transform per coefficient rather
        # than one per frame.
        projected = weighted * np.conj(basis[:, coefficient])[None, :, None, None]
        for coil in range(coils):
            values = np.ascontiguousarray(projected[coil].reshape(-1))
            if device == "cuda":
                gridded = plan.execute(torch.as_tensor(values).cuda())
                combined = gridded * torch.as_tensor(maps[coil]).cuda().conj()
                out[coefficient] += combined.cpu().numpy()
                del gridded, combined
            else:
                out[coefficient] += plan.execute(values) * np.conj(maps[coil])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--maps", required=True)
    arguments = parser.parse_args()

    shape = (arguments.size,) * 3
    acquisition = prepare()
    rank = acquisition.basis.shape[1]
    maps = np.load(arguments.maps)["maps"]
    assert maps.shape[1:] == shape, f"maps are {maps.shape[1:]}, asked for {shape}"
    seconds: dict[str, float] = {}

    def clock() -> float:
        if arguments.device == "cuda":
            torch.cuda.synchronize()
        return perf_counter()

    start = clock()
    backprojection = adjoint(acquisition, maps, shape, arguments.device)
    seconds["adjoint"] = clock() - start
    del backprojection

    start = clock()
    kernel = mt.subspace_kernel(
        acquisition.trajectory,
        acquisition.basis,
        shape,
        density=acquisition.density,
    )
    if arguments.device == "cuda":
        kernel = kernel.to("cuda")
    seconds["create"] = clock() - start
    extra = {"transfer_bytes": float(kernel.storage_nbytes)}

    # The real multicoil normal: one coefficient volume in and out, with the
    # sensitivities applied and summed over inside. Coils are taken one at a
    # time, which is what holds the footprint down.
    image = torch.zeros(
        (1, rank, *shape), dtype=torch.complex64, device=arguments.device
    )
    sensitivities = torch.as_tensor(maps).to(arguments.device)

    def apply():
        return mt.apply_sense(kernel, image, sensitivities, coil_batch_size=1)

    apply()  # warm: the first application compiles and autotunes
    start = clock()
    for _ in range(arguments.repeats):
        apply()
    seconds["apply"] = (clock() - start) / arguments.repeats

    print("BENCHMARK " + json.dumps({"seconds": seconds, "extra": extra}))


if __name__ == "__main__":
    main()
