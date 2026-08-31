"""The floor: what a multicoil subspace normal cannot go below.

One application of the normal has to transform every coil's every coefficient
onto the grid the convolution runs on and back. That is ``coils x rank``
volumes, one forward and one inverse each, and nothing else here is counted --
the SENSE multiply, the transfer multiply and every copy are taken as free.
No implementation can beat this, and how close one gets to it is the only
scale on which the runtimes mean anything.

There are two ways to do that transform and they are not the same cost. The
padded layout runs one pair on the doubled grid. A parity decomposition never
materialises that grid: it runs eight pairs on image-grid volumes, the same
number of cells but a smaller transform each, and a smaller log factor with
it. Timing only the doubled grid would set a floor the parity layout is
already under, so both are measured and the floor is the cheaper.

    python benchmarks/lane_floor.py --device cuda --size 256
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare import COILS, RANK


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--maps", default=None)  # accepted, unused
    arguments = parser.parse_args()

    def clock() -> float:
        if arguments.device == "cuda":
            torch.cuda.synchronize()
        return perf_counter()

    def time_pairs(shape, count) -> float:
        """Seconds for ``count`` forward-and-inverse pairs of ``shape``."""
        block = torch.zeros(
            (count, *shape), dtype=torch.complex64, device=arguments.device
        )
        axes = tuple(range(1, block.ndim))

        def pair(data=block) -> None:
            torch.fft.ifftn(torch.fft.fftn(data, dim=axes), dim=axes)

        pair()  # warm: the first call plans
        start = clock()
        for _ in range(arguments.repeats):
            pair()
        elapsed = (clock() - start) / arguments.repeats
        del block
        if arguments.device == "cuda":
            torch.cuda.empty_cache()
        return elapsed

    size = arguments.size
    padded = time_pairs((2 * size,) * 3, 1)
    # 2^3 parity components, each an image-grid volume: the same cells, a
    # cheaper transform.
    parity = time_pairs((size,) * 3, 8)

    volumes = COILS * RANK
    floor = min(padded, parity)
    print(
        "BENCHMARK "
        + json.dumps(
            {
                "seconds": {
                    "apply": floor * volumes,
                    "apply_padded": padded * volumes,
                    "apply_parity": parity * volumes,
                },
                "extra": {
                    "one_padded_pair": padded,
                    "one_parity_set": parity,
                    "volumes": float(volumes),
                },
            }
        )
    )


if __name__ == "__main__":
    main()
