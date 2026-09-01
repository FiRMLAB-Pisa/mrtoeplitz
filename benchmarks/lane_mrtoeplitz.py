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

    ``A^H = NUFFT_adjoint . C^H``, so it is one type-1 transform per coil and
    basis coefficient over every frame's samples at once, with the density
    folded in and the sensitivities taken off as it goes -- what comes back is
    one volume per coefficient rather than one per coil.

    The coil is the outer loop, so its map is expanded once and read by every
    coefficient rather than once per coefficient. What is on the device is that
    map, the coefficient's samples, the volume they grid onto and the answer
    being accumulated; the acquisition itself stays on the host. Samples cross
    on a stream of their own against a pair of buffers, so the upload of one
    coefficient overlaps the gridding of the one before it.
    """
    library = "cufinufft" if device == "cuda" else "finufft"
    module = __import__(library)

    coils = acquisition.kspace.shape[0]
    samples = acquisition.trajectory.reshape(-1, 3) * 2 * np.pi
    basis = acquisition.basis
    rank = basis.shape[1]
    n_samples = samples.shape[0]

    # The same gridding the transfer is built with: a looser tolerance on the
    # smaller working grid, which is 2.5x faster for an error an order of
    # magnitude under what compression leaves.
    options = {"eps": 1e-3, "upsampfac": 1.25}
    if device != "cuda":
        plan = module.Plan(1, shape, isign=1, dtype="complex64", **options)
        plan.setpts(*(np.ascontiguousarray(samples[:, axis]) for axis in range(3)))
        out = np.zeros((rank, *shape), dtype=np.complex64)
        for coil in range(coils):
            weighted = acquisition.kspace[coil] * acquisition.density
            for coefficient in range(rank):
                projected = weighted * np.conj(basis[:, coefficient])[:, None, None]
                out[coefficient] += plan.execute(
                    np.ascontiguousarray(projected.reshape(-1))
                ) * np.conj(maps[coil])
        return out

    import ctypes

    compute = torch.cuda.Stream()
    upload = torch.cuda.Stream()
    # The plan issues to the compute stream by name rather than to the legacy
    # default one, which is what leaves the upload stream free to run ahead.
    options["gpu_stream"] = ctypes.c_void_p(compute.cuda_stream)
    plan = module.Plan(1, shape, isign=1, dtype="complex64", **options)
    with torch.cuda.stream(compute):
        plan.setpts(
            *(
                torch.as_tensor(np.ascontiguousarray(samples[:, axis])).cuda()
                for axis in range(3)
            )
        )
    staged = [
        torch.empty(n_samples, dtype=torch.complex64, pin_memory=True) for _ in range(2)
    ]
    held = [
        torch.empty(n_samples, dtype=torch.complex64, device="cuda") for _ in range(2)
    ]
    gridded = torch.empty(shape, dtype=torch.complex64, device="cuda")
    # The answer is accumulated where it is made. Bringing each coefficient
    # back as it lands would synchronise the host every time and there would be
    # nothing left for the upload to overlap.
    total = torch.zeros((rank, *shape), dtype=torch.complex64, device="cuda")
    released = [None, None]
    uploaded = [None, None]

    for coil in range(coils):
        weighted = acquisition.kspace[coil] * acquisition.density
        # Allocated on the stream that reads it. Left on the default stream it
        # would be freed back to a pool the next coil allocates from while the
        # gridding is still reading it.
        with torch.cuda.stream(compute):
            coil_map = torch.as_tensor(maps[coil]).cuda().conj()
        for coefficient in range(rank):
            projected = weighted * np.conj(basis[:, coefficient])[:, None, None]
            slot = coefficient % 2
            # The upload out of this staging buffer may still be in flight.
            # Writing it again before that copy has landed puts one
            # coefficient's samples into another's transform, and nothing on
            # the device orders a host write against it.
            if uploaded[slot] is not None:
                uploaded[slot].synchronize()
            staged[slot].copy_(
                torch.from_numpy(np.ascontiguousarray(projected.reshape(-1)))
            )
            # Only the gridding that last read this slot has to finish; the one
            # in flight is reading the other.
            if released[slot] is not None:
                upload.wait_event(released[slot])
            with torch.cuda.stream(upload):
                held[slot].copy_(staged[slot], non_blocking=True)
                uploaded[slot] = torch.cuda.Event()
                uploaded[slot].record(upload)
            compute.wait_event(uploaded[slot])
            with torch.cuda.stream(compute):
                plan.execute(held[slot], out=gridded)
                total[coefficient] += gridded * coil_map
                released[slot] = torch.cuda.Event()
                released[slot].record(compute)
        del coil_map
    torch.cuda.current_stream().wait_stream(compute)
    torch.cuda.synchronize()
    return total.cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--maps", required=True)
    parser.add_argument(
        "--resident",
        action="store_true",
        help="move the transfer onto the device instead of streaming it",
    )
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

    # Where the transfer is gridded follows where the trajectory is: a host
    # trajectory builds with FINUFFT, a device one with CUFINUFFT. Handing it
    # a NumPy array and moving the kernel afterwards would time a host build
    # and a copy.
    trajectory = acquisition.trajectory
    density = acquisition.density
    if arguments.device == "cuda":
        trajectory = torch.as_tensor(trajectory).cuda()
        density = torch.as_tensor(density).cuda()

    # No policy and no options: what a caller gets by asking for nothing. The
    # build stages each basis pair's rows to pinned host memory as it finishes
    # them, on a stream of their own, so the transfer never accumulates on the
    # card -- only the one buffer the gridding reuses.
    start = clock()
    kernel = mt.subspace_kernel(
        trajectory,
        acquisition.basis,
        shape,
        density=density,
    )
    seconds["create"] = clock() - start
    extra = {"transfer_bytes": float(kernel.storage_nbytes)}

    # The trajectory and the density built the transfer and are not read
    # again; on this grid they are more than a gigabyte of the card, and an
    # apply that has to work around them runs half as fast again. Building on
    # the device also leaves the gridding plan's blocks in the caching
    # allocator, which Torch will not give back on its own.
    del trajectory, density
    if arguments.device == "cuda":
        torch.cuda.empty_cache()

    # A transfer on the host is streamed; one moved across is applied from
    # where it was put. The first is the default and the second is what it is
    # measured against.
    if arguments.resident and arguments.device == "cuda":
        kernel.to(arguments.device)

    # The real multicoil normal: one coefficient volume in and out, with the
    # sensitivities applied and summed over inside. Coils are taken one at a
    # time, which is what holds the footprint down.
    image = torch.zeros(
        (1, rank, *shape), dtype=torch.complex64, device=arguments.device
    )
    # Eight coil kernels and one map in flight, not a bank of eight. At this
    # size the dense bank is 1.07 GiB on the card; the kernels are a quarter
    # of a megabyte and a coil is expanded when the apply asks for it. These
    # maps floor at about 5e-3 however large the kernel, that residual being
    # the eigenvector normalisation's edge rather than the sensitivity, so
    # there is nothing to buy above a small one.
    sensitivities = mt.CoilKernels.from_maps(torch.as_tensor(maps), (16,) * 3).to(
        arguments.device
    )

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
