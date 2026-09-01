"""The BART lane: A^H y, kernel creation, kernel application.

BART has no call that builds a Toeplitz kernel and stops, but ``pics`` can
export the one it built and import it again, which separates the three phases
cleanly:

    A^H y          ``nufft -a``, per coil, combined through the sensitivities
    kernel create  ``pics -i0 --psf_export``
    kernel apply   ``pics --psf_import`` over N iterations, minus the -i0 cost

The data is written once in BART's own format and cached; that conversion is
not part of any measurement.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare import prepare


def write_cfl(stem: Path, array: np.ndarray) -> None:
    """Write a BART cfl/hdr pair. BART's dimensions are Fortran-ordered."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    dims = list(array.shape) + [1] * (16 - array.ndim)
    with stem.with_suffix(".hdr").open("w") as handle:
        handle.write("# Dimensions\n" + " ".join(str(d) for d in dims) + "\n")
    with stem.with_suffix(".cfl").open("wb") as handle:
        # The header states the dimensions in BART's order, so the data has to
        # be written in it too. Writing C-ordered bytes under that header hands
        # BART every array transposed.
        array.astype(np.complex64).ravel(order="F").tofile(handle)


def stage(acquisition, maps, size: int, root: Path) -> dict[str, Path]:
    """Put the acquisition into BART's layout, once."""
    root.mkdir(parents=True, exist_ok=True)
    marker = root / f"staged_{size}"
    names = {
        name: root / f"{name}_{size}"
        for name in ("ksp", "traj", "basis", "maps", "dens")
    }
    if marker.exists():
        return names

    coils, frames, shots, points = acquisition.kspace.shape
    samples = shots * points
    rank = acquisition.basis.shape[1]
    # BART: (READ, PHS1, PHS2, COIL, MAPS, TE) with the frames on TE.
    ksp = acquisition.kspace.reshape(coils, frames, samples).transpose(2, 0, 1)
    ksp = ksp[None, :, None, :, None, :]  # (1, samples, 1, coils, 1, frames)
    write_cfl(names["ksp"], np.ascontiguousarray(ksp))

    # BART wants the trajectory in units of k-space samples, not normalized.
    trajectory = acquisition.trajectory.reshape(frames, samples, 3) * size
    trajectory = trajectory.transpose(2, 1, 0)[:, :, None, None, None, :]
    write_cfl(names["traj"], np.ascontiguousarray(trajectory.astype(np.complex64)))

    # nufft takes the density as a weighting, shaped like the samples.
    density = acquisition.density.reshape(frames, samples).T
    write_cfl(
        names["dens"],
        np.ascontiguousarray(
            density[None, :, None, None, None, :].astype(np.complex64)
        ),
    )

    # The basis maps TE (dim 5) onto COEFF (dim 6).
    basis = acquisition.basis.T.reshape(1, 1, 1, 1, 1, frames, rank).astype(
        np.complex64
    )
    write_cfl(names["basis"], np.ascontiguousarray(basis))

    write_cfl(names["maps"], np.ascontiguousarray(maps.transpose(1, 2, 3, 0)))
    marker.touch()
    return names


def run_bart(bart: str, arguments: list[str], gpu: bool) -> float:
    """Run one BART command, returning the seconds it took."""
    command = [bart, arguments[0]]
    if gpu:
        command.append("-g")
    command.extend(arguments[1:])
    start = perf_counter()
    finished = subprocess.run(  # noqa: S603
        command, capture_output=True, text=True, check=False
    )
    if finished.returncode != 0:
        tail = (finished.stderr or finished.stdout).strip().splitlines()
        raise RuntimeError(" ".join(command[:3]) + ": " + (tail[-1] if tail else "?"))
    return perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--maps", required=True)
    parser.add_argument("--bart", default=os.environ.get("BART_PATH", "bart"))
    parser.add_argument("--work", default=None)
    arguments = parser.parse_args()

    gpu = arguments.device == "cuda"
    acquisition = prepare()
    maps = np.load(arguments.maps)["maps"]
    work = Path(arguments.work or Path(arguments.maps).parent / "bart")
    names = stage(acquisition, maps, arguments.size, work)
    seconds: dict[str, float] = {}

    # A^H y: one call. nufft takes the basis and the density itself, so the
    # frames collapse onto coefficients inside BART rather than outside it.
    start = perf_counter()
    run_bart(
        arguments.bart,
        [
            "nufft",
            "-a",
            "--lowmem",
            "-B",
            str(names["basis"]),
            "-p",
            str(names["dens"]),
            str(names["traj"]),
            str(names["ksp"]),
            str(work / "adj"),
        ],
        gpu,
    )
    run_bart(
        arguments.bart,
        [
            "fmac",
            "-C",
            "-s",
            "8",
            str(work / "adj"),
            str(names["maps"]),
            str(work / "combined"),
        ],
        gpu,
    )
    seconds["adjoint"] = perf_counter() - start

    # Kernel creation: build the point spread function and stop.
    psf = work / f"psf_{arguments.size}"
    # BART carries the same four memory ideas this package does, so it is
    # measured with them on. Comparing our low-memory path against its
    # defaults would be measuring the flags, not the implementations.
    #
    #   real-psf        the transfer is real for a real basis
    #   compress-psf    keep only where the trajectory reached
    #   decomposed-psf  the parity decomposition of the doubled grid
    #   upper-triag-psf the packed rank (rank + 1) / 2 storage
    common = [
        "pics",
        "-B",
        str(names["basis"]),
        "-t",
        str(names["traj"]),
        "-S",
        "-U",  # lowmem
        "--nufft-conf",
        "real-psf,compress-psf,decomposed-psf,upper-triag-psf",
    ]
    seconds["create"] = run_bart(
        arguments.bart,
        [
            *common,
            "-i0",
            "--psf_export",
            str(psf),
            str(names["ksp"]),
            str(names["maps"]),
            str(work / "out"),
        ],
        gpu,
    )

    # Kernel application: N iterations against zero, less what -i0 costs.
    imported = [*common, "--psf_import", str(psf)]
    tail = [str(names["ksp"]), str(names["maps"]), str(work / "out")]
    floor = run_bart(arguments.bart, [*imported, "-i0", *tail], gpu)
    many = run_bart(arguments.bart, [*imported, f"-i{arguments.repeats}", *tail], gpu)
    seconds["apply"] = max(many - floor, 0.0) / arguments.repeats

    extra = {}
    header = psf.with_suffix(".cfl")
    if header.exists():
        extra["transfer_bytes"] = float(header.stat().st_size)
    print("BENCHMARK " + json.dumps({"seconds": seconds, "extra": extra}))


if __name__ == "__main__":
    main()
