"""The benchmark: three phases, three metrics, every lane, both devices.

    A^H y          the zero-filled reconstruction a solve starts from
    kernel create  building the Toeplitz transfer
    kernel apply   one application of the normal operator

for runtime, peak host memory and peak device memory. Each measurement is its
own process, so no lane's allocator is charged to another's peak.

    python benchmarks/run.py --sizes 64,96 --lanes mrtoeplitz,julia
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import run_measured
from prepare import _cache_path, prepare

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

#: Where the Julia lane and its project live. Julia is not a dependency of the
#: package; this is the environment prepared beside it.
JULIA = Path(
    os.environ.get(
        "MRTOEPLITZ_JULIA",
        "/home/mcencini/pulserver-project/refcode/recon/torchsim/benchmarks/.env/julia/bin/julia",
    )
)
JULIA_PROJECT = Path(os.environ.get("MRTOEPLITZ_JULIA_PROJECT", str(HERE / "julia")))
JULIA_DEPOT = Path(
    os.environ.get(
        "MRTOEPLITZ_JULIA_DEPOT",
        "/home/mcencini/pulserver-project/refcode/recon/torchsim/benchmarks/.env/juliadepot",
    )
)


def lane_command(lane: str, device: str, size: int, repeats: int, maps: Path):
    """The command and environment for one lane, or None if it is unavailable."""
    if lane == "mrtoeplitz":
        return (
            [
                str(ROOT / ".venv/bin/python"),
                str(HERE / "lane_mrtoeplitz.py"),
                "--device",
                device,
                "--size",
                str(size),
                "--repeats",
                str(repeats),
                "--maps",
                str(maps),
            ],
            None,
        )
    if lane == "floor":
        return (
            [
                str(ROOT / ".venv/bin/python"),
                str(HERE / "lane_floor.py"),
                "--device",
                device,
                "--size",
                str(size),
            ],
            None,
        )
    if lane == "bart":
        bart = os.environ.get("BART_PATH")
        if not bart or not Path(bart).exists():
            return None
        return (
            [
                str(ROOT / ".venv/bin/python"),
                str(HERE / "lane_bart.py"),
                "--device",
                device,
                "--size",
                str(size),
                "--repeats",
                str(repeats),
                "--maps",
                str(maps),
                "--bart",
                bart,
            ],
            None,
        )
    if lane == "julia":
        if not JULIA.exists() or not (JULIA_PROJECT / "lane.jl").exists():
            return None
        environment = dict(os.environ, JULIA_DEPOT_PATH=str(JULIA_DEPOT))
        return (
            [
                str(JULIA),
                # Julia is single-threaded unless told otherwise, and the
                # other lanes are not. Measuring one thread against however
                # many BLAS and FINUFFT take would say nothing about either.
                "-t",
                "auto",
                f"--project={JULIA_PROJECT}",
                str(JULIA_PROJECT / "lane.jl"),
                str(_cache_path()),
                str(size),
                device,
                str(repeats),
                str(maps),
            ],
            environment,
        )
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="256")
    parser.add_argument("--lanes", default="floor,mrtoeplitz,julia,bart")
    parser.add_argument("--devices", default="cpu,cuda")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out", default=str(HERE / "results.json"))
    parser.add_argument("--maps", default=None)
    arguments = parser.parse_args()

    prepare()  # cached; makes sure no lane pays for it
    print(f"data cached at {_cache_path()}\n")

    results = []
    for size in (int(s) for s in arguments.sizes.split(",")):
        maps = (
            Path(arguments.maps)
            if arguments.maps
            else _cache_path().with_name(f"coil_maps_{size}.npz")
        )
        if not maps.exists():
            print(f"no coil maps at {maps}; run benchmarks/julia/coil_maps.jl first")
            continue
        print(f"=== {size}^3 ===")
        for device in arguments.devices.split(","):
            for lane in arguments.lanes.split(","):
                prepared = lane_command(lane, device, size, arguments.repeats, maps)
                if prepared is None:
                    print(f"{lane + ' ' + device:>28}: not available here")
                    continue
                command, environment = prepared
                measurement = run_measured(f"{lane} {device}", command, env=environment)
                print("  " + measurement.row())
                results.append(
                    {
                        "size": size,
                        "lane": lane,
                        "device": device,
                        "seconds": measurement.seconds,
                        "extra": measurement.extra,
                        "peak_host": measurement.peak_host,
                        "peak_device": measurement.peak_device,
                        "failed": measurement.failed,
                    }
                )
        print()

    Path(arguments.out).write_text(json.dumps(results, indent=2))
    print(f"wrote {arguments.out}")


if __name__ == "__main__":
    main()
