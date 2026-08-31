"""Running one measurement in its own process, and watching what it costs.

Three implementations in three languages cannot be measured from inside a
single process, and a peak that includes a previous lane's allocator is not a
peak. So each measurement is a subprocess: the child times its own phases and
prints them, and this side watches what the operating system and the driver
say it took.

Host memory is the kernel's own ``VmHWM`` -- the peak resident set, exact and
free. Device memory is sampled from ``nvidia-smi`` for that process, which is
the only figure available for all three: a Torch allocator count would not be
comparable with what BART or Julia do.
"""

from __future__ import annotations

__all__ = ["Measurement", "device_bytes_in_use", "run_measured"]

import json
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

#: How often the device is asked what a process is holding. Anything finer is
#: dominated by the cost of asking.
_SAMPLE_SECONDS = 0.02

#: Resolved once: the sampler runs a few thousand times per benchmark.
_NVIDIA_SMI = shutil.which("nvidia-smi")


@dataclass
class Measurement:
    """What one lane cost."""

    name: str
    seconds: dict[str, float] = field(default_factory=dict)
    extra: dict[str, float] = field(default_factory=dict)
    peak_host: int = 0
    peak_device: int = 0
    failed: str | None = None

    def row(self) -> str:
        if self.failed:
            return f"{self.name:>28}: {self.failed}"
        phases = "  ".join(
            f"{phase} {1e3 * value:8.1f} ms" for phase, value in self.seconds.items()
        )
        return (
            f"{self.name:>28}: RAM {self.peak_host / 2**20:8.1f} MiB  "
            f"VRAM {self.peak_device / 2**20:8.1f} MiB  {phases}"
        )


def _peak_host_bytes(pid: int) -> int:
    """Peak resident set of a process, from the kernel rather than sampled."""
    try:
        status = Path(f"/proc/{pid}/status").read_text()
    except OSError:
        return 0
    found = re.search(r"^VmHWM:\s+(\d+) kB", status, re.MULTILINE)
    return int(found.group(1)) * 1024 if found else 0


def device_bytes_in_use() -> int:
    """Total device memory in use, in bytes.

    Per-process device memory is the figure one would want, and on this
    platform it is not available: under WSL2 ``--query-compute-apps`` reports
    nothing at all. So the whole device is read instead, and a baseline taken
    before the child starts is subtracted. That is sound only because lanes
    run one at a time, which is why they do.
    """
    if _NVIDIA_SMI is None:
        return 0
    try:
        answer = subprocess.run(  # noqa: S603 -- fixed argv, resolved above
            [
                _NVIDIA_SMI,
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return 0
    first = answer.stdout.strip().splitlines()
    return int(first[0].strip()) * 2**20 if first and first[0].strip().isdigit() else 0


def run_measured(
    name: str,
    command: list[str],
    *,
    env: dict[str, str] | None = None,
    timeout: float = 3600.0,
    cwd: str | None = None,
) -> Measurement:
    """Run ``command``, sampling what it holds, and read the timings it prints.

    The child is expected to print one JSON object on a line beginning
    ``BENCHMARK ``, with a ``seconds`` mapping of phase names to durations and
    an optional ``extra`` mapping of anything else worth recording. Whatever
    else it writes is kept only if it fails.
    """
    measurement = Measurement(name=name)
    baseline = device_bytes_in_use()
    process = subprocess.Popen(  # noqa: S603
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        cwd=cwd,
    )

    watching = True

    def watch() -> None:
        while watching:
            measurement.peak_host = max(
                measurement.peak_host, _peak_host_bytes(process.pid)
            )
            measurement.peak_device = max(
                measurement.peak_device, device_bytes_in_use() - baseline
            )
            time.sleep(_SAMPLE_SECONDS)

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    try:
        out, err = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        out, err = process.communicate()
        measurement.failed = f"timed out after {timeout:.0f} s"
    finally:
        # VmHWM survives until the process is reaped, so the last read is the
        # true peak however coarse the sampling was.
        measurement.peak_host = max(
            measurement.peak_host, _peak_host_bytes(process.pid)
        )
        watching = False
        watcher.join(timeout=1.0)

    if measurement.failed is None and process.returncode != 0:
        tail = (err or out or "").strip().splitlines()
        measurement.failed = tail[-1][:160] if tail else f"exit {process.returncode}"
    for line in (out or "").splitlines():
        if line.startswith("BENCHMARK "):
            reported = json.loads(line[len("BENCHMARK ") :])
            measurement.seconds = reported.get("seconds", {})
            measurement.extra = reported.get("extra", {})
    return measurement
