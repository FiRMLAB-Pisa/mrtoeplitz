"""The figure the README carries: four subspace coefficients, two stages.

The first pair of rows is the zero-filled reconstruction ``A^H y``, which is
where a solve starts. The second is one application of the normal operator on
top of it, which is what the rest of a solve spends its time on. Neither is a
reconstruction. What there is to check by eye is that the operator applied to
real data gives a head, that the coefficients carry the different contrasts a
subspace is for, and that the host and the device agree.

``N(A^H y)`` is noisier than ``A^H y`` because it is meant to be: at this
undersampling the normal operator is a long way from the identity, so it
amplifies what the adjoint left behind rather than cleaning it up. A solve
does that with a regulariser; this figure is the operator alone.

    python benchmarks/figure.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import delics
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lane_mrtoeplitz import adjoint
from prepare import prepare

import mrtoeplitz as mt

matplotlib.use("Agg")

HERE = Path(__file__).resolve().parent

#: The grid the comparison runs on, which is the one BART reconstructed at.
SIZE = 256

#: Coefficients shown, which is the rank the benchmark runs at.
SHOWN = 4

#: Where the slice is taken. Axis two is the axial one for this volume, and
#: this is the level that carries the lateral ventricles and the deep grey
#: nuclei, which is what makes a reconstruction worth looking at.
SLICE_AXIS = 2
SLICE_INDEX = 108


def _mrtoeplitz_rows(device: str) -> tuple[np.ndarray, np.ndarray]:
    """``A^H y`` and ``N(A^H y)`` on ``device``, coefficient by coefficient."""
    acquisition = prepare()
    shape = (SIZE,) * 3
    maps = np.load(delics.data_root() / f"coil_maps_{SIZE}.npz")["maps"]
    backprojection = adjoint(acquisition, maps, shape, device)

    trajectory = torch.as_tensor(acquisition.trajectory)
    density = torch.as_tensor(acquisition.density)
    if device == "cuda":
        trajectory, density = trajectory.cuda(), density.cuda()
    kernel = mt.subspace_kernel(trajectory, acquisition.basis, shape, density=density)
    if device == "cuda":
        torch.cuda.empty_cache()

    image = torch.as_tensor(backprojection)[None].to(device)
    sensitivities = mt.CoilKernels.from_maps(torch.as_tensor(maps), (16,) * 3).to(
        device
    )
    normal = mt.apply_sense(kernel, image, sensitivities, coil_batch_size=1)

    def slices(volume: Any) -> np.ndarray:
        """The chosen slice of each coefficient, the way a radiologist reads it.

        The reconstruction grid has the anterior direction along a column and
        the left-right one along a row; transposing and flipping puts the face
        at the top of the panel.
        """
        taken = volume.take(SLICE_INDEX, axis=SLICE_AXIS + 1)[:SHOWN]
        return np.flip(taken.transpose(0, 2, 1), axis=1)

    return slices(np.abs(backprojection)), slices(normal[0].abs().cpu().numpy())


def _scaled(row: np.ndarray) -> np.ndarray:
    """Put a row on [0, 1] by its own bright end, so all four share a window."""
    ceiling = np.percentile(row, 99.5)
    return np.clip(row / ceiling, 0.0, 1.0) if ceiling > 0 else row


def main() -> None:
    host, device = _mrtoeplitz_rows("cpu"), _mrtoeplitz_rows("cuda")
    # The two devices run different code and must not answer differently. A
    # cross-stream fault on the device shows up here and in nothing else: it
    # costs no time and leaves the shapes alone.
    for stage, (left, right) in enumerate(zip(host, device, strict=True)):
        difference = np.linalg.norm(left - right) / np.linalg.norm(left)
        print(f"stage {stage}: host and device differ by {difference:.2e}")
        if difference > 1e-2:
            raise RuntimeError(
                f"the host and the device disagree by {difference:.2e}, which "
                "is more than the gridding tolerance can account for"
            )
    rows = {
        "A^H y  CPU": host[0],
        "A^H y  CUDA": device[0],
        "N(A^H y)  CPU": host[1],
        "N(A^H y)  CUDA": device[1],
    }
    panel = np.concatenate(
        [np.concatenate(list(_scaled(row)), axis=1) for row in rows.values()],
        axis=0,
    )
    height, width = panel.shape
    figure = plt.figure(figsize=(width / 200, height / 200), dpi=200)
    axes = figure.add_axes((0, 0, 1, 1))
    axes.imshow(panel, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes.set_axis_off()
    for index, name in enumerate(rows):
        axes.text(
            6,
            index * SIZE + 16,
            name,
            color="white",
            fontsize=5,
            family="monospace",
        )
    for coefficient in range(SHOWN):
        axes.text(
            coefficient * SIZE + 6,
            height - 8,
            f"coefficient {coefficient}",
            color="white",
            fontsize=5,
            family="monospace",
        )
    target = HERE.parent / "examples/figures/benchmark.png"
    figure.savefig(target, dpi=200, pad_inches=0)
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
