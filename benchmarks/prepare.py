"""Preparing the Deli-CS acquisition once, for every lane to share.

The comparison is between implementations, so all three are handed exactly the
same arrays: the same shots, the same virtual coils, the same basis. That
preparation is not part of any measurement, and it is cached because it reads
seventeen gigabytes to produce four hundred megabytes.

Two things here are read off the data rather than assumed. Which axis of the
raw array is the shot and which the frame is not written down anywhere, and
getting it wrong still produces an image -- just not of a head. And the coil
compression is fitted on the k-space centre, where the array's correlations
are.
"""

from __future__ import annotations

__all__ = ["Acquisition", "prepare"]

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import delics

#: What the benchmark runs on: a rank-4 subspace over eight virtual coils.
RANK = 4
COILS = 8
SHOTS = 8
FRAMES = 500


@dataclass
class Acquisition:
    """One prepared acquisition, shared by every lane."""

    kspace: np.ndarray  # (coils, frames, shots, points) complex64
    trajectory: np.ndarray  # (frames, shots, points, 3) float32, normalized
    density: np.ndarray  # (frames, shots, points) float32
    basis: np.ndarray  # (frames, rank) float32 -- see _real_basis

    @property
    def samples(self) -> int:
        return int(np.prod(self.trajectory.shape[:-1]))


def _real_basis(basis: np.ndarray) -> np.ndarray:
    """Return the basis as real, having checked that it is.

    Deli-CS stores phi complex, but its imaginary part is a millionth of its
    real one: the components are real up to a global phase, which a subspace
    absorbs. Keeping it complex would cost both implementations -- ours in
    storage, MRISubspaceRecon's in a documented real-only NUFFT it would no
    longer take -- for nothing.
    """
    share = float(np.linalg.norm(basis.imag) / np.linalg.norm(basis.real))
    if share > 1e-4:
        raise ValueError(
            f"the basis is genuinely complex ({share:.2e} of it), so it cannot "
            f"be taken as real"
        )
    return np.ascontiguousarray(basis.real, dtype=np.float32)


def _cache_path() -> Path:
    return delics.data_root() / f"prepared_r{RANK}_c{COILS}_s{SHOTS}_f{FRAMES}.npz"


def _compress_coils(raw: np.ndarray, radius: np.ndarray) -> np.ndarray:
    """Compress the physical channels onto ``COILS`` virtual ones.

    The basis is fitted on the samples nearest k-space centre, which is where
    the array's correlations are and where the object is brightest, and then
    applied to everything. The covariance is channels by channels however many
    samples there are, so it is accumulated rather than formed at once.
    """
    channels = raw.shape[0]
    centre = radius < 0.1 * radius.max()
    covariance = np.zeros((channels, channels), dtype=np.complex128)
    flat = raw.reshape(channels, -1)
    inside = centre.reshape(-1)
    for start in range(0, flat.shape[1], 1 << 20):
        block = flat[:, start : start + (1 << 20)]
        block = block[:, inside[start : start + (1 << 20)]]
        if block.size:
            covariance += block @ block.conj().T
    values, vectors = np.linalg.eigh(covariance)
    keep = vectors[:, np.argsort(-values)[:COILS]].conj().T.astype(np.complex64)
    kept = float(np.sort(values)[::-1][:COILS].sum() / values.sum())
    print(f"  {channels} channels -> {COILS}, holding {100 * kept:.1f}% of the energy")
    return (keep @ flat).reshape(COILS, *raw.shape[1:])


def _gridded(samples: np.ndarray, values: np.ndarray, size: int = 64) -> np.ndarray:
    """A quick adjoint onto a coarse grid, for judging an ordering."""
    import finufft

    points = np.ascontiguousarray(samples.reshape(-1, 3).astype(np.float64) * 2 * np.pi)
    plan = finufft.Plan(1, (size,) * 3, isign=1, eps=1e-4, dtype="complex128")
    plan.setpts(*[np.ascontiguousarray(points[:, axis]) for axis in range(3)])
    return np.abs(plan.execute(values.reshape(-1).astype(np.complex128)))


def _sharpness(image: np.ndarray) -> float:
    """How concentrated an image is. Noise is flat; an object is not."""
    flat = image.reshape(-1)
    return float(flat.max() / flat.mean())


def _resolve_layout(raw: np.ndarray, trajectory: np.ndarray) -> tuple[str, int]:
    """Work out how the raw arms map onto the trajectory, by reconstructing.

    Neither the order of the 24000 arms nor which of the 2000 digitised
    samples the 1688 trajectory points are is written down. Both orderings
    produce an array of the right shape, and only one produces an image, so
    the question is settled by gridding a few arms each way and seeing which
    is concentrated.
    """
    frames, shots, points, _ = trajectory.shape
    probe_frames = 24
    best = ("", 0, 0.0)
    for order in ("shot-major", "frame-major"):
        for offset in (0, 2000 - points):
            arms, samples = [], []
            for frame in range(probe_frames):
                for shot in range(shots):
                    index = (
                        frame * shots + shot
                        if order == "frame-major"
                        else shot * frames + frame
                    )
                    arms.append(raw[offset : offset + points, 0, index])
                    samples.append(trajectory[frame, shot])
            score = _sharpness(_gridded(np.stack(samples), np.stack(arms)))
            print(f"  {order:>12}, offset {offset:>4}: sharpness {score:8.1f}")
            if score > best[2]:
                best = (order, offset, score)
    print(f"  -> {best[0]}, offset {best[1]}")
    return best[0], best[1]


def prepare(*, force: bool = False) -> Acquisition:
    """Return the shared acquisition, building and caching it the first time."""
    from scipy.io import loadmat

    cache = _cache_path()
    if cache.exists() and not force:
        held = np.load(cache)
        return Acquisition(
            kspace=held["kspace"],
            trajectory=held["trajectory"],
            density=held["density"],
            basis=_real_basis(held["basis"]),
        )

    shared = delics.fetch("shared") / "data" / "shared"
    matlab = loadmat(shared / "traj_grp48_inacc1.mat")
    # (points, axes, shots, frames) -> (frames, shots, points, axes). The
    # six-minute file carries its own aligned DCF, so the two cannot be paired
    # up wrongly.
    trajectory = np.ascontiguousarray(
        np.transpose(matlab["k_3d"], (3, 2, 0, 1))[:FRAMES, :SHOTS], dtype=np.float32
    )
    density = np.ascontiguousarray(
        np.transpose(matlab["DCF"], (2, 1, 0))[:FRAMES, :SHOTS], dtype=np.float32
    )
    # phi is (frames, rank): its components are its columns.
    basis = np.ascontiguousarray(
        loadmat(shared / "phi.mat")["phi"][:FRAMES, :RANK], dtype=np.complex64
    )

    raw_path = delics.fetch("raw") / "data" / "validation" / "case000" / "raw_mrf.npy"
    raw = np.load(raw_path, mmap_mode="r")  # (2000, 48, 24000), 17 GiB
    print(f"raw {raw.shape} {raw.dtype}, mapped not loaded")
    order, offset = _resolve_layout(raw, trajectory)

    points = trajectory.shape[2]
    channels = raw.shape[1]
    gathered = np.empty((channels, FRAMES, SHOTS, points), dtype=np.complex64)
    for frame in range(FRAMES):
        for shot in range(SHOTS):
            index = (
                frame * raw.shape[2] // FRAMES + shot
                if order == "frame-major"
                else shot * FRAMES + frame
            )
            gathered[:, frame, shot] = raw[offset : offset + points, :, index].T
    print(f"  gathered {gathered.shape}, {gathered.nbytes / 2**30:.2f} GiB")

    radius = np.linalg.norm(trajectory, axis=-1)
    kspace = _compress_coils(gathered, radius)
    del gathered

    np.savez(cache, kspace=kspace, trajectory=trajectory, density=density, basis=basis)
    print(f"  cached to {cache}")
    return Acquisition(
        kspace=kspace,
        trajectory=trajectory,
        density=density,
        basis=_real_basis(basis),
    )


if __name__ == "__main__":
    acquisition = prepare(force="--force" in sys.argv)
    print(
        f"kspace {acquisition.kspace.shape}, trajectory "
        f"{acquisition.trajectory.shape}, basis {acquisition.basis.shape}"
    )
