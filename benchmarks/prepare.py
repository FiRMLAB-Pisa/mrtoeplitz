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
#: Every group of the six-minute acquisition. Taking a subset of them
#: undersamples a scan that is already accelerated, and the normal operator
#: then sits far enough from the identity that applying it to a zero-filled
#: reconstruction amplifies the streaks rather than sharpening the object.
SHOTS = 48
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


def _compression_matrix(centre: np.ndarray) -> np.ndarray:
    """Return the projection of the physical channels onto ``COILS`` virtual ones.

    Fitted on the samples nearest k-space centre, which is where the array's
    correlations are and where the object is brightest. Every arm is centre-out,
    so those are the samples the readout opens with.
    """
    channels = centre.shape[0]
    flat = centre.reshape(channels, -1)
    covariance = np.zeros((channels, channels), dtype=np.complex128)
    for start in range(0, flat.shape[1], 1 << 20):
        block = flat[:, start : start + (1 << 20)]
        covariance += block @ block.conj().T
    values, vectors = np.linalg.eigh(covariance)
    keep = vectors[:, np.argsort(-values)[:COILS]].conj().T.astype(np.complex64)
    kept = float(np.sort(values)[::-1][:COILS].sum() / values.sum())
    print(f"  {channels} channels -> {COILS}, holding {100 * kept:.1f}% of the energy")
    return keep


def _gather_compressed(
    raw: np.ndarray,
    offset: int,
    points: int,
) -> np.ndarray:
    """Read the acquisition into ``(coils, frames, shots, points)``.

    The raw array is a readout axis, a channel axis and one arm axis holding
    every group of every frame, and the arm axis is the contiguous one. Taking
    one arm at a time therefore strides the whole file per arm; taking a slab
    of the readout at a time reads it in order instead, and the channels are
    projected onto the virtual ones as each slab lands, so the uncompressed
    acquisition is never held whole.
    """
    channels, arms = raw.shape[1], raw.shape[2]
    slab = 64
    keep = _compression_matrix(
        np.ascontiguousarray(raw[offset : offset + slab].transpose(1, 0, 2))
    )
    kspace = np.empty((COILS, FRAMES, SHOTS, points), dtype=np.complex64)
    for start in range(0, points, slab):
        stop = min(start + slab, points)
        block = np.asarray(raw[offset + start : offset + stop])
        # The arm axis is group-major over the frames, and only the first
        # SHOTS groups are wanted.
        block = block.reshape(stop - start, channels, arms // FRAMES, FRAMES)
        block = block[:, :, :SHOTS].transpose(1, 3, 2, 0)
        kspace[..., start:stop] = (keep @ block.reshape(channels, -1)).reshape(
            COILS, FRAMES, SHOTS, stop - start
        )
        print(f"  {stop}/{points} samples", end="\r", flush=True)
    print(f"  gathered {kspace.shape}, {kspace.nbytes / 2**30:.2f} GiB")
    return kspace


def _resolve_layout(raw: np.ndarray, trajectory: np.ndarray) -> tuple[str, int]:
    """Return how the raw arms map onto the trajectory, having checked it.

    Neither the order of the 24000 arms nor which of the 2000 digitised
    samples carry the 1688 trajectory points is written down, and scoring a
    reconstruction cannot settle either. A misaligned gridding puts every
    arm's centre sample at k = 0 and scatters the rest, which is a bright
    point at the origin surrounded by streaks -- more concentrated than any
    image, so every metric that rewards concentration picks it.

    Two properties of the acquisition settle it without reconstructing:

    - The flip-angle train varies over the 500 frames and repeats for each of
      the 48 groups, so the arm axis is periodic in the frame with period 500.
      Averaged over the folds, folding that way leaves the train; folding
      the other way leaves a flat line.
    - Every arm is centre-out and the trajectory's first point is k = 0, so
      the largest sample of a readout is its first, and the trajectory's
      points are the leading ones of the 2000.

    Returns
    -------
    tuple
        The arm ordering and the sample the readout starts at.

    Raises
    ------
    RuntimeError
        If the data does not have the periodicity or the centre-out readout
        this pairing assumes.
    """
    frames, _, points, _ = trajectory.shape
    arms = raw.shape[2]
    if not float(np.linalg.norm(trajectory[0, 0, 0])) < 1e-3:
        raise RuntimeError("the trajectory does not start at k = 0")

    # One number per arm: how much signal its readout opens with.
    profile = np.abs(np.asarray(raw[:40, 0, :])).mean(axis=0).astype(np.float64)
    kept = profile[: (arms // frames) * frames]

    def variation(shape: tuple[int, int]) -> float:
        """How much the fold-averaged curve varies, relative to its level.

        Averaging over the folds separates the evolution from the noise on any
        one of them: fold on the true period and what is left is the train,
        fold on the wrong one and it is flat.
        """
        curve = kept.reshape(shape).mean(axis=0)
        return float(curve.std() / curve.mean())

    along_frames = variation((arms // frames, frames))
    along_groups = variation((frames, arms // frames))
    if along_frames < 4 * along_groups:
        raise RuntimeError(
            "the arm axis is not periodic in the frame with period "
            f"{frames} ({along_frames:.3f} against {along_groups:.3f}); the "
            "acquisition is not laid out the way this expects"
        )

    opening = np.abs(np.asarray(raw[:, 0, :64])).mean(axis=1)
    peak = int(opening[: 2 * (raw.shape[0] - points)].argmax())
    if peak > 16:
        raise RuntimeError(
            f"the readout peaks at sample {peak}, so it does not open at "
            "k = 0 and the trajectory's points are not the leading ones"
        )
    return "shot-major", 0


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

    assert order == "shot-major"
    kspace = _gather_compressed(raw, offset, trajectory.shape[2])

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
