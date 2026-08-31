"""The subspace normal, built from a trajectory and a basis."""

import numpy as np
import pytest
import torch

import mrtoeplitz as mt

SINGLE = mt.toeplitz_options(
    compress=False, cuda_transfer_precision="float32", gridding_tolerance=1e-4
)


@pytest.fixture
def rotated(radial):
    """Per-frame rotated radial trajectories, shaped (frames, shots, points, axes)."""

    def build(n_frames=4, shape=(24, 24), n_spokes=16, n_samples=48):
        frames = []
        for index in range(n_frames):
            flat = radial(n_spokes, n_samples).reshape(n_spokes, n_samples, 2)
            angle = index * np.pi / (2 * n_frames)
            rotation = np.array(
                [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
                dtype=np.float32,
            )
            frames.append((flat @ rotation.T).astype(np.float32))
        return np.stack(frames), shape

    return build


def _exact_subspace_gram(exact_gram, trajectory, shape, basis, x):
    """``sum_t conj(U[j,t]) A_t^H A_t (sum_i U[i,t] x_i)``, the slow way."""
    rank, n_frames = basis.shape
    out = np.zeros_like(x)
    for frame in range(n_frames):
        image = sum(basis[i, frame] * x[i] for i in range(rank))
        back = exact_gram(trajectory[frame], shape, image)
        for j in range(rank):
            out[j] += np.conj(basis[j, frame]) * back
    return out


def test_the_subspace_kernel_reproduces_the_exact_subspace_gram(
    rotated, exact_gram, device
):
    trajectory, shape = rotated(n_frames=4)
    rng = np.random.default_rng(1)
    basis = rng.normal(size=(2, 4)).astype(np.float32)
    x = (rng.normal(size=(2, *shape)) + 1j * rng.normal(size=(2, *shape))).astype(
        np.complex64
    )

    kernel = mt.subspace_kernel(trajectory, basis, shape, options=SINGLE)
    fast = np.asarray(
        kernel.to(device)(torch.as_tensor(x)[None].to(device)).detach().cpu()
    ).reshape(x.shape)

    truth = _exact_subspace_gram(exact_gram, trajectory, shape, basis, x)
    assert np.linalg.norm(fast - truth) / np.linalg.norm(truth) < 1e-3


def test_a_shared_trajectory_reproduces_the_exact_subspace_gram(rotated, exact_gram):
    """One trajectory every frame was acquired on, stated by its rank."""
    trajectory, shape = rotated(n_frames=1)
    shared = trajectory[0]
    rng = np.random.default_rng(2)
    basis = rng.normal(size=(2, 5)).astype(np.float32)
    x = (rng.normal(size=(2, *shape)) + 1j * rng.normal(size=(2, *shape))).astype(
        np.complex64
    )

    kernel = mt.subspace_kernel(shared, basis, shape, options=SINGLE)
    fast = np.asarray(kernel(torch.as_tensor(x)[None]).detach().cpu()).reshape(x.shape)

    repeated = np.stack([shared] * 5)
    truth = _exact_subspace_gram(exact_gram, repeated, shape, basis, x)
    assert np.linalg.norm(fast - truth) / np.linalg.norm(truth) < 1e-3


def test_the_basis_may_come_either_way_round(rotated):
    trajectory, shape = rotated(n_frames=6)
    rng = np.random.default_rng(3)
    basis = rng.normal(size=(6, 2)).astype(np.float32)
    upright = mt.subspace_kernel(trajectory, basis, shape, options=SINGLE)
    sideways = mt.subspace_kernel(trajectory, basis.T, shape, options=SINGLE)
    assert upright.rank == sideways.rank == 2
    assert torch.allclose(upright.values, sideways.values, atol=1e-6)


def test_frames_sharing_a_trajectory_are_gridded_once(rotated, monkeypatch):
    """The saving the grouping exists for.

    A fingerprinting scan cycles its frames through a handful of rotations.
    Carrying each frame separately would grid every basis pair over a thousand
    copies of the same samples.
    """
    from mrtoeplitz import _build

    trajectory, shape = rotated(n_frames=4)
    cycled = np.stack([trajectory[index % 4] for index in range(40)])
    rng = np.random.default_rng(4)
    basis = rng.normal(size=(2, 40)).astype(np.float32)

    seen = {}
    original = _build._subspace_kernel_from_blocks

    def record(blocks, *args, **kwargs):
        seen["blocks"] = len(blocks)
        seen["samples"] = sum(block[0].shape[0] for block in blocks)
        return original(blocks, *args, **kwargs)

    monkeypatch.setattr(_build, "_subspace_kernel_from_blocks", record)
    mt.subspace_kernel(cycled, basis, shape, options=SINGLE)

    assert seen["blocks"] == 4
    # Ten frames collapsed onto each of the four rotations.
    assert seen["samples"] == 4 * 16 * 48


def test_grouping_does_not_change_the_operator(rotated, exact_gram):
    """Ten frames on one rotation must answer as ten separate frames would."""
    trajectory, shape = rotated(n_frames=2)
    cycled = np.stack([trajectory[index % 2] for index in range(6)])
    rng = np.random.default_rng(5)
    basis = rng.normal(size=(2, 6)).astype(np.float32)
    x = (rng.normal(size=(2, *shape)) + 1j * rng.normal(size=(2, *shape))).astype(
        np.complex64
    )

    kernel = mt.subspace_kernel(cycled, basis, shape, options=SINGLE)
    fast = np.asarray(kernel(torch.as_tensor(x)[None]).detach().cpu()).reshape(x.shape)
    truth = _exact_subspace_gram(exact_gram, cycled, shape, basis, x)
    assert np.linalg.norm(fast - truth) / np.linalg.norm(truth) < 1e-3


def test_a_density_may_be_given_per_frame_or_once(rotated):
    trajectory, shape = rotated(n_frames=3)
    rng = np.random.default_rng(6)
    basis = rng.normal(size=(2, 3)).astype(np.float32)
    per_sample = np.abs(rng.normal(size=(16, 48))).astype(np.float32) + 0.1

    once = mt.subspace_kernel(
        trajectory, basis, shape, density=per_sample, options=SINGLE
    )
    everywhere = mt.subspace_kernel(
        trajectory,
        basis,
        shape,
        density=np.stack([per_sample] * 3),
        options=SINGLE,
    )
    assert torch.allclose(once.values, everywhere.values, atol=1e-5)


def test_a_double_precision_basis_does_not_reach_the_kernel(rotated):
    """A NumPy basis is float64 unless asked otherwise -- what qr and svd of a
    dictionary return -- and a float64 basis reaching the transfer was worth a
    factor of 214."""
    trajectory, shape = rotated()
    basis = np.random.default_rng(7).normal(size=(2, 4))
    kernel = mt.subspace_kernel(trajectory, basis, shape, options=SINGLE)
    assert kernel.values.dtype in {torch.float32, torch.complex64}


def test_a_trajectory_of_the_wrong_rank_is_refused(rotated):
    trajectory, shape = rotated()
    with pytest.raises(ValueError, match=r"frames, shots, points, axes"):
        mt.subspace_kernel(trajectory[0, 0], np.eye(2, 4), shape)


def test_a_trajectory_whose_axes_do_not_match_the_image_is_refused(rotated):
    trajectory, _ = rotated()
    with pytest.raises(ValueError, match="axes it names"):
        mt.subspace_kernel(trajectory, np.eye(2, 4), (24, 24, 24))


def test_a_basis_with_no_axis_matching_the_frames_is_refused(rotated):
    trajectory, shape = rotated(n_frames=4)
    with pytest.raises(ValueError, match="no axis matching"):
        mt.subspace_kernel(trajectory, np.eye(2, 7), shape)


def test_a_density_that_does_not_fit_the_samples_is_refused(rotated):
    trajectory, shape = rotated(n_frames=3)
    with pytest.raises(ValueError, match="neither one per sample"):
        mt.subspace_kernel(
            trajectory, np.eye(2, 3), shape, density=np.ones(7, dtype=np.float32)
        )


def test_the_cartesian_subspace_kernel_lives_on_the_image_grid():
    """A Cartesian encoding needs no gridding, so no doubled grid either."""
    rng = np.random.default_rng(0)
    shape = (16, 16)
    masks = (rng.random((4, *shape)) < 0.4).astype(np.float32)
    basis = rng.normal(size=(2, 4)).astype(np.float32)

    kernel = mt.cartesian_subspace_kernel(masks, basis)
    assert kernel.spatial_shape == shape
    assert kernel.image_shape == shape
    assert kernel.rank == 2


def test_the_cartesian_basis_may_come_either_way_round():
    rng = np.random.default_rng(0)
    masks = (rng.random((4, 16, 16)) < 0.4).astype(np.float32)
    basis = rng.normal(size=(4, 2)).astype(np.float32)
    upright = mt.cartesian_subspace_kernel(masks, basis)
    sideways = mt.cartesian_subspace_kernel(masks, basis.T)
    assert torch.allclose(upright.values, sideways.values, atol=1e-6)


def test_one_shared_cartesian_mask_serves_every_frame():
    rng = np.random.default_rng(0)
    shared = (rng.random((1, 16, 16)) < 0.4).astype(np.float32)
    basis = rng.normal(size=(2, 4)).astype(np.float32)
    assert mt.cartesian_subspace_kernel(shared, basis).rank == 2


def test_a_cartesian_mask_count_that_is_neither_one_nor_the_frames_is_refused():
    rng = np.random.default_rng(0)
    masks = (rng.random((3, 16, 16)) < 0.4).astype(np.float32)
    basis = rng.normal(size=(2, 4)).astype(np.float32)
    with pytest.raises(
        ValueError, match=r"shared or have one mask per frame|no axis matching"
    ):
        mt.cartesian_subspace_kernel(masks, basis)


def test_a_compressed_subspace_kernel_keeps_what_the_trajectory_reached(rotated):
    """The subspace path reads the trajectory on the same scale as the scalar
    one, so compression keeps a sensible share rather than a fortieth of it."""
    trajectory, shape = rotated(n_frames=4, shape=(64, 64), n_spokes=64, n_samples=128)
    basis = np.random.default_rng(8).normal(size=(2, 4)).astype(np.float32)

    cut = mt.subspace_kernel(
        trajectory, basis, shape, options=mt.toeplitz_options(compress=True)
    )
    whole = mt.subspace_kernel(trajectory, basis, shape, options=SINGLE)
    kept = cut.n_locations / whole.n_locations
    assert 0.5 < kept < 1.0


def test_either_basis_orientation_gives_the_same_kernel(rotated):
    """The caller does not have to know which way round the basis is."""
    trajectory, shape = rotated(n_frames=6)
    rng = np.random.default_rng(4)
    basis = rng.normal(size=(6, 2)).astype(np.float32)  # (frames, rank)

    upright = mt.subspace_kernel(trajectory, basis, shape, options=SINGLE)
    flipped = mt.subspace_kernel(trajectory, basis.T, shape, options=SINGLE)
    x = torch.as_tensor(
        (rng.normal(size=(2, *shape)) + 1j * rng.normal(size=(2, *shape))).astype(
            np.complex64
        )
    )[None]
    reference = upright(x)
    difference = (reference - flipped(x)).abs().max() / reference.abs().max()
    assert float(difference) < 1e-6


def test_a_square_basis_reads_as_frames_by_rank(rotated):
    """A basis that compresses nothing is a real case, and it is ambiguous.

    Both readings fit a square basis, so one of them has to be chosen. It is
    ``(frames, rank)`` -- the form the documentation leads with -- and it has
    to be the same choice whether or not the trajectory stated a frame count,
    or a full temporal basis is silently transposed for one caller and not the
    other.
    """
    trajectory, shape = rotated(n_frames=3)
    rng = np.random.default_rng(5)
    basis = rng.normal(size=(3, 3)).astype(np.float32)

    per_frame = mt.subspace_kernel(trajectory, basis, shape, options=SINGLE)
    shared = mt.subspace_kernel(trajectory[0], basis, shape, options=SINGLE)
    # Transposing the basis by hand must reach the other reading, in both.
    other = mt.subspace_kernel(trajectory, basis.T, shape, options=SINGLE)

    x = torch.as_tensor(
        (rng.normal(size=(3, *shape)) + 1j * rng.normal(size=(3, *shape))).astype(
            np.complex64
        )
    )[None]
    assert per_frame.rank == shared.rank == 3
    reference = per_frame(x)
    # The two readings are genuinely different operators, so transposing by
    # hand has to reach the other one rather than being absorbed.
    difference = (reference - other(x)).abs().max() / reference.abs().max()
    assert float(difference) > 1e-3


def test_a_build_does_not_keep_its_gridding_plan(rotated):
    """The plan is the largest allocation a build makes, larger than what it
    produces, and the solve that follows needs that memory for its own
    transforms. Every builder has to release it, not just the scalar one.
    """
    from mrtoeplitz import _psf

    trajectory, shape = rotated(n_frames=4)
    rng = np.random.default_rng(7)
    basis = rng.normal(size=(4, 2)).astype(np.float32)

    _psf._PLAN_SLOT.clear()
    mt.subspace_kernel(trajectory, basis, shape, options=SINGLE)
    assert not _psf._PLAN_SLOT

    masks = (rng.random((4, *shape)) < 0.5).astype(np.float32)
    mt.cartesian_subspace_kernel(masks, basis, options=SINGLE)
    assert not _psf._PLAN_SLOT
