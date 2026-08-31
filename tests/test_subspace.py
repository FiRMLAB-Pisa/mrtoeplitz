"""The subspace normal, built from a trajectory and a basis."""

import numpy as np
import pytest
import torch

import mrtoeplitz as mt

SINGLE = mt.toeplitz_options(compress=False, cuda_transfer_precision="float32")


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


@pytest.fixture
def operators_for():
    """One MRI-NUFFT operator per frame, for the reference Gram."""
    mrinufft = pytest.importorskip("mrinufft")

    def build(trajectory, shape):
        return [
            mrinufft.get_operator("finufft")(
                samples=frame.reshape(-1, trajectory.shape[-1]),
                shape=shape,
                density=None,
                n_coils=1,
                squeeze_dims=False,
            )
            for frame in trajectory
        ]

    return build


def _exact_subspace_gram(operators, basis, x):
    """``sum_t conj(U[j,t]) A_t^H A_t (sum_i U[i,t] x_i)``, the slow way."""
    rank, n_frames = basis.shape
    shape = x.shape[1:]
    out = np.zeros_like(x)
    for frame in range(n_frames):
        image = sum(basis[i, frame] * x[i] for i in range(rank))
        operator = operators[frame]
        back = np.asarray(
            operator.adj_op(operator.op(image.reshape(1, 1, *shape)))
        ).reshape(shape)
        for j in range(rank):
            out[j] += np.conj(basis[j, frame]) * back
    return out


def test_the_subspace_kernel_reproduces_the_exact_subspace_gram(
    rotated, operators_for, device
):
    trajectory, shape = rotated(n_frames=4)
    rng = np.random.default_rng(1)
    basis = rng.normal(size=(2, 4)).astype(np.float32)
    x = (rng.normal(size=(2, *shape)) + 1j * rng.normal(size=(2, *shape))).astype(
        np.complex64
    )

    kernel = mt.subspace_kernel(trajectory, basis, shape, options=SINGLE)
    fast = np.asarray(
        kernel.to(device).apply(torch.as_tensor(x)[None].to(device)).detach().cpu()
    ).reshape(x.shape)

    truth = _exact_subspace_gram(operators_for(trajectory, shape), basis, x)
    assert np.linalg.norm(fast - truth) / np.linalg.norm(truth) < 1e-3


def test_a_shared_trajectory_reproduces_the_exact_subspace_gram(rotated, operators_for):
    """One trajectory every frame was acquired on, stated by its rank."""
    trajectory, shape = rotated(n_frames=1)
    shared = trajectory[0]
    rng = np.random.default_rng(2)
    basis = rng.normal(size=(2, 5)).astype(np.float32)
    x = (rng.normal(size=(2, *shape)) + 1j * rng.normal(size=(2, *shape))).astype(
        np.complex64
    )

    kernel = mt.subspace_kernel(shared, basis, shape, options=SINGLE)
    fast = np.asarray(kernel.apply(torch.as_tensor(x)[None]).detach().cpu()).reshape(
        x.shape
    )

    repeated = np.stack([shared] * 5)
    truth = _exact_subspace_gram(operators_for(repeated, shape), basis, x)
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


def test_grouping_does_not_change_the_operator(rotated, operators_for):
    """Ten frames on one rotation must answer as ten separate frames would."""
    trajectory, shape = rotated(n_frames=2)
    cycled = np.stack([trajectory[index % 2] for index in range(6)])
    rng = np.random.default_rng(5)
    basis = rng.normal(size=(2, 6)).astype(np.float32)
    x = (rng.normal(size=(2, *shape)) + 1j * rng.normal(size=(2, *shape))).astype(
        np.complex64
    )

    kernel = mt.subspace_kernel(cycled, basis, shape, options=SINGLE)
    fast = np.asarray(kernel.apply(torch.as_tensor(x)[None]).detach().cpu()).reshape(
        x.shape
    )
    truth = _exact_subspace_gram(operators_for(cycled, shape), basis, x)
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
