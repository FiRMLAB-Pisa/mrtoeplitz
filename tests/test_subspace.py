"""The subspace normal, and the block contract it is built from."""

import numpy as np
import pytest
import torch

import mrtoeplitz as mt

SINGLE = mt.toeplitz_options(compress=False, cuda_transfer_precision="float32")


@pytest.fixture
def frames(radial):
    """Four frames, each on its own rotated radial trajectory."""
    mrinufft = pytest.importorskip("mrinufft")

    def build(shape=(24, 24), n_frames=4, rank=2, seed=0):
        rng = np.random.default_rng(seed)
        operators, samples = [], []
        for index in range(n_frames):
            base = radial(n_spokes=16, n_samples=48)
            angle = index * np.pi / (2 * n_frames)
            rotation = np.array(
                [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
                dtype=np.float32,
            )
            turned = (base @ rotation.T).astype(np.float32)
            samples.append(turned)
            operators.append(
                mrinufft.get_operator("finufft")(
                    samples=turned,
                    shape=shape,
                    density=None,
                    n_coils=1,
                    squeeze_dims=False,
                )
            )
        basis = rng.normal(size=(rank, n_frames)).astype(np.float32)
        return operators, samples, basis, shape

    return build


def _blocks(samples, basis):
    """The contract: one (samples, weights, coefficients) entry per trajectory."""
    rank = basis.shape[0]
    rows, columns = torch.triu_indices(rank, rank)
    tensor = torch.as_tensor(basis)
    return [
        (
            torch.as_tensor(trajectory),
            None,
            tensor[rows, frame] * tensor[columns, frame].conj(),
        )
        for frame, trajectory in enumerate(samples)
    ]


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


def test_the_subspace_kernel_reproduces_the_exact_subspace_gram(frames, device):
    operators, samples, basis, shape = frames()
    rng = np.random.default_rng(1)
    x = (rng.normal(size=(2, *shape)) + 1j * rng.normal(size=(2, *shape))).astype(
        np.complex64
    )

    kernel = mt.subspace_kernel(_blocks(samples, basis), shape, options=SINGLE)
    fast = np.asarray(
        kernel.to(device).apply(torch.as_tensor(x)[None].to(device)).detach().cpu()
    ).reshape(x.shape)

    truth = _exact_subspace_gram(operators, basis, x)
    error = np.linalg.norm(fast - truth) / np.linalg.norm(truth)
    assert error < 1e-3


def test_frames_sharing_a_trajectory_are_one_block(frames):
    """The saving the contract exists for.

    A subspace kernel costs one gridding transform per basis *pair*, over
    every frame's samples concatenated -- not one per frame. Frames acquired
    on the same trajectory collapse into a single block before any of that,
    which is why the caller groups them.
    """
    _, samples, basis, shape = frames(n_frames=4)
    rank = basis.shape[0]
    rows, columns = torch.triu_indices(rank, rank)
    tensor = torch.as_tensor(basis)

    shared = torch.as_tensor(samples[0])
    summed = sum(
        tensor[rows, frame] * tensor[columns, frame].conj() for frame in range(4)
    )
    one_block = mt.subspace_kernel([(shared, None, summed)], shape, options=SINGLE)

    assert one_block.rank == rank
    # rank (rank + 1) / 2 packed pairs, whatever the frame count was.
    assert one_block.values.shape[0] == rank * (rank + 1) // 2


def test_a_double_precision_basis_does_not_reach_the_kernel(frames):
    """A NumPy basis is float64 unless asked otherwise -- what qr and svd of a
    dictionary return -- and a float64 basis reaching the transfer was worth a
    factor of 214."""
    _, samples, basis, shape = frames()
    blocks = _blocks(samples, basis.astype(np.float64))
    kernel = mt.subspace_kernel(blocks, shape, options=SINGLE)
    assert kernel.values.dtype in {torch.float32, torch.complex64}


def test_coefficients_that_are_not_a_packed_triangle_are_refused(frames):
    _, samples, _, shape = frames()
    bad = [(torch.as_tensor(samples[0]), None, torch.ones(4))]
    with pytest.raises(ValueError, match="packed upper triangle"):
        mt.subspace_kernel(bad, shape)


def test_a_subspace_kernel_needs_at_least_one_block():
    with pytest.raises(ValueError, match="at least one trajectory block"):
        mt.subspace_kernel([], (16, 16))


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


def test_one_shared_cartesian_mask_serves_every_frame():
    rng = np.random.default_rng(0)
    shape = (16, 16)
    shared = (rng.random((1, *shape)) < 0.4).astype(np.float32)
    basis = rng.normal(size=(2, 4)).astype(np.float32)
    assert mt.cartesian_subspace_kernel(shared, basis).rank == 2


def test_a_cartesian_mask_count_that_is_neither_one_nor_the_frames_is_refused():
    rng = np.random.default_rng(0)
    masks = (rng.random((3, 16, 16)) < 0.4).astype(np.float32)
    basis = rng.normal(size=(2, 4)).astype(np.float32)
    with pytest.raises(ValueError, match="shared or have one mask per frame"):
        mt.cartesian_subspace_kernel(masks, basis)
