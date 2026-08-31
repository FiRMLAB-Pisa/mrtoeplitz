"""Streaming a transfer that does not fit the device.

A compressed transfer for a high-resolution subspace reconstruction can be
larger than the card. It is held in pinned host memory and brought over in
chunks, with the copy of one chunk overlapping the multiply of the one before.
"""

import pytest
import torch

import mrtoeplitz as mt

WHOLE = mt.toeplitz_options(compress=False)
cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")


@pytest.fixture
def resident(radial, image):
    """The same operator applied the ordinary way, as the yardstick."""
    trajectory = radial(n_spokes=32, n_samples=64)
    x = torch.as_tensor(image(shape=(32, 32)))[None, None]
    kernel = mt.scalar_kernel(trajectory, (32, 32), options=WHOLE)
    return trajectory, x, kernel(x)


@pytest.mark.cuda
@cuda_only
@pytest.mark.parametrize("streams", [1, 2])
def test_a_streamed_transfer_computes_the_same_operator(resident, streams):
    trajectory, x, truth = resident
    policy = mt.CudaStreaming(streams=streams, transfer_precision="float32")
    kernel = mt.scalar_kernel(trajectory, (32, 32), options=WHOLE, streaming=policy)
    got = kernel.apply_streamed(x, policy)
    error = torch.linalg.vector_norm(got.cpu() - truth) / torch.linalg.vector_norm(
        truth
    )
    assert float(error) < 1e-5


@pytest.mark.cuda
@cuda_only
def test_a_streamed_transfer_is_held_on_the_host(resident):
    """The point: what does not fit the device does not go on it."""
    trajectory, _, _ = resident
    policy = mt.CudaStreaming(streams=2)
    kernel = mt.scalar_kernel(trajectory, (32, 32), options=WHOLE, streaming=policy)
    assert kernel.values.device.type == "cpu"


@pytest.mark.cuda
@cuda_only
def test_streaming_narrows_the_transfer_to_bfloat16_by_default(resident):
    """``auto`` halves what a transfer occupies and costs two decimal digits."""
    trajectory, x, truth = resident

    def error_at(precision):
        policy = mt.CudaStreaming(streams=2, transfer_precision=precision)
        kernel = mt.scalar_kernel(trajectory, (32, 32), options=WHOLE, streaming=policy)
        got = kernel.apply_streamed(x, policy).cpu()
        return float(
            torch.linalg.vector_norm(got - truth) / torch.linalg.vector_norm(truth)
        )

    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("no native bfloat16")
    assert error_at("float32") < 1e-5
    assert 1e-5 < error_at("auto") < 1e-2


def test_a_policy_states_the_devices_it_will_use():
    policy = mt.CudaStreaming(streams=2)
    assert policy.streams == 2
    assert policy.device == "cuda"
    assert policy.transfer_precision == "auto"
    assert policy.device_count >= 1
    # One entry per stream worker, round-robin over the GPUs.
    assert len(policy.execution_devices) == 2 * policy.device_count


def test_a_policy_can_be_narrowed_to_one_device():
    narrowed = mt.CudaStreaming().for_device("cuda:0", streams=1)
    assert narrowed.devices == ("cuda:0",)
    assert narrowed.streams == 1
    assert narrowed.device_count == 1


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("streams", 3, "one or two"),
        ("transfer_chunk_size", 0, "must be positive"),
        ("kernel_residency", "somewhere", "auto"),
        ("transfer_precision", "float16", "float32"),
        ("max_device_fraction", 1.5, r"\(0, 1\]"),
        ("device", "cpu", "must be a CUDA device"),
    ],
)
def test_a_policy_that_cannot_be_honoured_is_refused(field, value, message):
    with pytest.raises(ValueError, match=message):
        mt.CudaStreaming(**{field: value})


def test_duplicate_devices_are_refused():
    with pytest.raises(ValueError, match="unique"):
        mt.CudaStreaming(devices=("cuda:0", "cuda:0"))
