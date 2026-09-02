"""Streaming a transfer that does not fit the device.

A compressed transfer for a high-resolution subspace reconstruction can be
larger than the card. It is held in pinned host memory and brought over in
chunks, with the copy of one chunk overlapping the multiply of the one before.
"""

import pytest
import torch

import mrtoeplitz as mt

WHOLE = mt.toeplitz_options(compress=False, gridding_tolerance=1e-4)
#: The same, with the resident CUDA lane held at single precision. Its default
#: is bfloat16, which is two decimal digits away from anything a streamed lane
#: at float32 answers -- a yardstick has to be in the precision it measures.
EXACT = mt.toeplitz_options(
    compress=False,
    gridding_tolerance=1e-4,
    cuda_transfer_precision="float32",
)
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
    """A kernel built with a policy streams when called; nothing else changes."""
    trajectory, x, truth = resident
    policy = mt.CudaStreaming(streams=streams, transfer_precision="float32")
    kernel = mt.scalar_kernel(trajectory, (32, 32), options=WHOLE, streaming=policy)
    assert kernel.streaming is policy
    got = kernel(x)
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
        got = kernel(x).cpu()
        return float(
            torch.linalg.vector_norm(got - truth) / torch.linalg.vector_norm(truth)
        )

    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("no native bfloat16")
    assert error_at("float32") < 1e-5
    assert 1e-5 < error_at("auto") < 1e-2


@pytest.mark.cuda
@cuda_only
def test_a_policy_states_the_devices_it_will_use():
    policy = mt.CudaStreaming(streams=2)
    assert policy.streams == 2
    assert policy.device == "cuda"
    assert policy.transfer_precision == "auto"
    assert policy.device_count >= 1
    # One entry per stream worker, round-robin over the GPUs.
    assert len(policy.execution_devices) == 2 * policy.device_count


def test_naming_devices_describes_a_machine_you_are_not_on():
    """An explicit device list needs no driver to enumerate."""
    policy = mt.CudaStreaming(devices=("cuda:0", "cuda:1"))
    assert policy.device_count == 2
    assert len(policy.execution_devices) == 4


def test_covering_every_device_says_so_when_there_are_none():
    policy = mt.CudaStreaming()
    if torch.cuda.device_count():
        pytest.skip("this machine has a CUDA device")
    with pytest.raises(RuntimeError, match="there are none"):
        _ = policy.device_count


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


@pytest.mark.cuda
@cuda_only
def test_a_streamed_transfer_is_differentiable(resident):
    """The streamed lane is reached through the same Hermitian backward."""
    trajectory, _, _ = resident
    policy = mt.CudaStreaming(streams=2, transfer_precision="float32")
    kernel = mt.scalar_kernel(trajectory, (32, 32), options=WHOLE, streaming=policy)

    image = torch.randn(1, 1, 32, 32, dtype=torch.complex64, requires_grad=True)
    seed = torch.randn(1, 1, 32, 32, dtype=torch.complex64)
    kernel(image).backward(seed)

    with torch.no_grad():
        expected = kernel(seed)
    error = torch.linalg.vector_norm(image.grad - expected)
    assert float(error / torch.linalg.vector_norm(expected)) < 1e-5


def test_a_kernel_built_without_a_policy_holds_none(radial):
    kernel = mt.scalar_kernel(radial(n_spokes=12, n_samples=16), (8, 8))
    assert kernel.streaming is None


@pytest.mark.cuda
@cuda_only
def test_the_streamed_lane_agrees_with_itself_over_many_applications(resident):
    """A stream-ordering fault shows up as a rate, not as a failure.

    The chunk loop runs on side streams over tensors the default stream wrote,
    and unordered it can read them before they are finished -- a wrong answer
    rather than a slow one, and an intermittent one.

    It takes a few hundred applications to appear: the first disagreement in
    the run that found it was the 104th, after which it settled at 13%. Sixty
    trials pass with the fault still in, so this is deliberately long rather
    than trimmed to something quicker that would not catch it. It costs
    nothing where there is no CUDA device, which is where CI runs.
    """
    trajectory, _, _ = resident
    worst = 0.0
    for _ in range(400):
        policy = mt.CudaStreaming(streams=2, transfer_precision="float32")
        kernel = mt.scalar_kernel(trajectory, (32, 32), options=WHOLE, streaming=policy)
        image = torch.randn(1, 1, 32, 32, dtype=torch.complex64, requires_grad=True)
        seed = torch.randn(1, 1, 32, 32, dtype=torch.complex64)
        kernel(image).backward(seed)
        with torch.no_grad():
            expected = kernel(seed)
        error = torch.linalg.vector_norm(image.grad - expected)
        worst = max(worst, float(error / torch.linalg.vector_norm(expected)))
    assert worst < 1e-5


@pytest.fixture
def sense(radial, image):
    """A multicoil normal applied the ordinary way, as the yardstick."""
    trajectory = radial(n_spokes=32, n_samples=64)
    x = torch.as_tensor(image(shape=(32, 32)))[None, None]
    torch.manual_seed(0)
    maps = torch.randn(4, 32, 32, dtype=torch.complex64)
    maps = maps / maps.abs().amax()
    # A transfer on the device stays there: that is how a caller asks for the
    # resident lane, which is the yardstick a streamed one is measured against.
    kernel = mt.scalar_kernel(trajectory, (32, 32), options=EXACT).to("cuda")
    truth = mt.apply_sense(kernel, x.cuda(), maps.cuda(), coil_batch_size=1)
    return trajectory, x, maps, truth


@pytest.mark.cuda
@cuda_only
@pytest.mark.parametrize("coil_group", [1, 2, 4])
def test_a_streamed_sense_application_agrees_with_the_resident_one(sense, coil_group):
    """The group changes how often the transfer is read, not the answer."""
    trajectory, x, maps, truth = sense
    policy = mt.CudaStreaming(streams=2, transfer_precision="float32")
    kernel = mt.scalar_kernel(trajectory, (32, 32), options=EXACT)
    got = kernel._apply_sense_streamed(
        x.cuda(), maps.cuda(), policy, coil_group=coil_group
    )
    error = torch.linalg.vector_norm(got - truth) / torch.linalg.vector_norm(truth)
    assert float(error) < 1e-5


@pytest.mark.cuda
@cuda_only
def test_a_coil_group_reads_the_transfer_once_for_every_coil_in_it(sense, monkeypatch):
    """The transfer does not depend on the coil, so a group reads it once.

    Reading it per coil is what a streamed application costs most, and it is
    the same bytes every time. Counting the chunk multiplies is what
    distinguishes one walk of it from one per coil.
    """
    from mrtoeplitz import _kernel

    trajectory, x, maps, _ = sense
    policy = mt.CudaStreaming(streams=2, transfer_precision="float32")
    kernel = mt.scalar_kernel(trajectory, (32, 32), options=EXACT)

    def walks(coil_group):
        calls = 0
        original = _kernel._packed_cuda_matvec

        def counted(*arguments, **keywords):
            nonlocal calls
            calls += 1
            return original(*arguments, **keywords)

        monkeypatch.setattr(_kernel, "_packed_cuda_matvec", counted)
        kernel._apply_sense_streamed(
            x.cuda(), maps.cuda(), policy, coil_group=coil_group
        )
        monkeypatch.undo()
        return calls

    assert walks(4) * 4 == walks(1)


@pytest.mark.cuda
@cuda_only
def test_a_kernel_built_for_a_policy_streams_its_sense_application(sense):
    """Streaming is what the kernel was built for, not an argument to repeat."""
    trajectory, x, maps, truth = sense
    policy = mt.CudaStreaming(streams=2, transfer_precision="float32")
    kernel = mt.scalar_kernel(trajectory, (32, 32), options=EXACT, streaming=policy)
    assert kernel.values.device.type == "cpu"
    got = mt.apply_sense(kernel, x.cuda(), maps.cuda())
    assert kernel.values.device.type == "cpu"
    error = torch.linalg.vector_norm(got - truth) / torch.linalg.vector_norm(truth)
    assert float(error) < 1e-5


@pytest.mark.cuda
@cuda_only
def test_the_streamed_sense_lane_agrees_with_itself_over_many_applications(sense):
    """A stream-ordering fault shows up as a rate, not as a failure.

    The chunk loop runs on side streams over spectra the default stream wrote,
    and unordered it reads them before they are finished. Four hundred trials
    is what it took to see the scalar lane's fault at all; this holds the
    coil loop to the same standard.
    """
    trajectory, x, maps, truth = sense
    policy = mt.CudaStreaming(streams=2, transfer_precision="float32")
    kernel = mt.scalar_kernel(trajectory, (32, 32), options=EXACT)
    scale = torch.linalg.vector_norm(truth)
    worst = 0.0
    for trial in range(400):
        got = kernel._apply_sense_streamed(
            x.cuda(), maps.cuda(), policy, coil_group=1 + trial % 4
        )
        worst = max(worst, float(torch.linalg.vector_norm(got - truth) / scale))
    assert worst < 1e-5


@pytest.mark.cuda
@cuda_only
def test_a_kept_map_is_expanded_once_for_the_whole_component_sweep(sense):
    """A map does not depend on the parity component that reads it.

    A transfer filed by parity applies one component after another over the
    same image grid, and expanding a bank's coil afresh for each of them is
    that work repeated. What the device has room to keep, it keeps.
    """
    trajectory, x, maps, _ = sense
    policy = mt.CudaStreaming(streams=2, transfer_precision="float32")
    kernel = mt.scalar_kernel(trajectory, (32, 32), options=EXACT)
    bank = mt.CoilKernels.from_maps(maps, (8, 8)).to("cuda")

    class Counted:
        """A bank that records how often it is asked to expand a coil."""

        def __init__(self, held):
            self.held = held
            self.expansions = 0

        def __getattr__(self, name):
            return getattr(self.held, name)

        def __getitem__(self, index):
            self.expansions += 1
            return self.held[index]

    def run(map_cache):
        counted = Counted(bank)
        got = kernel._apply_sense_streamed(
            x.cuda(),
            counted,
            policy,
            coil_group=len(maps),
            map_cache=map_cache,
        )
        return counted.expansions, got

    kept, with_cache = run(len(maps))
    fresh, without_cache = run(0)
    assert kept == len(maps)
    assert fresh > kept
    # What is kept is what would have been expanded again, so keeping it
    # cannot change the answer.
    assert torch.equal(with_cache, without_cache)


@pytest.mark.cuda
@cuda_only
def test_a_staged_build_grids_the_same_transfer_as_an_unstaged_one(radial):
    """Staging is where the rows go, not what they are.

    A build under a policy puts each basis pair's rows on the host as it
    finishes them, on a stream of their own so the copy overlaps the gridding
    of the next pair. The gridding itself is untouched, and what it produces
    has to agree to what a build agrees with itself to -- which is not exact:
    the spreading behind it accumulates in parallel.
    """
    torch.manual_seed(0)
    trajectory = torch.as_tensor(radial(n_spokes=48, n_samples=96)).cuda()
    basis = torch.linalg.qr(torch.randn(60, 3))[0]

    def transfer(streaming):
        kernel = mt.subspace_kernel(
            trajectory, basis, (32, 32), options=WHOLE, streaming=streaming
        )
        return torch.cat([part.values.cpu().flatten() for _, part in kernel.components])

    reproducible = float((transfer(None) - transfer(None)).abs().max())
    policy = mt.CudaStreaming(streams=2)
    staged = float((transfer(policy) - transfer(None)).abs().max())
    assert staged <= max(reproducible * 4, 1e-9)


@pytest.mark.cuda
@cuda_only
def test_a_host_held_transfer_is_streamed_without_being_asked(sense):
    """The default: what a build left on the host is not pulled across.

    A transfer is the largest thing in a subspace reconstruction. Moving it
    onto the card to apply it puts it there for the length of the solve, and
    the point of the package is that it does not have to be.
    """
    trajectory, x, maps, truth = sense
    kernel = mt.scalar_kernel(trajectory, (32, 32), options=EXACT)
    assert kernel.values.device.type == "cpu"
    got = mt.apply_sense(kernel, x.cuda(), maps.cuda())
    assert kernel.values.device.type == "cpu"
    error = torch.linalg.vector_norm(got - truth) / torch.linalg.vector_norm(truth)
    # Streamed by default is streamed in bfloat16 by default, which is what
    # halves what crosses; two decimal digits is what it costs.
    assert float(error) < 5e-3


@pytest.mark.cuda
@cuda_only
def test_a_transfer_put_on_the_device_is_left_there(sense):
    """Moving it across is a decision to keep it across."""
    trajectory, x, maps, _ = sense
    kernel = mt.scalar_kernel(trajectory, (32, 32), options=EXACT).to("cuda")
    mt.apply_sense(kernel, x.cuda(), maps.cuda())
    assert kernel.values.device.type == "cuda"


@pytest.mark.cuda
@cuda_only
def test_a_kernel_gives_the_device_back_when_it_is_released(sense):
    """What an application keeps is the kernel's, and goes when the kernel does.

    A kernel holds the buffers its lane works out of between calls so a solve
    does not allocate them per iteration. Nothing else may hold them: releasing
    the kernel has to return every byte of it.
    """
    trajectory, x, maps, _ = sense
    kernel = mt.scalar_kernel(trajectory, (32, 32), options=EXACT)
    image, coils = x.cuda(), maps.cuda()

    def resident():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return torch.cuda.memory_allocated()

    before = resident()
    for _ in range(3):
        mt.apply_sense(kernel, image, coils)
    held = resident() - before
    assert held > 0, "an application is expected to keep its working buffers"
    kernel.release()
    assert resident() == before


@pytest.mark.cuda
@cuda_only
def test_the_parities_of_a_transfer_divide_and_add(radial, image):
    """A share of the parities is a transfer, and the shares add.

    This is what lets the components be dealt out to several devices: each is
    a transfer over the image grid in its own right, and the normalisation one
    carries is over the whole doubled grid rather than over the components
    present, so shares answer on one scale however they are cut.
    """
    from mrtoeplitz._kernel import PolyphaseToeplitzKernel

    torch.manual_seed(0)
    trajectory = torch.as_tensor(radial(n_spokes=48, n_samples=64)).cuda()
    basis = torch.linalg.qr(torch.randn(48, 2))[0]
    kernel = mt.subspace_kernel(trajectory, basis, (16, 16))
    maps = torch.randn(3, 16, 16, dtype=torch.complex64).cuda()
    x = torch.randn(1, 2, 16, 16, dtype=torch.complex64).cuda()

    def share(start, stop):
        return PolyphaseToeplitzKernel(
            kernel.components[start:stop],
            kernel.image_shape,
            kernel.rank,
            truncation_bound=kernel.truncation_bound,
        )

    whole = mt.apply_sense(kernel, x, maps)
    parts = len(kernel.components)
    for cuts in ((parts // 2,), (1, parts - 1)):
        edges = (0, *cuts, parts)
        total = sum(
            mt.apply_sense(share(edges[i], edges[i + 1]), x, maps)
            for i in range(len(edges) - 1)
        )
        error = torch.linalg.vector_norm(total - whole) / torch.linalg.vector_norm(
            whole
        )
        assert float(error) < 1e-5, f"cut at {cuts}"


@pytest.mark.cuda
@cuda_only
def test_the_batch_divides_with_nothing_to_sum(radial):
    """Entries of the batch are independent applications, so they only join."""
    torch.manual_seed(0)
    trajectory = torch.as_tensor(radial(n_spokes=48, n_samples=64)).cuda()
    basis = torch.linalg.qr(torch.randn(48, 2))[0]
    kernel = mt.subspace_kernel(trajectory, basis, (16, 16))
    maps = torch.randn(3, 16, 16, dtype=torch.complex64).cuda()
    x = torch.randn(3, 2, 16, 16, dtype=torch.complex64).cuda()

    whole = mt.apply_sense(kernel, x, maps)
    joined = torch.cat(
        [mt.apply_sense(kernel, x[start : start + 1], maps) for start in range(3)]
    )
    error = torch.linalg.vector_norm(joined - whole) / torch.linalg.vector_norm(whole)
    assert float(error) < 1e-5


def test_the_cheapest_axis_that_divides_is_the_one_taken(monkeypatch):
    """Batch before parities before coils, and none of them when one will do.

    The batch costs no sum at all, the parities divide the transfer between the
    cards, and coils make every card hold the whole of it. So they are tried in
    that order, and the choice does not depend on having the cards to hand.
    """
    from mrtoeplitz import _sense

    taken = []
    for name in ("_batches", "_parities", "_coils"):
        monkeypatch.setattr(
            _sense,
            f"{name}_split_across_devices",
            lambda *a, name=name, **k: taken.append(name) or "answered",
        )

    class Policy:
        device_count = 4

    class Filed:
        components = [(0, None)] * 8

    def choose(kernel, batch, coils):
        taken.clear()
        image = torch.zeros(batch, 1, 4, 4)
        _sense._divided_across_devices(
            kernel, image, None, Policy(), batched_maps=False, n_coils=coils
        )
        return taken

    assert choose(Filed(), batch=3, coils=8) == ["_batches"]
    assert choose(Filed(), batch=1, coils=8) == ["_parities"]
    assert choose(object(), batch=1, coils=8) == ["_coils"]
    assert choose(object(), batch=1, coils=1) == []
