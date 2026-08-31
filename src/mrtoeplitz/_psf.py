"""Gridding a point spread function onto the doubled grid.

Building a transfer needs exactly one thing of a NUFFT: a type-1 transform,
nonuniform samples onto a uniform grid. FINUFFT does that on the host and
CUFINUFFT on the device, and CUFINUFFT takes Torch tensors directly -- it
reads anything exposing ``__cuda_array_interface__`` -- so a CUDA build never
leaves Torch and never needs a second array library.

Which of the two runs is decided by where the trajectory is. A trajectory on
the host builds a host transfer, one on a device builds it there, and a kernel
moves afterwards with :meth:`~mrtoeplitz.CompactToeplitzKernel.to`.
"""

from __future__ import annotations

__all__ = ["PsfPlan", "psf_plan", "psf_plans", "within_psf_plans"]

from contextlib import contextmanager, suppress
from functools import wraps
from importlib import import_module
from typing import Any

#: A NUFFT spreads onto a grid of its own on the way to the one it answers on,
#: and how much larger that grid is decides what a plan costs. On the doubled
#: grid the default factor of two is the largest allocation a build makes --
#: measured at 3962 MiB for a 384-cubed transfer -- where 1.25 is 1338 MiB for
#: the same tolerance, the same accuracy, and slightly less time: the smaller
#: working grid saves more than the wider interpolation kernel costs. Nothing
#: below 1.25 is available; CUFINUFFT refuses to plan at 1.125.
_UPSAMPLING = 1.25

#: The transfer is held as complex64, cut to the support the scan reached and
#: encoded in bfloat16 on the device, so gridding it tighter than this buys
#: nothing that survives any of those. Measured on a 384-cubed transfer, the
#: gridding leaves 3.95e-04 where compression leaves 1.15e-02, and going to
#: 1e-4 costs 2.5x the build for an error the compression swamps.
_TOLERANCE = 1e-3

#: One slot. A plan on the doubled grid is the largest allocation a build
#: makes -- larger than the kernel it produces -- and holding a second is what
#: makes a build run out of device memory.
_PLAN_SLOT: dict[tuple[Any, ...], Any] = {}


def _finufft(device: Any) -> Any:
    """Import the transform library for ``device``, or say which extra has it."""
    cuda = "cuda" in str(device)
    name = "cufinufft" if cuda else "finufft"
    extra = "cuda" if cuda else "nufft"
    try:
        return import_module(name)
    except ImportError as error:
        raise ImportError(
            f"gridding a transfer on {'a device' if cuda else 'the host'} "
            f"requires {name}: pip install mrtoeplitz[{extra}]"
        ) from error


def _yield_cached_device_memory(device: Any) -> None:
    """Hand the allocator's spare blocks back to the driver.

    A NUFFT plan is allocated outside Torch, so blocks Torch is holding for
    reuse are neither available to it nor counted as free -- and Torch does
    not release them when another library runs out. What a build measures and
    what it can take are both only true once these are returned.
    """
    torch = import_module("torch")
    if "cuda" not in str(device):
        return
    with suppress(RuntimeError):
        torch.cuda.empty_cache()


def _plan_options(tolerance: float | None) -> dict[str, Any]:
    """Return the tolerance and spreading grid to plan with."""
    return {
        "eps": _TOLERANCE if tolerance is None else float(tolerance),
        "upsampfac": _UPSAMPLING,
    }


class PsfPlan:
    """A type-1 NUFFT from a trajectory onto the doubled grid.

    ``isign=+1`` and centred mode ordering are what make the gridded point
    spread function the adjoint of the weights, which is the transform the
    transfer is built from.

    Parameters
    ----------
    spatial_shape
        The grid to grid onto: twice the image in every dimension.
    samples
        The trajectory, ``(samples, axes)``, in normalized k-space. Its device
        decides whether the transform runs on the host or on a device.
    """

    def __init__(
        self,
        spatial_shape: tuple[int, ...],
        samples: Any,
        tolerance: float | None = None,
    ) -> None:
        torch = import_module("torch")
        self.spatial_shape = tuple(int(size) for size in spatial_shape)
        self.device = samples.device
        self._cuda = "cuda" in str(self.device)
        library = _finufft(self.device)
        _yield_cached_device_memory(self.device)
        self._plan = library.Plan(
            1,
            self.spatial_shape,
            n_trans=1,
            isign=1,
            dtype="complex64",
            # Centred: the transfer's own centring is folded in downstream as a
            # sign on the locations it keeps, and that assumes k-space centre
            # sits in the middle of what comes back.
            modeord=0,
            **_plan_options(tolerance),
        )
        self._out = (
            torch.empty(self.spatial_shape, dtype=torch.complex64, device=self.device)
            if self._cuda
            else None
        )
        self.n_samples = 0
        self.setpts(samples)

    def setpts(self, samples: Any) -> None:
        """Retarget the plan at a trajectory of the same length.

        Planning is the expensive part of gridding, so one plan serves every
        trajectory of a build. Samples arrive in normalized k-space and the
        transform works in radians, which is the one conversion this package
        performs and the one place it belongs.
        """
        import math

        torch = import_module("torch")
        radians = (samples.to(torch.float32) * (2.0 * math.pi)).to(self.device)
        columns = [radians[:, axis].contiguous() for axis in range(radians.shape[1])]
        if not self._cuda:
            columns = [column.numpy() for column in columns]
        self._plan.setpts(*columns)
        self.n_samples = int(samples.shape[0])

    def grid(self, weights: Any) -> Any:
        """Return the point spread function of one weight per sample.

        The result is the raw transform: unnormalised, so what divides it is
        stated where the transfer is assembled rather than hidden here.
        """
        torch = import_module("torch")
        weights = weights.reshape(-1).to(torch.complex64).contiguous()
        if self._cuda:
            self._plan.execute(weights, out=self._out)
            return self._out
        gridded = self._plan.execute(weights.cpu().numpy())
        return torch.from_numpy(gridded).to(torch.complex64)


def psf_plan(
    spatial_shape: tuple[int, ...],
    samples: Any,
    tolerance: float | None = None,
) -> PsfPlan:
    """Return the build's gridding plan, retargeted at ``samples``."""
    shape = tuple(int(size) for size in spatial_shape)
    key = (
        str(samples.device).split(":")[0],
        shape,
        int(samples.shape[0]),
        tolerance,
    )
    held = _PLAN_SLOT.get(key)
    if held is not None:
        held.setpts(samples)
        return held
    _PLAN_SLOT.clear()
    plan = PsfPlan(shape, samples, tolerance)
    _PLAN_SLOT[key] = plan
    return plan


def within_psf_plans(build: Any) -> Any:
    """Release the gridding plan a builder makes when its build ends."""

    @wraps(build)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with psf_plans():
            return build(*args, **kwargs)

    return wrapper


@contextmanager
def psf_plans() -> Any:
    """Hold one gridding plan for the length of a build, then release it.

    A plan on the doubled grid is the largest allocation a build makes -- more
    than the kernel it produces -- and the solve that follows needs that
    memory for its own transforms.
    """
    try:
        yield
    finally:
        _PLAN_SLOT.clear()
        with suppress(ImportError, AttributeError):
            torch = import_module("torch")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
