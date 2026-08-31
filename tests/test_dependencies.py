"""What this package is allowed to import.

mrtoeplitz sits low in the family and depends on Torch alone; MRI-NUFFT is
needed only to *build* a transfer and is an extra. Reaching for a sibling
would not merely add a dependency -- the nearest one, mrutils, offers the
centred orthonormal Fourier convention, and every transform here is
deliberately a raw, uncentred one taken ``norm="forward"`` into a reused
buffer. That form is what lets the resident lane fit, and routing through the
centred helpers would quietly undo it.
"""

import ast
import pathlib
import re

import pytest

import mrtoeplitz

FORBIDDEN = {
    "deepinv",
    "deepmr",
    "mrdistortion",
    "mrllr",
    "mrmotion",
    "mrutils",
    "torchsolve",
}

#: Distribution names whose import name differs.
IMPORT_NAMES: dict[str, str] = {}

#: Imported without being declared, because Torch brings them itself and
#: pinning a version here would fight it.
BUNDLED_WITH_TORCH = {"triton"}


def _declared_modules():
    """The import names this package's own metadata promises will be there."""
    tomllib = pytest.importorskip("tomllib", reason="needs Python 3.11+")
    pyproject = pathlib.Path(__file__).resolve().parent.parent / "pyproject.toml"
    if not pyproject.exists():
        pytest.skip("not a source checkout")
    with pyproject.open("rb") as handle:
        project = tomllib.load(handle)["project"]

    names = set()
    groups = [project.get("dependencies", [])]
    groups.extend(project.get("optional-dependencies", {}).values())
    for group in groups:
        for requirement in group:
            distribution = re.split(r"[<>=!~;\[ ]", requirement, maxsplit=1)[0].strip()
            if not distribution or distribution == "mrtoeplitz":
                continue
            names.add(IMPORT_NAMES.get(distribution, distribution.replace("-", "_")))
    return names | BUNDLED_WITH_TORCH


def _sources():
    root = pathlib.Path(mrtoeplitz.__file__).parent
    return sorted(root.glob("*.py"))


def _imported_roots(path):
    tree = ast.parse(path.read_text())
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call):
            # import_module("torch") and friends are imports too.
            function = node.func
            name = getattr(function, "id", None) or getattr(function, "attr", None)
            if name == "import_module" and node.args:
                first = node.args[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    roots.add(first.value.split(".")[0])
    return roots


def test_no_sibling_package_is_imported():
    offenders = {}
    for path in _sources():
        found = _imported_roots(path) & FORBIDDEN
        if found:
            offenders[path.name] = sorted(found)
    assert not offenders, f"sibling packages imported: {offenders}"


def test_every_imported_module_is_declared_in_the_metadata():
    """An undeclared import is a package that works on the author's machine."""
    import sys

    allowed = _declared_modules()
    stdlib = set(sys.stdlib_module_names)
    unexpected = {}
    for path in _sources():
        roots = _imported_roots(path)
        strange = {
            root
            for root in roots
            if root not in stdlib
            and root not in allowed
            and root != "mrtoeplitz"
            and not root.startswith("_")
            and root != ""
        }
        if strange:
            unexpected[path.name] = sorted(strange)
    assert not unexpected, f"undeclared imports: {unexpected}"


def test_a_cuda_transfer_is_built_without_importing_cupy():
    """The CUDA path uses Torch tensors, not a second CUDA array library.

    CUFINUFFT reads anything exposing ``__cuda_array_interface__``, so a Torch
    tensor goes straight in. Its own requirements are NumPy and packaging;
    CuPy is a choice some wrappers make and this package does not.
    """
    import sys

    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    try:
        import cufinufft  # noqa: F401
    except ImportError as error:
        # The wheel installs anywhere; its library only loads where there is a
        # CUDA runtime, and that failure is an ImportError from deep inside it
        # rather than a missing module.
        pytest.skip(f"cufinufft unavailable: {error}")

    import numpy as np

    import mrtoeplitz as mt

    angles = np.linspace(0, np.pi, 16, endpoint=False)
    radius = np.linspace(-0.5, 0.5, 32, endpoint=False)
    samples = np.stack(
        [np.outer(np.cos(angles), radius), np.outer(np.sin(angles), radius)], -1
    ).astype(np.float32)

    sys.modules.pop("cupy", None)
    kernel = mt.scalar_kernel(
        torch.as_tensor(samples, device="cuda"),
        (32, 32),
        options=mt.toeplitz_options(compress=False, cuda_transfer_precision="float32"),
    )
    image = torch.randn(1, 1, 32, 32, dtype=torch.complex64, device="cuda")
    kernel.to("cuda")(image)

    assert "cupy" not in sys.modules
