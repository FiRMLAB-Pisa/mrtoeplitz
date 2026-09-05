"""Every docstring example runs, against the installed package.

The examples are the documentation, so one that has drifted is a broken one.
They are collected from the package as imported rather than from the source
tree, so this says the same thing about an editable checkout and about an
installed wheel.
"""

import doctest
import importlib
import pkgutil

import pytest

import mrtoeplitz as package

#: Accelerators the package works without. A module that exists only to hold
#: one is skipped when it is absent, rather than failing collection for
#: everything else.
OPTIONAL = {"triton"}


def _modules():
    yield package.__name__
    for info in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        yield info.name


@pytest.mark.parametrize("name", list(_modules()))
def test_every_docstring_example_runs(name, capsys):
    try:
        module = importlib.import_module(name)
    except ModuleNotFoundError as missing:
        if missing.name in OPTIONAL:
            pytest.skip(f"{name} needs {missing.name}, which is not installed")
        raise
    results = doctest.testmod(
        module, verbose=False, report=False, optionflags=doctest.ELLIPSIS
    )
    captured = capsys.readouterr()
    assert results.failed == 0, captured.out
