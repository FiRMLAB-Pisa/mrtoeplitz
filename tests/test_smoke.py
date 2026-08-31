"""The package imports and reports a version."""

import mrtoeplitz


def test_the_package_reports_a_version():
    assert isinstance(mrtoeplitz.__version__, str)
    assert mrtoeplitz.__version__
