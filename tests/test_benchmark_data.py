"""The Zenodo fetcher behind the benchmark.

Nothing here touches the network. What is worth pinning is the behaviour a
user meets before any download starts, and the two things that must not go
wrong once one has: a corrupt archive must not be left behind claiming to be
valid, and a second run must not fetch again.
"""

import sys
import tarfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))

import delics


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("MRTOEPLITZ_DATA", str(tmp_path))
    return tmp_path


def _archive(path, name="phi.mat", payload=b"basis"):
    """A one-file tarball, standing in for a Zenodo download."""
    member = path.parent / name
    member.write_bytes(payload)
    with tarfile.open(path, "w:gz") as tar:
        tar.add(member, arcname=name)
    member.unlink()
    return path


def test_the_download_location_follows_the_environment(root):
    assert delics.data_root() == root


def test_an_unknown_dataset_says_which_ones_there_are():
    with pytest.raises(KeyError, match="shared"):
        delics.fetch("nonexistent")


def test_an_archive_already_verified_is_not_fetched_again(root, monkeypatch):
    _archive(root / "shared.tar.gz")
    (root / "shared.tar.gz.verified").touch()

    def refuse(record):
        raise AssertionError("the record was queried for a file already held")

    monkeypatch.setattr(delics, "_listing", refuse)
    assert delics.fetch("shared") == root / "shared"


def test_an_unpacked_dataset_is_not_unpacked_again(root, monkeypatch):
    (root / "shared").mkdir()
    monkeypatch.setattr(
        delics, "_listing", lambda record: pytest.fail("queried needlessly")
    )
    assert delics.fetch("shared") == root / "shared"


def test_an_archive_that_fails_its_checksum_is_removed(root, monkeypatch):
    """Otherwise the next run finds it, trusts it, and reads garbage."""
    archive = _archive(root / "shared.tar.gz")
    monkeypatch.setattr(
        delics,
        "_listing",
        lambda record: {
            "shared.tar.gz": {
                "checksum": "md5:" + "0" * 32,
                "size": archive.stat().st_size,
                "links": {"self": "https://example.invalid/shared.tar.gz"},
            }
        },
    )
    with pytest.raises(RuntimeError, match="does not match the MD5"):
        delics.fetch("shared")
    assert not archive.exists()
    assert not (root / "shared.tar.gz.verified").exists()


def test_a_file_the_record_no_longer_carries_is_reported(root, monkeypatch):
    monkeypatch.setattr(delics, "_listing", lambda record: {})
    with pytest.raises(RuntimeError, match="does not carry"):
        delics.fetch("shared")


def test_unpacking_lands_the_members_where_they_are_expected(root):
    archive = _archive(root / "shared.tar.gz", payload=b"basis")
    unpacked = delics._unpack(archive, root / "shared")
    assert (unpacked / "phi.mat").read_bytes() == b"basis"
    # The staging directory is renamed into place, so a half-written unpack is
    # never mistaken for a finished one.
    assert not (root / "shared.unpacking").exists()


def test_a_download_location_that_is_not_zenodo_is_refused():
    """The URL comes out of a JSON payload, so it is remote input."""
    for url in ("file:///etc/passwd", "http://zenodo.org/x", "https://elsewhere/x"):
        with pytest.raises(RuntimeError, match="refusing to fetch"):
            delics._opened(url)


def test_every_named_dataset_points_at_a_record_and_a_file():
    for name, (record, filename) in delics.RECORDS.items():
        assert isinstance(record, int)
        assert filename.endswith(".tar.gz"), name
