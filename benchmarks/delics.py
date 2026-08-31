"""Fetching the Deli-CS dataset from Zenodo, once.

The benchmark runs on real acquisitions rather than on a phantom, and those
are too large to keep in a repository. This resolves them through Zenodo's
REST API at run time: the record listing carries each file's size and MD5, so
what to download and whether it arrived intact both come from the archive
rather than from constants written here.

Nothing is downloaded twice. A file that is present and has been verified once
is left alone, and an interrupted download resumes from where it stopped.

Deli-CS is Iyer, Schauman, Sandino et al., *Deep Learning Initialized
Compressed Sensing in Volumetric Spatio-Temporal Subspace Reconstruction* --
3D spiral-projection MRF on a GE Premier with a 48-channel head coil, released
under the BSD licence.

Set ``MRTOEPLITZ_DATA`` to choose where it lands; the default is
``~/.cache/mrtoeplitz/delics``.
"""

from __future__ import annotations

__all__ = ["RECORDS", "data_root", "fetch", "fetch_all"]

import hashlib
import json
import os
import shutil
import sys
import tarfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

#: Zenodo records, and what the benchmark takes from each. Keys are the names
#: this module answers to; values are ``(record, file)``.
RECORDS: dict[str, tuple[int, str]] = {
    # Trajectories, density weights, the subspace basis and the dictionary --
    # the metadata every part of the benchmark needs.
    "shared": (7734431, "shared.tar.gz"),
    # One raw acquisition, the smallest of the twelve.
    "raw": (7697373, "val_case000.tar.gz"),
    # BART's own reconstruction of the two-minute scan, as a correctness
    # anchor: non-density-compensated, beside a density-compensated SigPy one.
    "bart": (7734431, "bartcompare.tar.gz"),
}

_API = "https://zenodo.org/api/records/{record}"
_HOST = "zenodo.org"
_CHUNK = 1 << 20


def _opened(url: str, **headers: str) -> Any:
    """Open an https URL on the archive's own host, and nothing else.

    The download location is read out of a JSON payload the archive serves, so
    it is remote input: a revised record could name any scheme or host, and
    ``urlopen`` would follow a ``file:`` one straight into the filesystem.
    """
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != _HOST:
        raise RuntimeError(f"refusing to fetch {url!r}: not https on {_HOST}")
    request = urllib.request.Request(url)  # noqa: S310 -- scheme checked above
    for name, value in headers.items():
        request.add_header(name, value)
    return urllib.request.urlopen(request, timeout=120)  # noqa: S310


def data_root() -> Path:
    """Where downloads land, from ``MRTOEPLITZ_DATA`` or the default cache."""
    stated = os.environ.get("MRTOEPLITZ_DATA")
    if stated:
        return Path(stated).expanduser()
    return Path.home() / ".cache" / "mrtoeplitz" / "delics"


def _listing(record: int) -> dict[str, dict]:
    """Files in a Zenodo record, by name, with their size and checksum."""
    with _opened(_API.format(record=record), Accept="application/json") as response:
        payload = json.load(response)
    return {entry["key"]: entry for entry in payload.get("files", [])}


def _digest(path: Path) -> str:
    """MD5 of a file, read in chunks so a multi-gigabyte one fits."""
    md5 = hashlib.md5()  # noqa: S324 -- Zenodo publishes MD5, not our choice
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            md5.update(chunk)
    return md5.hexdigest()


def _report(name: str, done: int, total: int) -> None:
    if not sys.stderr.isatty() or not total:
        return
    share = done / total
    bar = "#" * int(40 * share)
    print(
        f"\r  {name:<28} [{bar:<40}] {100 * share:5.1f}%  "
        f"{done / 2**30:5.2f}/{total / 2**30:.2f} GiB",
        end="",
        file=sys.stderr,
    )


def _download(url: str, target: Path, size: int, name: str) -> None:
    """Fetch ``url`` into ``target``, resuming a partial file if there is one."""
    partial = target.with_suffix(target.suffix + ".part")
    have = partial.stat().st_size if partial.exists() else 0
    if have > size:  # a stale part from a different revision of the file
        partial.unlink()
        have = 0

    while have < size:
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with (
                _opened(url, **headers) as response,
                partial.open("ab" if have else "wb") as handle,
            ):
                while chunk := response.read(_CHUNK):
                    handle.write(chunk)
                    have += len(chunk)
                    _report(name, have, size)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            settled = partial.stat().st_size if partial.exists() else 0
            if settled <= have and settled == 0:
                raise
            # Resume from whatever landed rather than starting again.
            print(f"\n  {name}: {error}; resuming", file=sys.stderr)
            have = settled
    if sys.stderr.isatty():
        print(file=sys.stderr)
    partial.replace(target)


def fetch(name: str, *, extract: bool = True) -> Path:
    """Return the local path of one dataset, downloading it if it is absent.

    Parameters
    ----------
    name
        A key of :data:`RECORDS`.
    extract
        Unpack the archive and return the directory. ``False`` returns the
        archive itself.

    Returns
    -------
    pathlib.Path
        The unpacked directory, or the archive when ``extract`` is false.

    Raises
    ------
    KeyError
        If ``name`` is not a known dataset.
    RuntimeError
        If the record does not carry the file, or what arrived does not match
        the checksum the record publishes.
    """
    if name not in RECORDS:
        raise KeyError(f"unknown dataset {name!r}; known: {sorted(RECORDS)}")
    record, filename = RECORDS[name]

    root = data_root()
    root.mkdir(parents=True, exist_ok=True)
    archive = root / filename
    unpacked = root / filename.split(".")[0]
    verified = root / f"{filename}.verified"

    if extract and unpacked.is_dir():
        return unpacked
    if archive.exists() and verified.exists():
        return _unpack(archive, unpacked) if extract else archive

    entry = _listing(record).get(filename)
    if entry is None:
        raise RuntimeError(
            f"Zenodo record {record} does not carry {filename}; the record may "
            f"have been revised"
        )
    checksum = entry["checksum"].split(":")[-1]
    size = int(entry["size"])

    if not archive.exists():
        print(
            f"fetching {filename} ({size / 2**30:.2f} GiB) from Zenodo record {record}",
            file=sys.stderr,
        )
        _download(entry["links"]["self"], archive, size, filename)

    if _digest(archive) != checksum:
        archive.unlink()
        raise RuntimeError(
            f"{filename} does not match the MD5 the record publishes; the "
            f"partial file has been removed, so running again re-fetches it"
        )
    verified.touch()
    return _unpack(archive, unpacked) if extract else archive


def _unpack(archive: Path, target: Path) -> Path:
    """Extract an archive beside itself, once."""
    if target.is_dir():
        return target
    staging = target.with_name(target.name + ".unpacking")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    print(f"unpacking {archive.name}", file=sys.stderr)
    with tarfile.open(archive) as tar:
        # filter="data" refuses absolute paths, parent traversal and device
        # nodes; an archive is remote content whatever its licence says.
        tar.extractall(staging, filter="data")
    staging.replace(target)
    return target


def fetch_all(names: tuple[str, ...] = tuple(RECORDS)) -> dict[str, Path]:
    """Fetch several datasets, returning where each one landed."""
    return {name: fetch(name) for name in names}


if __name__ == "__main__":
    wanted = tuple(sys.argv[1:]) or tuple(RECORDS)
    for name, path in fetch_all(wanted).items():
        print(f"{name:>8}: {path}")
