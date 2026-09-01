"""Generic, engine-independent artifact closure and publication primitives.

The review laboratory and the evaluation harness have different manifest
semantics.  This module contains only the byte, path, inventory, and
no-overwrite publication operations that are safe for both boundaries.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import unicodedata
from collections.abc import Callable, Iterable
from pathlib import Path, PurePosixPath, PureWindowsPath


HEX64 = re.compile(r"^[0-9a-f]{64}$")
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class ArtifactError(ValueError):
    """Artifact bytes, paths, manifests, or publication are unsafe."""


def _closed_json_object(pairs: list[tuple[str, object]]) -> dict:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ArtifactError(f"JSON object contains duplicate key {key!r}")
        result[key] = value
    return result


def _invalid_json_constant(value: str):
    raise ArtifactError(f"JSON contains non-standard numeric constant {value}")


def canonical_json_bytes(value: object) -> bytes:
    """Serialize JSON data to the laboratory's canonical byte representation.

    The representation is deliberately small and explicit rather than a claim
    of implementing an external canonical-JSON standard.  Hashes always name
    these exact bytes.
    """

    try:
        text = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ArtifactError(f"value is not canonical JSON data: {exc}") from exc
    return text.encode("utf-8")


def read_strict_json(path: str | Path) -> object:
    """Read UTF-8 JSON while rejecting duplicate keys and NaN/Infinity."""

    try:
        return json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_closed_json_object,
            parse_constant=_invalid_json_constant,
        )
    except ArtifactError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"JSON file is unreadable: {path}: {exc}") from exc


def sha256_bytes(value: bytes | bytearray | memoryview) -> str:
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise TypeError("value must be bytes-like")
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ArtifactError(f"artifact is unreadable: {path}: {exc}") from exc
    return digest.hexdigest()


def safe_relative_path(value: object) -> str:
    """Return a canonical POSIX relative path or fail closed."""

    if not isinstance(value, str) or not value:
        raise ArtifactError("artifact path must be a non-empty string")
    if "\\" in value or "\x00" in value:
        raise ArtifactError(f"artifact path is not canonical POSIX: {value!r}")
    if unicodedata.normalize("NFC", value) != value:
        raise ArtifactError(f"artifact path is not NFC-normalized: {value!r}")
    path = PurePosixPath(value)
    if (
        value in (".", "..")
        or path.is_absolute()
        or PureWindowsPath(value).drive
        or value != path.as_posix()
        or any(part in ("", ".", "..") for part in path.parts)
    ):
        raise ArtifactError(f"artifact path is not relative/canonical: {value!r}")
    return value


def inventory_tree(
    root: str | Path, *, excluded_root_names: Iterable[str] = ()
) -> list[dict[str, object]]:
    """Hash every regular file below *root* and reject ambiguous trees."""

    root = Path(root).resolve()
    if not root.is_dir():
        raise ArtifactError(f"artifact root is not a directory: {root}")
    excluded = set(excluded_root_names)
    for name in excluded:
        safe_relative_path(name)
        if "/" in name:
            raise ArtifactError("excluded root names must be direct child names")

    files: list[dict[str, object]] = []
    casefolded: dict[str, str] = {}
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise ArtifactError(f"symlinks are forbidden in artifacts: {path}")
        relative = safe_relative_path(path.relative_to(root).as_posix())
        if relative.split("/", 1)[0] in excluded:
            continue
        if path.is_dir():
            continue
        if not path.is_file():
            raise ArtifactError(f"special filesystem entries are forbidden: {path}")
        folded = relative.casefold()
        previous = casefolded.get(folded)
        if previous is not None and previous != relative:
            raise ArtifactError(
                "artifact paths collide under case folding: "
                f"{previous!r}, {relative!r}"
            )
        casefolded[folded] = relative
        files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return files


def write_canonical_json_exclusive(path: str | Path, value: object) -> Path:
    """Create one canonical JSON file without replacing an existing record."""

    path = Path(path)
    data = canonical_json_bytes(value)
    try:
        with path.open("xb") as handle:
            handle.write(data)
    except FileExistsError:
        raise
    except OSError as exc:
        raise ArtifactError(f"cannot write canonical JSON {path}: {exc}") from exc
    return path


def create_staging_directory(destination: str | Path, *, label: str = "build") -> Path:
    """Create a sibling staging directory while refusing an existing target."""

    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"published artifact already exists: {destination}")
    if not isinstance(label, str) or not SAFE_ID.fullmatch(label):
        raise ArtifactError("staging label is invalid")
    return Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.{label}-", dir=destination.parent
        )
    )


def publish_verified_directory(
    staging: str | Path,
    destination: str | Path,
    verifier: Callable[[Path], object],
) -> Path:
    """Verify and atomically publish a sibling directory without overwrite.

    Failed staging directories remain in place for diagnosis.  The exclusive
    lock coordinates cooperating local publishers; an abandoned lock is
    reported rather than guessed stale or silently removed.
    """

    staging = Path(staging).resolve()
    destination = Path(destination).resolve()
    if staging.parent != destination.parent:
        raise ArtifactError("staging and destination must be siblings")
    if not staging.is_dir() or staging.is_symlink():
        raise ArtifactError("staging must be a real directory")
    if not callable(verifier):
        raise TypeError("verifier must be callable")
    verifier(staging)

    lock = destination.parent / f".{destination.name}.publish.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise ArtifactError(
            f"publication lock already exists and requires diagnosis: {lock}"
        ) from exc
    try:
        os.close(descriptor)
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite artifact: {destination}")
        staging.rename(destination)
    finally:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass
    return destination
