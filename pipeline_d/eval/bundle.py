"""Safe, engine-independent utilities for append-only evaluation stages."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path, PurePosixPath, PureWindowsPath

STAGE_SCHEMA_ID = "urn:mapgen:pipeline-c:eval:stage-manifest:v1"
MANIFEST_NAME = "manifest.json"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
STAGES = {"bundle", "submission", "result"}


class BundleError(ValueError):
    """An evaluation stage is unsafe, incomplete, or has changed."""


def _closed_json_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise BundleError(f"JSON object contains duplicate key {key!r}")
        result[key] = value
    return result


def _invalid_json_constant(value: str):
    raise BundleError(f"JSON contains non-standard numeric constant {value}")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative_path(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise BundleError("artifact path must be a non-empty string")
    if "\\" in value or "\x00" in value:
        raise BundleError(f"artifact path is not canonical POSIX: {value!r}")
    path = PurePosixPath(value)
    if (value in (".", "..") or path.is_absolute()
            or PureWindowsPath(value).drive or value != path.as_posix()):
        raise BundleError(f"artifact path is not relative/canonical: {value!r}")
    if any(part in ("", ".", "..") for part in path.parts):
        raise BundleError(f"artifact path escapes or aliases its root: {value!r}")
    return value


def _root_name(value: object) -> str:
    value = safe_relative_path(value)
    if "/" in value:
        raise BundleError("visible/hidden roots must be one directory name")
    return value


def _exact_keys(value: object, expected: set[str], label: str) -> dict:
    if not isinstance(value, dict):
        raise BundleError(f"{label} must be an object")
    extras = set(value) - expected
    missing = expected - set(value)
    if extras or missing:
        raise BundleError(
            f"{label} keys mismatch; missing={sorted(missing)}, "
            f"extra={sorted(extras)}")
    return value


def _iso_datetime(value: object) -> str:
    if not isinstance(value, str):
        raise BundleError("created_at_utc must be a string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise BundleError("created_at_utc is not ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise BundleError("created_at_utc must include a UTC offset")
    if parsed.utcoffset().total_seconds() != 0:
        raise BundleError("created_at_utc must be UTC")
    return value


def inventory_tree(root: str | Path) -> list[dict[str, object]]:
    root = Path(root).resolve()
    if not root.is_dir():
        raise BundleError(f"stage root is not a directory: {root}")
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise BundleError(f"symlinks are forbidden in a stage: {path}")
        if not path.is_file() or path.name == MANIFEST_NAME and path.parent == root:
            continue
        relative = safe_relative_path(path.relative_to(root).as_posix())
        files.append({
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return files


def make_stage_manifest(
    root: str | Path,
    *,
    stage: str,
    stage_id: str,
    created_at_utc: str,
    parent_manifest_sha256s: list[str],
    prompt_ids: list[str],
    schema_ids: list[str],
    judge_visible_roots: list[str],
    hidden_roots: list[str],
) -> dict[str, object]:
    manifest = {
        "schema_id": STAGE_SCHEMA_ID,
        "schema_version": 1,
        "stage": stage,
        "stage_id": stage_id,
        "created_at_utc": created_at_utc,
        "parent_manifest_sha256s": parent_manifest_sha256s,
        "prompt_ids": prompt_ids,
        "schema_ids": schema_ids,
        "judge_visible_roots": judge_visible_roots,
        "hidden_roots": hidden_roots,
        "files": inventory_tree(root),
    }
    validate_manifest_shape(manifest)
    return manifest


def validate_manifest_shape(manifest: object) -> dict:
    expected = {
        "schema_id", "schema_version", "stage", "stage_id",
        "created_at_utc", "parent_manifest_sha256s", "prompt_ids",
        "schema_ids", "judge_visible_roots", "hidden_roots", "files",
    }
    manifest = _exact_keys(manifest, expected, "manifest")
    if manifest["schema_id"] != STAGE_SCHEMA_ID:
        raise BundleError("unexpected stage-manifest schema_id")
    if manifest["schema_version"] != 1:
        raise BundleError("unexpected stage-manifest schema_version")
    if manifest["stage"] not in STAGES:
        raise BundleError("invalid stage")
    if not isinstance(manifest["stage_id"], str) or not SAFE_ID.fullmatch(
        manifest["stage_id"]
    ):
        raise BundleError("invalid stage_id")
    _iso_datetime(manifest["created_at_utc"])
    parent_hashes = manifest["parent_manifest_sha256s"]
    if not isinstance(parent_hashes, list):
        raise BundleError("parent_manifest_sha256s must be an array")
    if any(not isinstance(value, str) or not HEX64.fullmatch(value)
           for value in parent_hashes):
        raise BundleError("parent_manifest_sha256s contains an invalid hash")
    if len(set(parent_hashes)) != len(parent_hashes):
        raise BundleError("parent_manifest_sha256s must be unique")
    if parent_hashes != sorted(parent_hashes):
        raise BundleError("parent_manifest_sha256s must be sorted canonically")
    if manifest["stage"] == "bundle" and parent_hashes:
        raise BundleError("a bundle stage must have no parent manifests")
    if manifest["stage"] == "submission" and len(parent_hashes) != 1:
        raise BundleError("a submission stage needs exactly one parent manifest")
    if manifest["stage"] == "result" and not parent_hashes:
        raise BundleError("a result stage needs one or more parent manifests")

    for field in ("prompt_ids", "schema_ids"):
        values = manifest[field]
        if (not isinstance(values, list)
                or any(not isinstance(value, str) or not value for value in values)
                or len(set(values)) != len(values)):
            raise BundleError(f"{field} must contain unique non-empty strings")

    visible = manifest["judge_visible_roots"]
    hidden = manifest["hidden_roots"]
    if not isinstance(visible, list) or not isinstance(hidden, list):
        raise BundleError("visible/hidden roots must be arrays")
    visible = [_root_name(value) for value in visible]
    hidden = [_root_name(value) for value in hidden]
    if len(set(visible + hidden)) != len(visible) + len(hidden):
        raise BundleError("visible and hidden roots must be unique and disjoint")
    declared_roots = set(visible + hidden)
    if not declared_roots:
        raise BundleError("a stage must declare at least one content root")

    files = manifest["files"]
    if not isinstance(files, list) or not files:
        raise BundleError("a stage must contain at least one manifested file")
    paths = []
    for index, item in enumerate(files):
        item = _exact_keys(item, {"path", "bytes", "sha256"}, f"files[{index}]")
        path = safe_relative_path(item["path"])
        if path.split("/", 1)[0] not in declared_roots:
            raise BundleError(f"file is outside declared content roots: {path}")
        if isinstance(item["bytes"], bool) or not isinstance(item["bytes"], int):
            raise BundleError(f"file byte count is not an integer: {path}")
        if item["bytes"] < 0:
            raise BundleError(f"file byte count is negative: {path}")
        if not isinstance(item["sha256"], str) or not HEX64.fullmatch(
            item["sha256"]
        ):
            raise BundleError(f"invalid file hash: {path}")
        paths.append(path)
    if len(set(paths)) != len(paths):
        raise BundleError("manifest contains duplicate file paths")
    return manifest


def write_stage_manifest(root: str | Path, manifest: dict) -> Path:
    root = Path(root)
    validate_manifest_shape(manifest)
    target = root / MANIFEST_NAME
    with target.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return target


def load_and_verify_stage(root: str | Path) -> dict:
    root = Path(root).resolve()
    try:
        manifest = json.loads(
            (root / MANIFEST_NAME).read_text(encoding="utf-8"),
            object_pairs_hook=_closed_json_object,
            parse_constant=_invalid_json_constant,
        )
    except BundleError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BundleError(f"stage manifest is unreadable: {exc}") from exc
    validate_manifest_shape(manifest)
    expected = manifest["files"]
    actual = inventory_tree(root)
    if actual != expected:
        expected_by_path = {item["path"]: item for item in expected}
        actual_by_path = {item["path"]: item for item in actual}
        missing = sorted(set(expected_by_path) - set(actual_by_path))
        added = sorted(set(actual_by_path) - set(expected_by_path))
        changed = sorted(
            path for path in set(expected_by_path) & set(actual_by_path)
            if expected_by_path[path] != actual_by_path[path]
        )
        raise BundleError(
            "stage content does not match its closed manifest; "
            f"missing={missing}, added={added}, changed={changed}")
    return manifest


def create_staging_directory(destination: str | Path) -> Path:
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"published stage already exists: {destination}")
    return Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.build-", dir=destination.parent))


def publish_staged_directory(staging: str | Path, destination: str | Path) -> Path:
    """Publish a verified sibling staging directory without overwriting.

    The exclusive lock serializes cooperating builders. The destination is
    checked again under that lock, then the directory is renamed. Failed
    staging content is deliberately retained for diagnosis.
    """
    staging = Path(staging).resolve()
    destination = Path(destination).resolve()
    if staging.parent != destination.parent:
        raise BundleError("staging and destination must be siblings")
    load_and_verify_stage(staging)
    lock = destination.parent / f".{destination.name}.publish.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise BundleError(f"another publisher holds {lock.name}") from exc
    try:
        os.close(descriptor)
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite stage: {destination}")
        staging.rename(destination)
    finally:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass
    return destination
