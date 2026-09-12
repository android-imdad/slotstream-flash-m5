#!/usr/bin/env python3
"""Shared, dependency-free receipt helpers."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
FLASH_ROOT = ROOT / ".build" / "flash"
RECEIPT_FORMAT = "slotstream-flash-receipt-v1"
SELECTION_FORMAT = "slotstream-flash-selection-v1"


class EvidenceError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise EvidenceError(f"invalid JSON at {path}: {error}") from error
    if not isinstance(value, dict):
        raise EvidenceError(f"JSON object required at {path}")
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def fresh_output(path: Path) -> Path:
    FLASH_ROOT.mkdir(parents=True, exist_ok=True)
    root = FLASH_ROOT.resolve(strict=True)
    requested = Path(os.path.abspath(path))
    lexical_root = Path(os.path.abspath(FLASH_ROOT))
    if not requested.is_relative_to(lexical_root) or requested == lexical_root:
        raise EvidenceError(f"output must be under {root}")
    current = root
    relative = requested.relative_to(lexical_root)
    for component in relative.parts[:-1]:
        current = current / component
        if current.is_symlink():
            raise EvidenceError(f"symlinked output ancestor is not accepted: {current}")
        if current.exists():
            if not current.is_dir() or not current.resolve().is_relative_to(root):
                raise EvidenceError(f"unsafe output ancestor: {current}")
        else:
            current.mkdir(mode=0o700)
    result = current / relative.name
    if result.exists() or result.is_symlink():
        raise EvidenceError(f"output already exists: {result}")
    result.mkdir(mode=0o700)
    return result.resolve(strict=True)


def regular_file(path: Path, *, within: Path | None = None) -> Path:
    if path.is_symlink():
        raise EvidenceError(f"symlink is not accepted: {path}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise EvidenceError(f"missing artifact: {path}") from error
    if within is not None and not resolved.is_relative_to(within.resolve()):
        raise EvidenceError(f"artifact escapes {within}: {path}")
    if not resolved.is_file():
        raise EvidenceError(f"regular file required: {path}")
    return resolved


def copy_verified(source: Path, destination: Path, *, within: Path | None = None) -> dict[str, Any]:
    source = regular_file(source, within=within)
    before = sha256(source)
    shutil.copy2(source, destination, follow_symlinks=False)
    after = sha256(destination)
    if before != after:
        destination.unlink(missing_ok=True)
        raise EvidenceError(f"artifact changed while copying: {source}")
    return {"path": str(destination), "bytes": destination.stat().st_size, "sha256": after}


def build_source_paths(root: Path) -> list[Path]:
    candidates = [*root.joinpath("Sources").rglob("*"), root / "Package.swift",
                  root / "Package.resolved", root / "Makefile", root / "Tools/build_identity.py",
                  root / "Tools/fetch_metallib.sh"]
    if any(path.is_symlink() for path in candidates):
        raise EvidenceError("build source symlinks require an explicit archived dependency")
    return sorted(path for path in candidates if not path.is_dir())


def current_source_hashes(identity: dict[str, Any], *, root: Path = ROOT) -> dict[str, str]:
    source = identity.get("source")
    if not isinstance(source, dict) or not source:
        raise EvidenceError("build identity has no source map")
    root = root.resolve()
    candidates = build_source_paths(root)
    live = {str(path.relative_to(root)) for path in candidates}
    if live != set(source):
        missing = sorted(set(source) - live)
        added = sorted(live - set(source))
        raise EvidenceError(f"build source inventory changed; missing={missing}, added={added}")
    actual: dict[str, str] = {}
    for relative, expected in source.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise EvidenceError("malformed build source map")
        path = regular_file(root / relative, within=root)
        actual[relative] = sha256(path)
    return actual


def validate_source_map(identity: dict[str, Any], *, root: Path = ROOT) -> None:
    if current_source_hashes(identity, root=root) != identity["source"]:
        raise EvidenceError("current source tree does not match build identity")


def harness_hashes() -> dict[str, str]:
    """Bind receipts to every versioned file that interprets their evidence."""
    base = ROOT / "Tools" / "flash"
    files = sorted(path for path in base.rglob("*") if path.is_file() and "__pycache__" not in path.parts)
    return {str(path.relative_to(ROOT)): sha256(path) for path in files}


def _validate_source_archive(path: Path, source: dict[str, str]) -> None:
    try:
        with tarfile.open(path, "r:gz") as archive:
            members = archive.getmembers()
            names = [member.name for member in members]
            if len(names) != len(set(names)) or set(names) != set(source):
                raise EvidenceError("source archive inventory does not match build identity")
            for member in members:
                member_path = Path(member.name)
                if member_path.is_absolute() or ".." in member_path.parts or not member.isfile():
                    raise EvidenceError(f"unsafe source archive member: {member.name}")
                stream = archive.extractfile(member)
                if stream is None:
                    raise EvidenceError(f"missing source archive payload: {member.name}")
                digest = hashlib.sha256()
                for part in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(part)
                if digest.hexdigest() != source[member.name]:
                    raise EvidenceError(f"source archive content mismatch: {member.name}")
    except (OSError, tarfile.TarError) as error:
        raise EvidenceError(f"invalid source archive: {error}") from error


def validate_build_identity(binary: Path, *, root: Path = ROOT) -> tuple[dict[str, Any], dict[str, Path]]:
    binary = regular_file(binary)
    directory = binary.parent
    paths = {
        "binary": binary,
        "metallib": regular_file(directory / "mlx.metallib", within=directory),
        "source_archive": regular_file(directory / "build-source.tar.gz", within=directory),
        "identity": regular_file(directory / "build-identity.json", within=directory),
        "source_before": regular_file(directory / "build-source-before.json", within=directory),
    }
    identity = read_json(paths["identity"])
    required = {"source", "source_archive_sha256", "binary_sha256", "metallib_sha256"}
    if set(identity) != required:
        raise EvidenceError(f"unexpected build identity fields: {sorted(identity)}")
    expected = {
        "binary_sha256": sha256(paths["binary"]),
        "metallib_sha256": sha256(paths["metallib"]),
        "source_archive_sha256": sha256(paths["source_archive"]),
    }
    for key, actual in expected.items():
        if identity.get(key) != actual:
            raise EvidenceError(f"{key} does not match build identity")
    validate_source_map(identity, root=root)
    if read_json(paths["source_before"]) != identity["source"]:
        raise EvidenceError("pre-build source receipt does not match build identity")
    _validate_source_archive(paths["source_archive"], identity["source"])
    return identity, paths
