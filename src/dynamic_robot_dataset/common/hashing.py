"""Deterministic hashing helpers used for identity, provenance, and QC.

Hashes in metadata are SHA-256 unless an algorithm is explicitly recorded.  JSON
hashes use a canonical UTF-8 representation so they are stable across processes
and Python versions.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Mapping

DEFAULT_CHUNK_SIZE = 8 * 1024 * 1024


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, set):
        return sorted(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("NaN and infinity are not permitted in canonical JSON")
    raise TypeError(f"Cannot serialize {type(value).__name__} to canonical JSON")


def canonical_json_bytes(value: Any) -> bytes:
    """Return a canonical, finite JSON encoding for *value*."""

    return json.dumps(
        value,
        default=_json_default,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(data: bytes | bytearray | memoryview) -> str:
    """Hash an in-memory byte sequence."""

    return hashlib.sha256(data).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash a value after canonical JSON serialization."""

    return sha256_bytes(canonical_json_bytes(value))


def sha256_stream(stream: BinaryIO, chunk_size: int = DEFAULT_CHUNK_SIZE) -> str:
    """Hash a binary stream from its current position to EOF."""

    digest = hashlib.sha256()
    while chunk := stream.read(chunk_size):
        digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: str | Path, chunk_size: int = DEFAULT_CHUNK_SIZE) -> str:
    """Hash a regular file without loading it into memory."""

    with Path(path).open("rb") as handle:
        return sha256_stream(handle, chunk_size)


def stable_uint64(value: Any, namespace: str = "") -> int:
    """Map a value to a stable unsigned 64-bit integer."""

    digest = hashlib.sha256(namespace.encode("utf-8") + b"\0" + canonical_json_bytes(value))
    return int.from_bytes(digest.digest()[:8], "big", signed=False)


def hash_manifest(paths: Iterable[str | Path], root: str | Path) -> dict[str, str]:
    """Return relative-path to content-hash mappings for regular files.

    Every input must resolve below *root*.  The result is sorted to make the
    mapping suitable for provenance records and reproducibility checks.
    """

    base = Path(root).resolve(strict=True)
    result: dict[str, str] = {}
    for raw_path in paths:
        path = Path(raw_path).resolve(strict=True)
        try:
            relative = path.relative_to(base)
        except ValueError as exc:
            raise ValueError(f"Path is outside manifest root: {path}") from exc
        if path.is_file():
            result[relative.as_posix()] = sha256_file(path)
    return dict(sorted(result.items()))


def combined_manifest_hash(manifest: Mapping[str, str]) -> str:
    """Compute one identity hash for a relative-path content manifest."""

    return sha256_json(dict(sorted(manifest.items())))


def hamming_distance_hex(left: str, right: str) -> int:
    """Return bitwise Hamming distance between equal-length hexadecimal strings."""

    if len(left) != len(right):
        raise ValueError("Hex digests must have equal lengths")
    try:
        return (int(left, 16) ^ int(right, 16)).bit_count()
    except ValueError as exc:
        raise ValueError("Inputs must be hexadecimal") from exc

