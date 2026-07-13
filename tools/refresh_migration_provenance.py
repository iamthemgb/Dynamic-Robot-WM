#!/usr/bin/env python3
"""Refresh stat evidence in the immutable-source migration ledgers.

The tool reads, hashes, and stats the original and copied files. It never writes
under a source root or ``legacy_sources``. Ledger replacement happens only
after every configured digest and byte-identical copy has been revalidated.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import grp
import hashlib
import json
import os
from pathlib import Path
import pwd
import tempfile
from typing import Any, Mapping

import yaml


TIMESTAMP_BASIS = (
    "copied_file_mtime_utc: destination mtime observed after the historical "
    "byte copy; no separate transaction timestamp was logged"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_mtime(path: Path) -> str:
    value = datetime.fromtimestamp(path.stat().st_mtime_ns / 1_000_000_000, timezone.utc)
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _user_name(uid: int) -> str:
    try:
        return pwd.getpwuid(uid).pw_name
    except KeyError:
        return str(uid)


def _group_name(gid: int) -> str:
    try:
        return grp.getgrgid(gid).gr_name
    except KeyError:
        return str(gid)


def _stat_evidence(source: Path, copied: Path) -> dict[str, Any]:
    source_stat = source.stat()
    copied_stat = copied.stat()
    copied_mtime = _utc_mtime(copied)
    return {
        "source_owner": _user_name(source_stat.st_uid),
        "source_owner_uid": source_stat.st_uid,
        "source_group": _group_name(source_stat.st_gid),
        "source_group_gid": source_stat.st_gid,
        "source_mtime_utc": _utc_mtime(source),
        "copied_file_owner": _user_name(copied_stat.st_uid),
        "copied_file_owner_uid": copied_stat.st_uid,
        "copied_file_group": _group_name(copied_stat.st_gid),
        "copied_file_group_gid": copied_stat.st_gid,
        "copied_file_mtime_utc": copied_mtime,
        "copy_timestamp_utc": copied_mtime,
        "copy_timestamp_basis": TIMESTAMP_BASIS,
    }


def _replace_text(path: Path, text: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def refresh(repository_root: Path) -> tuple[int, int]:
    migration = repository_root / "migration"
    mapping_path = migration / "source_mapping.yaml"
    integrity_path = migration / "source_before_after_hashes.json"
    mapping = yaml.safe_load(mapping_path.read_text(encoding="utf-8"))
    integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    mapping_entries = list(mapping["entries"])
    integrity_entries = list(integrity["entries"])
    integrity_by_source = {entry["original_path"]: entry for entry in integrity_entries}
    if len(mapping_entries) != len(integrity_entries) or len(integrity_by_source) != len(mapping_entries):
        raise RuntimeError("Migration ledgers do not contain the same unique source files")

    for mapping_entry in mapping_entries:
        original = str(mapping_entry["original_path"])
        integrity_entry = integrity_by_source.get(original)
        if integrity_entry is None:
            raise RuntimeError(f"Missing integrity entry for {original}")
        copied_relative = str(mapping_entry["copied_legacy_path"])
        if integrity_entry["copied_legacy_path"] != copied_relative:
            raise RuntimeError(f"Copied path differs between ledgers for {original}")
        source = Path(original).resolve(strict=True)
        copied = (repository_root / copied_relative).resolve(strict=True)
        expected = str(mapping_entry["sha256"])
        source_digest = _sha256(source)
        copied_digest = _sha256(copied)
        integrity_digests = {
            str(integrity_entry["source_sha256_at_copy"]),
            str(integrity_entry["source_sha256_after_copy"]),
            str(integrity_entry["legacy_sha256"]),
        }
        if source_digest != expected or copied_digest != expected or integrity_digests != {expected}:
            raise RuntimeError(f"Digest mismatch; refusing to refresh provenance for {original}")
        evidence = _stat_evidence(source, copied)
        mapping_entry.update(evidence)
        integrity_entry.update(evidence)

    mapping["stat_evidence"] = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "copy_timestamp_interpretation": TIMESTAMP_BASIS,
        "mutation_policy": "read/hash/stat sources and legacy copies; write migration ledgers only",
    }
    integrity["stat_evidence"] = dict(mapping["stat_evidence"])
    _replace_text(
        mapping_path,
        yaml.safe_dump(mapping, sort_keys=False, allow_unicode=True, width=120),
    )
    _replace_text(
        integrity_path,
        json.dumps(integrity, indent=2, sort_keys=False, ensure_ascii=False, allow_nan=False) + "\n",
    )
    return len(mapping_entries), len(integrity_entries)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    arguments = parser.parse_args()
    mapping_count, integrity_count = refresh(arguments.repository_root.resolve(strict=True))
    print(json.dumps({"mapping_entries": mapping_count, "integrity_entries": integrity_count}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
