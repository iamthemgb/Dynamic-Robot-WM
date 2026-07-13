"""Source inventories and reproducible generator provenance records."""

from __future__ import annotations

import os
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .hashing import combined_manifest_hash, hash_manifest, sha256_file

INVENTORY_SUFFIXES = {
    ".py",
    ".sh",
    ".yaml",
    ".yml",
    ".json",
    ".toml",
    ".xml",
    ".mjcf",
    ".urdf",
    ".txt",
    ".md",
}
INVENTORY_NAMES = {"README", "requirements", "environment", "pyproject.toml"}
INVENTORY_MAX_FILE_BYTES = 10 * 1024 * 1024
INVENTORY_PRUNED_DIRECTORIES = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "__pycache__",
    ".cache",
    "cache",
    "caches",
    "site-packages",
    "node_modules",
    "third_party",
    "third-party",
    "external",
    "externals",
    "build",
    "dist",
    "outputs",
    "output",
    "datasets",
    "data",
    "videos",
    "frames",
    "checkpoints",
    "logs",
    "runs",
    "wandb",
    "backups",
    ".backups",
    "smoke_tests",
    "previews",
}


@dataclass(slots=True, frozen=True)
class SourceFile:
    """Immutable source-file fingerprint used by the migration ledger."""

    absolute_path: str
    relative_path: str
    size_bytes: int
    mode: str
    owner_uid: int
    group_gid: int
    mtime_ns: int
    sha256: str


@dataclass(slots=True)
class SourceInventory:
    """Read-only recursive inventory, including exact access errors."""

    root: str
    files: list[SourceFile] = field(default_factory=list)
    access_errors: list[dict[str, str]] = field(default_factory=list)
    skipped: list[dict[str, str]] = field(default_factory=list)
    tree_hash: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "files": [asdict(value) for value in self.files]}


def _is_inventory_candidate(path: Path) -> bool:
    name_lower = path.name.lower()
    return (
        path.suffix.lower() in INVENTORY_SUFFIXES
        or name_lower.startswith("readme")
        or name_lower.startswith("requirements")
        or name_lower.startswith("environment")
    )


def inventory_source(root: str | Path) -> SourceInventory:
    """Inventory relevant small source/config files without writing to the root."""

    base = Path(root).resolve(strict=True)
    result = SourceInventory(root=str(base))

    def on_error(error: OSError) -> None:
        result.access_errors.append({"path": error.filename or str(base), "error": str(error)})

    for directory, directories, files in os.walk(base, onerror=on_error, followlinks=False):
        kept_directories: list[str] = []
        for name in directories:
            path = Path(directory) / name
            if name.lower() in INVENTORY_PRUNED_DIRECTORIES or path.is_symlink():
                result.skipped.append(
                    {"path": str(path), "reason": "excluded directory or symlink by inventory policy"}
                )
            else:
                kept_directories.append(name)
        directories[:] = kept_directories
        for name in files:
            path = Path(directory) / name
            if not _is_inventory_candidate(path):
                continue
            try:
                if path.is_symlink():
                    result.skipped.append(
                        {"path": str(path), "reason": "symlinked file excluded by inventory policy"}
                    )
                    continue
                stat = path.stat()
                if stat.st_size > INVENTORY_MAX_FILE_BYTES:
                    result.skipped.append(
                        {
                            "path": str(path),
                            "reason": f"candidate exceeds {INVENTORY_MAX_FILE_BYTES} byte source-file limit",
                        }
                    )
                    continue
                relative = path.relative_to(base).as_posix()
                result.files.append(
                    SourceFile(
                        absolute_path=str(path),
                        relative_path=relative,
                        size_bytes=stat.st_size,
                        mode=oct(stat.st_mode & 0o7777),
                        owner_uid=stat.st_uid,
                        group_gid=stat.st_gid,
                        mtime_ns=stat.st_mtime_ns,
                        sha256=sha256_file(path),
                    )
                )
            except OSError as error:
                result.access_errors.append({"path": str(path), "error": str(error)})
    result.files.sort(key=lambda value: value.relative_path)
    result.access_errors.sort(key=lambda value: value["path"])
    result.skipped.sort(key=lambda value: value["path"])
    result.tree_hash = combined_manifest_hash({value.relative_path: value.sha256 for value in result.files})
    return result


def get_git_commit(path: str | Path) -> str:
    """Return the repository HEAD commit, or ``unknown`` for unversioned sources."""

    try:
        process = subprocess.run(
            ["git", "-C", str(Path(path).resolve()), "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return process.stdout.strip() or "unknown"


@dataclass(slots=True)
class GenerationProvenance:
    """Generator, configuration, simulator, asset, and source identities."""

    source_generator: str
    source_generator_version: str
    generator_git_commit: str
    config_hash: str
    simulator_name: str
    simulator_version: str
    renderer: str
    asset_hashes: dict[str, str] = field(default_factory=dict)
    source_hashes: dict[str, str] = field(default_factory=dict)
    command: list[str] = field(default_factory=list)
    environment_hash: str = ""
    creation_timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def verify_source_snapshot(inventory: SourceInventory) -> list[str]:
    """Return changes since an inventory was captured; never modifies sources."""

    problems: list[str] = []
    for record in inventory.files:
        path = Path(record.absolute_path)
        if not path.is_file():
            problems.append(f"missing: {path}")
            continue
        current = sha256_file(path)
        if current != record.sha256:
            problems.append(f"content changed: {path} ({record.sha256} -> {current})")
    return problems
