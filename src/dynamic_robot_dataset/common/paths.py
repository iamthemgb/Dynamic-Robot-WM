"""Portable dataset paths and non-destructive atomic filesystem operations."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import ctypes
import errno
from contextlib import AbstractContextManager
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from .hashing import canonical_json_bytes, sha256_json


# ``paths.py`` lives at ``<repository>/src/dynamic_robot_dataset/common`` in
# both the source checkout and the editable environment used by this project.
# The migrated bytes are provenance inputs, not a legal dataset output target.
REPOSITORY_LEGACY_SOURCE_ROOT = Path(__file__).resolve().parents[3] / "legacy_sources"

DEFAULT_READ_ONLY_SOURCE_ROOTS = (
    Path("/gpfs/radev/scratch/sous/mzl7"),
    Path("/gpfs/radev/scratch/sous/zl664"),
    Path("/gpfs/radev/scratch/sous/zss8"),
    # Canonical collaborator-owned generator trees inspected during migration.
    Path("/gpfs/radev/project/sous/mzl7"),
    Path("/gpfs/radev/project/sous/zl664/demo_mujoco_arm_gripper"),
    Path("/gpfs/radev/project/sous/zl664/demo_mujoco_deformable"),
    Path("/gpfs/radev/project/sous/zss8/dataset-generation"),
    Path("/gpfs/radev/home/zl664/project/demo_mujoco_arm_gripper"),
    Path("/gpfs/radev/home/zl664/project/demo_mujoco_deformable"),
    # This repository must never mutate the separately scoped Wan experiment.
    Path("/gpfs/radev/project/sous/zl664/wan_scripts"),
    # The byte-identical migration snapshot is immutable after initialization.
    # This guard is used only by runtime output helpers; it does not participate
    # in repository creation or the one-time, already-completed source copy.
    REPOSITORY_LEGACY_SOURCE_ROOT,
)


class ExistingOutputError(FileExistsError):
    """Raised when an operation would silently overwrite an existing artifact."""


class ResumeMismatchError(RuntimeError):
    """Raised when resume is requested with a different resolved configuration."""


def portable_relative_path(value: str | os.PathLike[str]) -> str:
    """Validate and normalize a dataset-relative POSIX path.

    Absolute paths, parent traversal, empty paths, and backslash-based paths are
    rejected.  Metadata can therefore be moved together with its dataset root.
    """

    raw = os.fspath(value)
    if not raw or "\\" in raw:
        raise ValueError(f"Not a portable relative path: {raw!r}")
    path = PurePosixPath(raw)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"Not a portable relative path: {raw!r}")
    return path.as_posix()


def resolve_dataset_path(root: str | Path, relative: str | os.PathLike[str]) -> Path:
    """Resolve a validated relative path beneath a dataset root."""

    base = Path(root).resolve()
    normalized = portable_relative_path(relative)
    candidate = (base / normalized).resolve()
    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise ValueError(f"Path escapes dataset root: {relative}") from exc
    return candidate


def as_dataset_relative(path: str | Path, root: str | Path) -> str:
    """Return a portable dataset-relative path for an existing or planned path."""

    base = Path(root).resolve()
    candidate = Path(path).resolve()
    try:
        relative = candidate.relative_to(base)
    except ValueError as exc:
        raise ValueError(f"Path is outside dataset root: {candidate}") from exc
    return portable_relative_path(relative.as_posix())


def ensure_not_source_path(
    path: str | Path,
    source_roots: Iterable[str | Path] = DEFAULT_READ_ONLY_SOURCE_ROOTS,
) -> Path:
    """Reject a write target located inside any immutable source root."""

    candidate = Path(path).resolve()
    for raw_root in source_roots:
        root = Path(raw_root).resolve()
        if candidate == root or root in candidate.parents:
            raise PermissionError(f"Refusing to write inside read-only source root: {candidate}")
    return candidate


def ensure_output_root(path: str | Path, *, create: bool = True) -> Path:
    """Validate an output root and optionally create it."""

    root = ensure_not_source_path(path)
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root


def _link_commit(temp_path: Path, destination: Path) -> None:
    """Atomically publish a temporary file without replacement."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(temp_path, destination)
    except FileExistsError as exc:
        raise ExistingOutputError(f"Output already exists: {destination}") from exc
    finally:
        temp_path.unlink(missing_ok=True)


def atomic_write_bytes(destination: str | Path, data: bytes, mode: int = 0o644) -> Path:
    """Write bytes atomically, failing rather than replacing an existing file."""

    target = ensure_not_source_path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    temp_path = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temp_path, mode)
        _link_commit(temp_path, target)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
    return target


def atomic_write_json(destination: str | Path, value: Any, *, indent: int = 2) -> Path:
    """Serialize JSON atomically without overwriting existing output."""

    payload = json.dumps(value, indent=indent, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return atomic_write_bytes(destination, (payload + "\n").encode("utf-8"))


def atomic_copy(source: str | Path, destination: str | Path) -> Path:
    """Copy a file and publish it atomically without replacement."""

    src = Path(source).resolve(strict=True)
    target = ensure_not_source_path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    os.close(fd)
    temp_path = Path(temporary)
    try:
        shutil.copyfile(src, temp_path)
        with temp_path.open("rb") as handle:
            os.fsync(handle.fileno())
        _link_commit(temp_path, target)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
    return target


class AtomicDirectory(AbstractContextManager[Path]):
    """Build a directory privately and publish it with a no-replace rename.

    The temporary directory lives beside the final directory, keeping the rename
    on one filesystem.  A failed context is removed; an existing destination is
    never touched.
    """

    def __init__(self, destination: str | Path):
        self.destination = ensure_not_source_path(destination)
        self.temporary: Path | None = None

    def __enter__(self) -> Path:
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        if self.destination.exists():
            raise ExistingOutputError(f"Output already exists: {self.destination}")
        self.temporary = Path(
            tempfile.mkdtemp(prefix=f".{self.destination.name}.", suffix=".tmp", dir=self.destination.parent)
        )
        return self.temporary

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        if self.temporary is None:
            return False
        if exc_type is not None:
            shutil.rmtree(self.temporary, ignore_errors=True)
            return False
        try:
            _rename_directory_noreplace(self.temporary, self.destination)
        except FileExistsError as error:
            shutil.rmtree(self.temporary, ignore_errors=True)
            raise ExistingOutputError(f"Output already exists: {self.destination}") from error
        return False


def _rename_directory_noreplace(source: Path, destination: Path) -> None:
    """Use Linux ``renameat2(RENAME_NOREPLACE)`` with a locked fallback."""

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is not None:
        renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            -100,
            os.fsencode(source),
            -100,
            os.fsencode(destination),
            1,  # RENAME_NOREPLACE
        )
        if result == 0:
            return
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise FileExistsError(destination)
        if error_number not in {errno.ENOSYS, errno.EINVAL, errno.EOPNOTSUPP}:
            raise OSError(error_number, os.strerror(error_number), destination)
    lock = destination.parent / f".{destination.name}.publish.lock"
    try:
        descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise FileExistsError(destination) from exc
    try:
        os.close(descriptor)
        if destination.exists():
            raise FileExistsError(destination)
        os.rename(source, destination)
    finally:
        lock.unlink(missing_ok=True)


class ResumeGuard:
    """Protect a generation root with an immutable resolved-configuration hash."""

    FILE_NAME = ".generation.json"

    def __init__(self, output: str | Path, config: Any, *, resume: bool = False):
        self.output = ensure_not_source_path(output)
        self.config = config
        self.config_hash = sha256_json(config)
        self.resume = resume

    @property
    def marker(self) -> Path:
        return self.output / self.FILE_NAME

    def initialize(self) -> str:
        """Create or verify the generation marker and return its config hash."""

        if self.output.exists() and any(self.output.iterdir()):
            if not self.resume:
                raise ExistingOutputError(
                    f"Non-empty output exists; pass --resume with the identical config: {self.output}"
                )
            if not self.marker.is_file():
                raise ResumeMismatchError(f"Resume marker is missing: {self.marker}")
            marker = json.loads(self.marker.read_text(encoding="utf-8"))
            if marker.get("config_hash") != self.config_hash:
                raise ResumeMismatchError(
                    f"Resolved config changed: expected {marker.get('config_hash')}, got {self.config_hash}"
                )
            return self.config_hash
        self.output.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            self.marker,
            {"config_hash": self.config_hash, "resolved_config": self.config},
        )
        return self.config_hash
