"""Read-only, content-pinned dependencies for ``source_mujoco``.

Only scene/model assets are consumed from the external demo repository.  No
controller is imported from it, and no path beneath either dependency root is
opened for writing.  The small source allowlist below is deliberately exact:
changing any consumed builder or robot descriptor blocks execution until the
owned backend is audited and its profile is versioned.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Any, Mapping, Sequence

from ...common.hashing import combined_manifest_hash, sha256_file


DEFAULT_SOURCE_ROOT = Path(
    "/gpfs/radev/project/sous/zl664/demo_mujoco_arm_gripper"
)
DEFAULT_ROBOCASA_ROOT = Path("/gpfs/radev/project/sous/mzl7/robocasa")

# Hashes measured from the read-only source snapshot used for the calibrated
# July 2026 experiments.  Runtime comparison is mandatory; these values are
# not merely descriptive provenance.
PINNED_SOURCE_FILES: Mapping[str, str] = {
    "scripts_mujoco/scene_builder.py": "f0fba618860ddc19754d4803c9c06d96a08a2874ea17831d846d47d8a9ec943d",
    "scripts_mujoco/variants.py": "01f39393cc34d346934cbc876d959e5ce5c2c61776cc027a855338bf543312f5",
    "scripts_mujoco/robocasa_assets.py": "ca716f680e15bea10494977841e01f7878eb6cda661c848886be03d230a4f874",
    "scripts_mujoco/assets.py": "ed8bd6c5e875db63b9ebc4355de8ae2449bda20ea12420ee764f2f0e1ef9ebfa",
    "scripts_mujoco/utils.py": "c80d797b4ee95885ba8c1971df17ce2c2a293fd46d706a6169470c8475d61dfb",
    "third_party/mujoco_menagerie/franka_emika_panda/panda.xml": "96ad67da03710f17f798c9478fd9e9efdf24a3bf8359f05e456dd9fb158ea273",
    "third_party/mujoco_menagerie/franka_emika_panda/panda_nohand.xml": "e7090a5e2384b18a223ef532d98bf914f7f9a6b74dc8bc9f74c8291201074071",
    "third_party/mujoco_menagerie/franka_emika_panda/LICENSE": "a6cba85bc92e0cff7a450b1d873c0eaa2e9fc96bf472df0247a26bec77bf3ff9",
    "third_party/mujoco_menagerie/robotiq_2f85/2f85.xml": "d48aca5f9151798ffd38111ce4e8b2081f3ec2d4f525161b33643451580010de",
    "third_party/mujoco_menagerie/robotiq_2f85/LICENSE": "3d5edd18d1e22a17666e9bb84270171ed5ac90b192d24e0177fd171708ea2316",
}
PINNED_ROBOCASA_LICENSE_SHA256 = (
    "5da18670b3f00c59847b1ded9c28dee59940d963b1e03b528b0108d9c5a09885"
)
PINNED_SOURCE_MANIFEST_SHA256 = (
    "823463e7095fac9a0819cae2688d75df80a7a38ce6c93b1e72e1323fe469ae99"
)


class SourceDependencyError(RuntimeError):
    """An external scene/asset dependency is absent or differs from its pin."""


@dataclass(frozen=True, slots=True)
class SourceDependencyManifest:
    source_root: str
    file_sha256: Mapping[str, str]
    manifest_sha256: str
    read_only_usage: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RoboCasaDependency:
    source_root: str
    asset_root: str
    license_path: str
    license_sha256: str
    usage: str = "visual_only_background"
    collision_enabled: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _resolved_root(
    explicit: str | Path | None,
    *,
    environment_name: str,
    default: Path,
) -> Path:
    raw = explicit
    if raw is None:
        raw = os.environ.get(environment_name) or default
    try:
        root = Path(raw).expanduser().resolve(strict=True)
    except (FileNotFoundError, OSError) as error:
        raise SourceDependencyError(
            f"required dependency root is unavailable: {raw}"
        ) from error
    if not root.is_dir():
        raise SourceDependencyError(f"dependency root is not a directory: {root}")
    return root


def resolve_source_dependency(
    source_root: str | Path | None = None,
) -> SourceDependencyManifest:
    """Resolve the external demo tree and require every consumed file pin."""

    root = _resolved_root(
        source_root,
        environment_name="SOURCE_MUJOCO_SCENE_ROOT",
        default=DEFAULT_SOURCE_ROOT,
    )
    actual: dict[str, str] = {}
    failures: list[str] = []
    for relative, expected in PINNED_SOURCE_FILES.items():
        path = root / relative
        if not path.is_file():
            failures.append(f"missing:{relative}")
            continue
        digest = sha256_file(path)
        actual[relative] = digest
        if digest != expected:
            failures.append(f"hash_mismatch:{relative}")
    if failures:
        raise SourceDependencyError(
            "source_mujoco scene dependency is not the calibrated pinned snapshot: "
            + ", ".join(failures)
        )
    ordered = dict(sorted(actual.items()))
    manifest_sha256 = combined_manifest_hash(ordered)
    if manifest_sha256 != PINNED_SOURCE_MANIFEST_SHA256:
        raise SourceDependencyError(
            "the verified ten-file source manifest differs from its aggregate pin"
        )
    return SourceDependencyManifest(
        source_root=str(root),
        file_sha256=ordered,
        manifest_sha256=manifest_sha256,
    )


def resolve_robocasa_dependency(
    robocasa_root: str | Path | None = None,
) -> RoboCasaDependency:
    """Require the real RoboCasa tree and its pinned license notice."""

    root = _resolved_root(
        robocasa_root,
        environment_name="SOURCE_MUJOCO_ROBOCASA_ROOT",
        default=DEFAULT_ROBOCASA_ROOT,
    )
    asset_root = root / "robocasa/models/assets"
    license_path = root / "LICENSE"
    if not asset_root.is_dir():
        raise SourceDependencyError(
            f"real RoboCasa asset root is unavailable: {asset_root}"
        )
    if not license_path.is_file():
        raise SourceDependencyError(
            f"RoboCasa license metadata is unavailable: {license_path}"
        )
    license_sha256 = sha256_file(license_path)
    if license_sha256 != PINNED_ROBOCASA_LICENSE_SHA256:
        raise SourceDependencyError(
            "RoboCasa license differs from the pinned reviewed dependency"
        )
    return RoboCasaDependency(
        source_root=str(root),
        asset_root=str(asset_root),
        license_path=str(license_path),
        license_sha256=license_sha256,
    )


def referenced_asset_manifest(
    xml: str,
    *,
    allowed_roots: Sequence[str | Path],
) -> dict[str, str]:
    """Hash every file referenced by a compiled MJCF string.

    External builders normalize RoboCasa paths to absolute paths.  Menagerie
    mesh paths remain relative to the absolute compiler mesh/texture dirs, so
    this resolver handles both forms and rejects missing or escaping files.
    """

    try:
        root = ET.fromstring(xml)
    except ET.ParseError as error:
        raise SourceDependencyError("compiled scene XML is malformed") from error
    compiler = root.find("compiler")
    meshdir = Path(compiler.get("meshdir")) if compiler is not None and compiler.get("meshdir") else None
    texturedir = (
        Path(compiler.get("texturedir"))
        if compiler is not None and compiler.get("texturedir")
        else None
    )
    roots = tuple(Path(value).resolve(strict=True) for value in allowed_roots)
    if not roots:
        raise SourceDependencyError("compiled MJCF asset hashing requires allowed roots")
    manifest: dict[str, str] = {}
    for element in root.findall(".//*[@file]"):
        raw = element.get("file")
        if not raw:
            continue
        path = Path(raw)
        if not path.is_absolute():
            base = texturedir if element.tag == "texture" and texturedir is not None else meshdir
            if base is None:
                raise SourceDependencyError(
                    f"relative MJCF asset has no compiler directory: {raw}"
                )
            path = base / path
        try:
            resolved = path.expanduser().resolve(strict=True)
        except (FileNotFoundError, OSError) as error:
            raise SourceDependencyError(f"compiled MJCF asset is unavailable: {path}") from error
        if not resolved.is_file():
            raise SourceDependencyError(f"compiled MJCF asset is not a file: {resolved}")
        if not any(
            resolved == root or root in resolved.parents
            for root in roots
        ):
            raise SourceDependencyError(
                f"compiled MJCF asset escapes the verified source/RoboCasa roots: {resolved}"
            )
        manifest[str(resolved)] = sha256_file(resolved)
    if not manifest:
        raise SourceDependencyError("compiled MJCF has no content-bound asset files")
    return dict(sorted(manifest.items()))


__all__ = [
    "DEFAULT_ROBOCASA_ROOT",
    "DEFAULT_SOURCE_ROOT",
    "PINNED_ROBOCASA_LICENSE_SHA256",
    "PINNED_SOURCE_MANIFEST_SHA256",
    "PINNED_SOURCE_FILES",
    "RoboCasaDependency",
    "SourceDependencyError",
    "SourceDependencyManifest",
    "referenced_asset_manifest",
    "resolve_robocasa_dependency",
    "resolve_source_dependency",
]
