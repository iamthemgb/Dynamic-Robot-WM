"""Admission checks for external simulation and visual assets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any

from .hashing import sha256_file


@dataclass(slots=True, frozen=True)
class AssetAdmission:
    asset_id: str
    descriptor_path: str
    source_root: str
    scale_to_meters: float | None
    collision_validated: bool
    render_validated: bool
    license_notice_path: str | None
    descriptor_sha256: str
    admitted: bool
    blockers: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def inspect_asset_for_admission(
    asset_id: str,
    descriptor_path: str | Path,
    *,
    source_root: str | Path,
    scale_to_meters: float | None,
    collision_validated: bool,
    render_validated: bool,
    license_notice_path: str | Path | None,
) -> AssetAdmission:
    """Admit only assets with explicit scale, collision, render, and notice evidence."""

    root = Path(source_root).resolve(strict=True)
    descriptor = Path(descriptor_path).resolve(strict=True)
    try:
        descriptor.relative_to(root)
    except ValueError as error:
        raise ValueError(f"asset descriptor escapes configured source root: {descriptor}") from error
    blockers: list[str] = []
    if scale_to_meters is None or not math.isfinite(scale_to_meters) or scale_to_meters <= 0:
        blockers.append("scale_not_validated")
    if not collision_validated:
        blockers.append("collision_not_validated")
    if not render_validated:
        blockers.append("render_not_validated")
    notice_value: str | None = None
    if license_notice_path is None:
        blockers.append("license_notice_missing")
    else:
        notice = Path(license_notice_path).resolve(strict=True)
        try:
            notice.relative_to(root)
        except ValueError as error:
            raise ValueError(f"license notice escapes configured source root: {notice}") from error
        notice_value = str(notice)
    return AssetAdmission(
        asset_id=asset_id,
        descriptor_path=str(descriptor),
        source_root=str(root),
        scale_to_meters=scale_to_meters,
        collision_validated=collision_validated,
        render_validated=render_validated,
        license_notice_path=notice_value,
        descriptor_sha256=sha256_file(descriptor),
        admitted=not blockers,
        blockers=tuple(blockers),
    )
