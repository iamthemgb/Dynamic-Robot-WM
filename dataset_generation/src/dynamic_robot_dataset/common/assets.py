"""Admission checks for external simulation and visual assets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Sequence
import xml.etree.ElementTree as ET

import yaml

from .hashing import combined_manifest_hash, sha256_file


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


@dataclass(slots=True, frozen=True)
class AxisAlignedBoundingBox:
    """World-space bounds persisted for scene-clearance review."""

    minimum_m: tuple[float, float, float]
    maximum_m: tuple[float, float, float]

    def validate(self) -> None:
        if len(self.minimum_m) != 3 or len(self.maximum_m) != 3:
            raise ValueError("AABB bounds must be XYZ triples")
        if any(
            not math.isfinite(float(value))
            for value in (*self.minimum_m, *self.maximum_m)
        ):
            raise ValueError("AABB bounds must be finite")
        if any(low > high for low, high in zip(self.minimum_m, self.maximum_m)):
            raise ValueError("AABB minimum cannot exceed maximum")

    def intersects(self, other: "AxisAlignedBoundingBox") -> bool:
        self.validate()
        other.validate()
        return all(
            left_high >= right_low and right_high >= left_low
            for left_low, left_high, right_low, right_high in zip(
                self.minimum_m,
                self.maximum_m,
                other.minimum_m,
                other.maximum_m,
            )
        )


@dataclass(slots=True, frozen=True)
class RoboCasaBackgroundAdmission:
    """Content-bound admission evidence for one visual-only scene asset."""

    schema_version: str
    catalog_version: str
    asset_id: str
    source_root: str
    descriptor_path: str
    referenced_file_sha256: Mapping[str, str]
    manifest_sha256: str
    license_notice_path: str | None
    license_notice_sha256: str | None
    scale_to_meters: float | None
    world_aabb: AxisAlignedBoundingBox
    transform_row_major_4x4: tuple[float, ...]
    visual_only: bool
    collision_enabled: bool
    swept_volume_clear: bool
    fixture_intersection_clear: bool
    occlusion_validated: bool
    admitted: bool
    blockers: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class RoboCasaCatalogPolicy:
    """Resolved, read-only RoboCasa catalog root and license binding."""

    schema_version: str
    catalog_version: str
    catalog_path: str
    catalog_sha256: str
    source_root: str
    asset_root: str
    license_notice_path: str
    license_notice_sha256: str
    profiles: tuple[str, ...]
    assets: tuple["RoboCasaCatalogAsset", ...]
    configured_asset_count: int
    review_candidate_asset_count: int
    admitted_asset_count: int
    review_ready_profiles: tuple[str, ...]
    release_ready_profiles: tuple[str, ...]
    review_ready: bool
    release_ready: bool
    readiness_blockers: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class RoboCasaCatalogAsset:
    """A static, content-bound candidate for rendered scene admission.

    Static catalog rows intentionally do not claim scene-specific clearance or
    rendered occlusion before a compiled review scene exists.  A row may be a
    review candidate while remaining unadmitted for release.
    """

    asset_id: str
    profiles: tuple[str, ...]
    descriptor_path: str
    descriptor_sha256: str
    referenced_file_sha256: Mapping[str, str]
    manifest_sha256: str
    license_notice_sha256: str
    scale_to_meters: float
    visual_only: bool
    collision_enabled: bool
    scene_clearance_validated: bool
    fixture_intersection_validated: bool
    occlusion_validated: bool
    admitted: bool
    blockers: tuple[str, ...]
    review_candidate: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_ROBOCASA_REVIEW_PROFILES = frozenset(
    {"lab", "kitchen", "workbench", "storage", "tabletop"}
)
_ROBOCASA_CANDIDATE_PENDING_BLOCKERS = frozenset(
    {
        "scene_specific_clearance_pending",
        "fixture_intersection_clearance_pending",
        "rendered_occlusion_review_pending",
    }
)
_ROBOCASA_CATALOG_ASSET_KEYS = frozenset(
    {
        "asset_id",
        "profiles",
        "descriptor_path",
        "descriptor_sha256",
        "referenced_file_sha256",
        "manifest_sha256",
        "license_notice_sha256",
        "scale_to_meters",
        "visual_only",
        "collision_enabled",
        "scene_clearance_validated",
        "fixture_intersection_validated",
        "occlusion_validated",
        "admitted",
        "blockers",
    }
)
_ROBOCASA_REVIEW_CANDIDATE_POLICY = {
    "require_descriptor_mesh_texture_hashes": True,
    "require_license_hash": True,
    "require_scale_to_meters": True,
    "visual_only": True,
    "collision_enabled": False,
    "allowed_pending_validations": [
        "scene_specific_clearance",
        "fixture_intersection_clearance",
        "rendered_occlusion_review",
    ],
}
_ROBOCASA_RELEASE_ADMISSION_POLICY = {
    "require_world_aabb_and_transform": True,
    "require_task_swept_volume_clearance": True,
    "require_fixture_intersection_clearance": True,
    "require_rendered_occlusion_validation": True,
    "require_admitted": True,
}


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _descriptor_content_paths(root: Path, descriptor: Path) -> tuple[Path, ...]:
    """Resolve every mesh/texture file loaded by one simple RoboCasa MJCF."""

    try:
        xml_root = ET.parse(descriptor).getroot()
    except ET.ParseError as error:
        raise ValueError(f"RoboCasa descriptor is not valid XML: {descriptor}") from error
    if xml_root.findall(".//include"):
        raise ValueError(
            f"RoboCasa catalog candidates with MJCF includes are unsupported: {descriptor}"
        )
    compiler = xml_root.find("compiler")
    mesh_dir = Path(compiler.get("meshdir", "")) if compiler is not None else Path()
    texture_dir = (
        Path(compiler.get("texturedir", "")) if compiler is not None else Path()
    )
    paths = {descriptor}
    for tag, relative_dir in (("mesh", mesh_dir), ("texture", texture_dir)):
        for element in xml_root.findall(f".//asset/{tag}"):
            file_value = str(element.get("file") or "").strip()
            if not file_value:
                continue
            candidate = Path(file_value)
            if not candidate.is_absolute():
                candidate = descriptor.parent / relative_dir / candidate
            resolved, _ = _asset_relative_file(
                root, candidate, label=f"RoboCasa {tag} content"
            )
            if not resolved.is_file():
                raise FileNotFoundError(
                    f"RoboCasa descriptor references unavailable {tag}: {resolved}"
                )
            paths.add(resolved)
    return tuple(sorted(paths))


def _load_catalog_asset(
    raw: Mapping[str, Any],
    *,
    root: Path,
    license_sha256: str,
) -> RoboCasaCatalogAsset:
    if set(raw) != _ROBOCASA_CATALOG_ASSET_KEYS:
        raise ValueError("RoboCasa catalog asset keys differ from the candidate contract")
    asset_id = str(raw.get("asset_id") or "").strip()
    if not asset_id:
        raise ValueError("RoboCasa catalog assets need non-empty asset_id values")
    raw_profiles = raw.get("profiles")
    if not isinstance(raw_profiles, Sequence) or isinstance(
        raw_profiles, (str, bytes, bytearray)
    ):
        raise ValueError(f"RoboCasa catalog asset {asset_id} profiles must be a sequence")
    profiles = tuple(str(value) for value in raw_profiles)
    if not profiles or len(profiles) != len(set(profiles)):
        raise ValueError(f"RoboCasa catalog asset {asset_id} profiles must be unique")
    if not set(profiles) <= _ROBOCASA_REVIEW_PROFILES:
        raise ValueError(f"RoboCasa catalog asset {asset_id} has an unknown review profile")

    descriptor, descriptor_relative = _asset_relative_file(
        root, str(raw.get("descriptor_path") or ""), label="RoboCasa descriptor"
    )
    if not descriptor.is_file():
        raise FileNotFoundError(f"RoboCasa descriptor is unavailable: {descriptor}")
    actual_descriptor_sha256 = sha256_file(descriptor)
    if raw.get("descriptor_sha256") != actual_descriptor_sha256:
        raise ValueError(f"RoboCasa descriptor hash changed for {asset_id}")

    declared_manifest = raw.get("referenced_file_sha256")
    if not isinstance(declared_manifest, Mapping) or not declared_manifest:
        raise ValueError(f"RoboCasa catalog asset {asset_id} lacks a content manifest")
    if any(not isinstance(path, str) or not path for path in declared_manifest):
        raise ValueError(f"RoboCasa catalog asset {asset_id} has malformed manifest paths")
    if any(not _valid_sha256(digest) for digest in declared_manifest.values()):
        raise ValueError(f"RoboCasa catalog asset {asset_id} has malformed content hashes")
    expected_paths = _descriptor_content_paths(root, descriptor)
    actual_manifest = {
        path.relative_to(root).as_posix(): sha256_file(path) for path in expected_paths
    }
    if dict(sorted(declared_manifest.items())) != actual_manifest:
        raise ValueError(
            f"RoboCasa catalog asset {asset_id} manifest does not exactly bind its "
            "descriptor, meshes, and textures"
        )
    if actual_manifest.get(descriptor_relative) != actual_descriptor_sha256:
        raise ValueError(f"RoboCasa descriptor is absent from the manifest for {asset_id}")
    manifest_sha256 = combined_manifest_hash(actual_manifest)
    if raw.get("manifest_sha256") != manifest_sha256:
        raise ValueError(f"RoboCasa manifest hash changed for {asset_id}")
    if raw.get("license_notice_sha256") != license_sha256:
        raise ValueError(f"RoboCasa license binding changed for {asset_id}")

    scale = raw.get("scale_to_meters")
    if (
        isinstance(scale, bool)
        or not isinstance(scale, (int, float))
        or not math.isfinite(float(scale))
        or float(scale) <= 0.0
    ):
        raise ValueError(f"RoboCasa scale is not validated for {asset_id}")
    if raw.get("visual_only") is not True or raw.get("collision_enabled") is not False:
        raise ValueError(
            f"RoboCasa catalog asset {asset_id} must be visual-only with collisions disabled"
        )
    validation_fields = (
        "scene_clearance_validated",
        "fixture_intersection_validated",
        "occlusion_validated",
        "admitted",
    )
    if any(type(raw.get(field)) is not bool for field in validation_fields):
        raise ValueError(f"RoboCasa catalog asset {asset_id} validation flags must be booleans")
    derived_blockers: set[str] = set()
    if raw["scene_clearance_validated"] is not True:
        derived_blockers.add("scene_specific_clearance_pending")
    if raw["fixture_intersection_validated"] is not True:
        derived_blockers.add("fixture_intersection_clearance_pending")
    if raw["occlusion_validated"] is not True:
        derived_blockers.add("rendered_occlusion_review_pending")
    blockers_value = raw.get("blockers")
    if not isinstance(blockers_value, Sequence) or isinstance(
        blockers_value, (str, bytes, bytearray)
    ):
        raise ValueError(f"RoboCasa catalog asset {asset_id} blockers must be a sequence")
    blockers = tuple(sorted(str(value) for value in blockers_value))
    if blockers != tuple(sorted(derived_blockers)):
        raise ValueError(f"RoboCasa catalog asset {asset_id} blockers disagree with evidence")
    admitted = raw["admitted"] is True
    if admitted != (not blockers):
        raise ValueError(f"RoboCasa catalog asset {asset_id} has a false admission claim")
    review_candidate = set(blockers) <= _ROBOCASA_CANDIDATE_PENDING_BLOCKERS
    return RoboCasaCatalogAsset(
        asset_id=asset_id,
        profiles=profiles,
        descriptor_path=str(descriptor),
        descriptor_sha256=actual_descriptor_sha256,
        referenced_file_sha256=actual_manifest,
        manifest_sha256=manifest_sha256,
        license_notice_sha256=license_sha256,
        scale_to_meters=float(scale),
        visual_only=True,
        collision_enabled=False,
        scene_clearance_validated=raw["scene_clearance_validated"],
        fixture_intersection_validated=raw["fixture_intersection_validated"],
        occlusion_validated=raw["occlusion_validated"],
        admitted=admitted,
        blockers=blockers,
        review_candidate=review_candidate,
    )


def load_robocasa_catalog_policy(
    path: str | Path,
    *,
    source_root_override: str | Path | None = None,
) -> RoboCasaCatalogPolicy:
    """Resolve a catalog and fail if its external root/license is unavailable."""

    catalog_path = Path(path).resolve(strict=True)
    raw = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("RoboCasa catalog must contain a YAML mapping")
    if raw.get("schema_version") != "dynamic-robot-robocasa-catalog/v1":
        raise ValueError("unsupported RoboCasa catalog schema")
    if raw.get("read_only") is not True or raw.get("copy_external_assets") is not False:
        raise ValueError("RoboCasa catalog must keep its external source read-only")
    if raw.get("review_candidate_policy") != _ROBOCASA_REVIEW_CANDIDATE_POLICY:
        raise ValueError("RoboCasa review-candidate policy differs from the fail-closed contract")
    if raw.get("release_admission_policy") != _ROBOCASA_RELEASE_ADMISSION_POLICY:
        raise ValueError("RoboCasa release-admission policy differs from the fail-closed contract")
    root = Path(source_root_override or str(raw.get("source_root") or "")).resolve(
        strict=True
    )
    asset_root, _ = _asset_relative_file(
        root, str(raw.get("asset_root") or ""), label="RoboCasa asset root"
    )
    if not asset_root.is_dir():
        raise FileNotFoundError(f"RoboCasa asset root is unavailable: {asset_root}")
    notice, _ = _asset_relative_file(
        root, str(raw.get("license_notice") or ""), label="RoboCasa license"
    )
    if not notice.is_file():
        raise FileNotFoundError(f"RoboCasa license notice is unavailable: {notice}")
    profiles = tuple(str(value) for value in raw.get("profiles", ()))
    if set(profiles) != _ROBOCASA_REVIEW_PROFILES or len(profiles) != len(
        _ROBOCASA_REVIEW_PROFILES
    ):
        raise ValueError("RoboCasa catalog must declare the five review profiles")
    assets = raw.get("assets")
    if not isinstance(assets, Sequence) or isinstance(assets, (str, bytes, bytearray)):
        raise ValueError("RoboCasa catalog assets must be a sequence")
    license_sha256 = sha256_file(notice)
    resolved_assets: list[RoboCasaCatalogAsset] = []
    seen_asset_ids: set[str] = set()
    for value in assets:
        if not isinstance(value, Mapping):
            raise ValueError("RoboCasa catalog asset rows must be mappings")
        asset = _load_catalog_asset(
            value,
            root=root,
            license_sha256=license_sha256,
        )
        if asset.asset_id in seen_asset_ids:
            raise ValueError(f"duplicate RoboCasa catalog asset ID: {asset.asset_id}")
        seen_asset_ids.add(asset.asset_id)
        resolved_assets.append(asset)
    review_ready_profiles = tuple(
        profile
        for profile in profiles
        if any(
            profile in asset.profiles and asset.review_candidate
            for asset in resolved_assets
        )
    )
    release_ready_profiles = tuple(
        profile
        for profile in profiles
        if any(
            profile in asset.profiles
            and asset.admitted
            and asset.occlusion_validated
            for asset in resolved_assets
        )
    )
    review_ready = set(review_ready_profiles) == _ROBOCASA_REVIEW_PROFILES
    release_ready = set(release_ready_profiles) == _ROBOCASA_REVIEW_PROFILES
    readiness_blockers = tuple(
        [
            f"review_profile_has_no_content_bound_candidate:{profile}"
            for profile in profiles
            if profile not in review_ready_profiles
        ]
        + [
            f"release_profile_has_no_admitted_occlusion_validated_asset:{profile}"
            for profile in profiles
            if profile not in release_ready_profiles
        ]
    )
    return RoboCasaCatalogPolicy(
        schema_version="dynamic-robot-robocasa-catalog/v1",
        catalog_version=str(raw.get("catalog_version") or ""),
        catalog_path=str(catalog_path),
        catalog_sha256=sha256_file(catalog_path),
        source_root=str(root),
        asset_root=str(asset_root),
        license_notice_path=str(notice),
        license_notice_sha256=license_sha256,
        profiles=profiles,
        assets=tuple(resolved_assets),
        configured_asset_count=len(resolved_assets),
        review_candidate_asset_count=sum(
            int(asset.review_candidate) for asset in resolved_assets
        ),
        admitted_asset_count=sum(int(asset.admitted) for asset in resolved_assets),
        review_ready_profiles=review_ready_profiles,
        release_ready_profiles=release_ready_profiles,
        review_ready=review_ready,
        release_ready=release_ready,
        readiness_blockers=readiness_blockers,
    )


def _asset_relative_file(
    root: Path,
    raw_path: str | Path,
    *,
    label: str,
) -> tuple[Path, str]:
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes configured RoboCasa root: {resolved}") from error
    return resolved, relative.as_posix()


def inspect_robocasa_background_asset(
    asset_id: str,
    descriptor_path: str | Path,
    *,
    source_root: str | Path,
    catalog_version: str,
    referenced_files: Sequence[str | Path],
    license_notice_path: str | Path | None,
    scale_to_meters: float | None,
    world_aabb: AxisAlignedBoundingBox,
    transform_row_major_4x4: Sequence[float],
    collision_enabled: bool,
    visual_only: bool,
    task_swept_volume: AxisAlignedBoundingBox,
    fixture_aabbs: Sequence[AxisAlignedBoundingBox] = (),
    occlusion_validated: bool,
) -> RoboCasaBackgroundAdmission:
    """Inspect a RoboCasa background asset without copying or modifying it.

    A caller must provide the bounds and collision state from the compiled
    scene, not merely from the source MJCF.  Any missing provenance or spatial
    evidence leaves the entry present but unadmitted so planning can report the
    precise blocker without falling back to procedural stand-ins.
    """

    if not asset_id.strip() or not catalog_version.strip():
        raise ValueError("asset_id and catalog_version are required")
    root = Path(source_root).resolve(strict=True)
    descriptor, descriptor_relative = _asset_relative_file(
        root, descriptor_path, label="asset descriptor"
    )
    blockers: list[str] = []
    manifest: dict[str, str] = {}
    if not descriptor.is_file():
        blockers.append("descriptor_missing")
    else:
        manifest[descriptor_relative] = sha256_file(descriptor)
    for raw_path in referenced_files:
        path, relative = _asset_relative_file(root, raw_path, label="referenced asset")
        if not path.is_file():
            blockers.append(f"referenced_file_missing:{relative}")
        else:
            manifest[relative] = sha256_file(path)
    notice_path_value: str | None = None
    notice_hash: str | None = None
    if license_notice_path is None:
        blockers.append("license_notice_missing")
    else:
        notice, notice_relative = _asset_relative_file(
            root, license_notice_path, label="license notice"
        )
        notice_path_value = str(notice)
        if not notice.is_file():
            blockers.append("license_notice_missing")
        else:
            notice_hash = sha256_file(notice)
            manifest[notice_relative] = notice_hash
    if scale_to_meters is None or not math.isfinite(scale_to_meters) or scale_to_meters <= 0:
        blockers.append("scale_not_validated")
    world_aabb.validate()
    task_swept_volume.validate()
    transform = tuple(float(value) for value in transform_row_major_4x4)
    if len(transform) != 16 or any(not math.isfinite(value) for value in transform):
        raise ValueError("asset transform must be a finite row-major 4x4 matrix")
    if collision_enabled or not visual_only:
        blockers.append("background_collision_not_disabled")
    swept_volume_clear = not world_aabb.intersects(task_swept_volume)
    if not swept_volume_clear:
        blockers.append("background_intersects_task_swept_volume")
    fixture_intersection_clear = True
    for fixture in fixture_aabbs:
        fixture.validate()
        if world_aabb.intersects(fixture):
            fixture_intersection_clear = False
            break
    if not fixture_intersection_clear:
        blockers.append("background_intersects_physical_fixture")
    if not occlusion_validated:
        blockers.append("background_occlusion_not_validated")
    if not manifest:
        blockers.append("empty_content_manifest")
    manifest = dict(sorted(manifest.items()))
    return RoboCasaBackgroundAdmission(
        schema_version="dynamic-robot-robocasa-admission/v1",
        catalog_version=catalog_version,
        asset_id=asset_id,
        source_root=str(root),
        descriptor_path=str(descriptor),
        referenced_file_sha256=manifest,
        manifest_sha256=combined_manifest_hash(manifest),
        license_notice_path=notice_path_value,
        license_notice_sha256=notice_hash,
        scale_to_meters=scale_to_meters,
        world_aabb=world_aabb,
        transform_row_major_4x4=transform,
        visual_only=visual_only,
        collision_enabled=collision_enabled,
        swept_volume_clear=swept_volume_clear,
        fixture_intersection_clear=fixture_intersection_clear,
        occlusion_validated=occlusion_validated,
        admitted=not blockers,
        blockers=tuple(sorted(set(blockers))),
    )


def validate_robocasa_asset_manifest(
    values: Sequence[Mapping[str, Any] | RoboCasaBackgroundAdmission],
    *,
    required_asset_ids: Sequence[str] = (),
    allow_pending_render_review: bool = False,
) -> None:
    """Validate selected, content-bound assets for review or release.

    Release validation remains strict: ``admitted`` must be true and blockers
    empty.  A review-only caller may explicitly allow the single remaining
    ``rendered_occlusion_review_pending`` blocker, but only after collision,
    swept-volume, fixture, scale, hash, and license checks already pass.  That
    narrow exception lets automated review QC finish without converting a
    pending human decision into a release claim.
    """

    by_id: dict[str, Mapping[str, Any]] = {}
    for raw in values:
        value = raw.to_dict() if isinstance(raw, RoboCasaBackgroundAdmission) else raw
        asset_id = str(value.get("asset_id") or "")
        if not asset_id or asset_id in by_id:
            raise ValueError("RoboCasa admission manifest needs unique non-empty asset IDs")
        by_id[asset_id] = value
        if value.get("schema_version") != "dynamic-robot-robocasa-admission/v1":
            raise ValueError(f"RoboCasa asset {asset_id} uses an unsupported admission schema")
        blockers = tuple(str(item) for item in value.get("blockers") or ())
        pending_render_review = (
            allow_pending_render_review
            and value.get("admitted") is False
            and blockers == ("rendered_occlusion_review_pending",)
            and value.get("selection_source") == "owned_content_bound_catalog"
            and value.get("visual_only") is True
            and value.get("collision_enabled") is False
            and value.get("swept_volume_clear") is True
            and value.get("fixture_intersection_clear") is True
            and value.get("occlusion_validated") is False
            and isinstance(value.get("scale_to_meters"), (int, float))
            and not isinstance(value.get("scale_to_meters"), bool)
            and math.isfinite(float(value["scale_to_meters"]))
            and float(value["scale_to_meters"]) > 0.0
            and _valid_sha256(value.get("descriptor_sha256"))
            and _valid_sha256(value.get("license_notice_sha256"))
            and _valid_sha256(value.get("catalog_sha256"))
        )
        if (
            value.get("admitted") is not True or blockers
        ) and not pending_render_review:
            raise ValueError(f"RoboCasa asset {asset_id} is not admitted: {value.get('blockers')}")
        hashes = value.get("referenced_file_sha256")
        if not isinstance(hashes, Mapping) or not hashes:
            raise ValueError(f"RoboCasa asset {asset_id} has no file-hash manifest")
        if any(
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            for digest in hashes.values()
        ):
            raise ValueError(f"RoboCasa asset {asset_id} has malformed content hashes")
        if combined_manifest_hash(hashes) != value.get("manifest_sha256"):
            raise ValueError(f"RoboCasa asset {asset_id} manifest hash disagrees with its files")
    missing = sorted(set(required_asset_ids) - set(by_id))
    if missing:
        raise ValueError(f"selected RoboCasa assets lack admission rows: {missing}")


__all__ = [
    "AssetAdmission",
    "AxisAlignedBoundingBox",
    "RoboCasaBackgroundAdmission",
    "RoboCasaCatalogAsset",
    "RoboCasaCatalogPolicy",
    "inspect_asset_for_admission",
    "inspect_robocasa_background_asset",
    "load_robocasa_catalog_policy",
    "validate_robocasa_asset_manifest",
]
