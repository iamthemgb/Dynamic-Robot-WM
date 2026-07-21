"""Compile pinned external scenes into owned, calibrated MuJoCo models."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
import types
import xml.etree.ElementTree as ET
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from ...common.assets import RoboCasaCatalogAsset, load_robocasa_catalog_policy
from ...common.embodiments import FRANKA_HAND, ROBOTIQ_2F85_THICK_PAD
from ...scenarios.registry import load_scenario_definition
from .compiler import PhysicalSurface, SourceMujocoCompiledScenario
from .profiles import RIGID_REVIEW_PROFILE
from .provenance import (
    RoboCasaDependency,
    RollingIslandDependencyManifest,
    SourceDependencyManifest,
    referenced_asset_manifest,
)
from .rolling_island import (
    ROLLING_ISLAND_WRAPPER_NAME,
    install_rolling_island_scene,
)


_EXTERNAL_PACKAGE = "_dynamic_robot_dataset_source_mujoco_scene"
_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
_ROBOCASA_CATALOG = _REPOSITORY_ROOT / "configs/assets/robocasa_catalog_v1.yaml"

# Fixed, supported R1 review poses.  The Z values place each catalog model's
# measured bottom on the 0.74 m support surface used by the external R1 scene.
# X/Y stay well outside the owned task swept volume while remaining visible in
# the fixed review cameras.  These poses are part of the owned backend, not
# sampled by the external variants tree.
_ROBOCASA_CATALOG_POSES: Mapping[
    str, tuple[tuple[float, float, float], float]
] = {
    "lab": ((1.04, 0.78, 0.75566), 0.18),
    "kitchen": ((1.04, 0.78, 0.85807), -0.12),
    "workbench": ((1.04, 0.78, 0.89169), -0.16),
    "storage": ((1.04, 0.78, 0.77021), 0.14),
    "tabletop": ((1.04, 0.78, 0.77021), 0.08),
}

# Profile-level lifts measured from each compiled catalog candidate's true
# mesh AABB.  Applied only to owned P0 R1 support scenes, these put the lowest
# visual point at least 1 mm above the z=0.74 task surface.  F1 appearance
# poses remain unchanged.
_ROBOCASA_P0_CLEARANCE_LIFT_M: Mapping[str, float] = {
    "lab": 0.0030,
    "kitchen": 0.0032,
    "workbench": 0.0088,
    "storage": 0.0011,
    "tabletop": 0.0011,
}


@dataclass(frozen=True, slots=True)
class SourceModelIds:
    object_body: int
    object_geom: int
    object_qpos_adr: int
    object_qvel_adr: int
    robot_joint_ids: tuple[int, ...]
    robot_qpos_adrs: tuple[int, ...]
    robot_qvel_adrs: tuple[int, ...]
    actuator_ids: tuple[int, ...]
    left_gripper_geom_ids: tuple[int, ...]
    right_gripper_geom_ids: tuple[int, ...]
    left_gripper_body: int | None
    right_gripper_body: int | None
    hand_body: int | None
    grasp_site: int | None
    surface_geom_ids: Mapping[str, int]
    structural_support_geom_ids: tuple[int, ...]
    object_linked_equality_ids: tuple[int, ...]


@dataclass(slots=True)
class CompiledSourceModel:
    model: Any
    data: Any
    xml: str
    xml_sha256: str
    ids: SourceModelIds
    camera_names: Mapping[str, str]
    source_asset_sha256: Mapping[str, str]
    robocasa_assets: tuple[Mapping[str, Any], ...]
    stripped_robotwin_asset_count: int
    stripped_external_robocasa_element_count: int
    removed_visual_work_surface_names: tuple[str, ...]
    removed_task_volume_background_names: tuple[str, ...]
    removed_fixture_intersection_background_names: tuple[str, ...]
    relocated_visual_backgrounds: tuple[Mapping[str, Any], ...]
    background_geom_descriptors: tuple[Mapping[str, Any], ...]
    background_geom_exclusions: Mapping[str, str]
    robot_base_position_m: tuple[float, float, float] | None
    robot_base_quaternion_wxyz: tuple[float, float, float, float] | None
    external_camera_metadata: Mapping[str, Any]


def _is_floor_rooted_interception(corpus_leaf_id: str) -> bool:
    """Leaves whose task/base stays on the room floor in R0 and R1.

    All of F1, plus F2a since its v10 floor rooting: these are fixtureless
    ballistic interceptions, so the procedural central robot table is false
    appearance and the camera frame anchors at local height zero.  F2c and
    F2d joined with their owned floor-standing bounce pads and rebound
    walls: a counter-raised copy of either fixture re-creates the measured
    out-of-frame miss failures the F1/F2a rooting eliminated.
    """

    return corpus_leaf_id.startswith("F1") or corpus_leaf_id in {
        "F2a",
        "F2b",
        "F2c",
        "F2d",
        "F2e",
        "F2f",
    }


@contextmanager
def _temporary_environment(values: Mapping[str, str]) -> Iterator[None]:
    previous = {name: os.environ.get(name) for name in values}
    try:
        os.environ.update(values)
        yield
    finally:
        for name, old_value in previous.items():
            if old_value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old_value


def _load_external_scene_builder(source_root: Path) -> Any:
    """Load only the pinned scene package under a private module namespace."""

    package_dir = source_root / "scripts_mujoco"
    package = sys.modules.get(_EXTERNAL_PACKAGE)
    if package is None:
        package = types.ModuleType(_EXTERNAL_PACKAGE)
        package.__path__ = [str(package_dir)]  # type: ignore[attr-defined]
        package.__package__ = _EXTERNAL_PACKAGE
        sys.modules[_EXTERNAL_PACKAGE] = package
    module_name = f"{_EXTERNAL_PACKAGE}.scene_builder"
    existing = sys.modules.get(module_name)
    if existing is not None:
        existing_path = Path(existing.__file__).resolve()
        expected_path = (package_dir / "scene_builder.py").resolve()
        if existing_path != expected_path:
            raise RuntimeError("external scene-builder module cache points at another source root")
        return existing
    path = package_dir / "scene_builder.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load pinned scene builder: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def _surface_dict(value: PhysicalSurface) -> dict[str, Any]:
    value.validate()
    structural = value.role == "structural_support"
    return {
        "name": value.name,
        "type": "box",
        "pos": value.position_m,
        "size": value.half_size_m,
        "euler": value.euler_rad,
        "condim": 3,
        "friction": " ".join(f"{item:.9g}" for item in value.friction),
        "solref": " ".join(f"{item:.9g}" for item in value.solref),
        "solimp": "0.94 0.995 0.001",
        "priority": 2,
        "material": (
            "source_mujoco_structural_support"
            if structural
            else "source_mujoco_task_fixture"
        ),
        "material_rgba": (
            (0.12, 0.16, 0.20, 1.0)
            if structural
            else (0.46, 0.31, 0.18, 1.0)
        ),
        "roughness": 0.72 if structural else 0.62,
    }


def _build_external_sample(scene_builder: Any, scenario: SourceMujocoCompiledScenario) -> Any:
    scene_rng = np.random.default_rng(
        int(scenario.rng_subseeds["scene_construction"])
    )
    asset_rng = np.random.default_rng(int(scenario.rng_subseeds["assets"]))
    camera_rng = np.random.default_rng(int(scenario.rng_subseeds["camera"]))
    external_variant_seed = int(asset_rng.integers(0, 2**31, dtype=np.int64))
    sample = scene_builder.sample_episode(
        scene_rng,
        scenario.scene_variant,
        seed=external_variant_seed,
        fps=scenario.video_hz,
        duration=scenario.duration_s,
    )
    # The pinned builder historically drew camera and appearance values from
    # the same generator as construction.  Replace those fields explicitly so
    # counterfactual siblings can vary one declared RNG stream at a time.
    object_color = tuple(float(value) for value in asset_rng.uniform(0.15, 0.95, 3)) + (
        1.0,
    )
    lighting_intensity = float(asset_rng.uniform(0.85, 1.18))
    floor_material_jitter = float(asset_rng.uniform(-0.04, 0.04))
    camera_jitter = tuple(
        float(value)
        for value in camera_rng.normal(0.0, (0.035, 0.035, 0.025))
    )
    # Floor-rooted interception scenes omit the procedural center table.
    # F2c/F2d still own physical bounce/barrier fixtures, but those fixtures
    # are explicitly supported from the floor instead of floating on a
    # generic worktop.
    free_space_f1 = _is_floor_rooted_interception(scenario.corpus_leaf_id)
    tabletop_height = (
        None
        if free_space_f1
        else 0.74
        if scenario.requires_real_robocasa
        else None
    )
    task_surfaces = tuple(
        surface for surface in scenario.surfaces if surface.expected_task_contact
    )
    contact_groups = tuple(
        {"name": surface.role, "geoms": (surface.name,)}
        for surface in task_surfaces
    )
    return replace(
        sample,
        timestep=1.0 / scenario.simulation_hz,
        duration_s=scenario.duration_s,
        fps=scenario.video_hz,
        seed=external_variant_seed,
        gravity=scenario.gravity_m_s2,
        ball_radius=scenario.object_radius_m,
        ball_mass=scenario.object_mass_kg,
        ball_color=object_color,
        ball_initial_position=scenario.object_initial_position_m,
        ball_initial_velocity=scenario.object_initial_linear_velocity_m_s,
        lighting_intensity=lighting_intensity,
        camera_jitter=camera_jitter,
        floor_material_jitter=floor_material_jitter,
        release_time_s=0.0,
        robot_base_position=(scenario.robot_base_position_m or (0.0, 0.0, 0.0)),
        robot_base_euler=(scenario.robot_base_euler_rad or (0.0, 0.0, 0.0)),
        catch_center_z=(scenario.controller_target_position_m or (0.0, 0.0, 0.5))[2],
        tabletop_height=tabletop_height,
        planned_intercept_time_s=scenario.ballistic_event_time_s,
        planned_intercept_position=scenario.controller_target_position_m,
        interception_subfamily=scenario.subfamily,
        expected_contact_sequence=tuple(surface.role for surface in task_surfaces),
        interception_surface_specs=tuple(_surface_dict(surface) for surface in scenario.surfaces),
        surface_contact_groups=contact_groups,
        scene_randomization_level=("balanced" if scenario.requires_real_robocasa else "clean"),
        offscreen_width=RIGID_REVIEW_PROFILE.width,
        offscreen_height=RIGID_REVIEW_PROFILE.height,
    )


def _remove_children_with_prefix(root: ET.Element, tag: str, prefix: str) -> int:
    removed = 0
    for parent in root.iter():
        for child in list(parent):
            if child.tag == tag and str(child.get("name") or "").startswith(prefix):
                parent.remove(child)
                removed += 1
    return removed


def _strip_robotwin(root: ET.Element) -> int:
    removed = _remove_children_with_prefix(root, "body", "rw_")
    removed += _remove_children_with_prefix(root, "mesh", "rw_")
    return removed


def _strip_external_robocasa_selection(root: ET.Element) -> int:
    """Remove every RoboCasa model sampled by the external variants tree."""

    removed = 0
    for tag in ("body", "mesh", "texture", "material"):
        removed += _remove_children_with_prefix(root, tag, "rc_")
    return removed


def _remove_external_visual_work_surfaces(
    root: ET.Element,
    scenario: SourceMujocoCompiledScenario,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Remove procedural worktops that intersect owned physical fixtures.

    The pinned scene builder puts ``robot_table_top`` and
    ``back_counter_top`` at a top height of exactly 0.74 m.  Every fixed R1
    P0 fixture is also rendered at or through that height.  Merely disabling
    contact therefore leaves two coplanar (or crossing) visible surfaces and
    produces large, disconnected depth-fighting polygons in both cameras.

    Fixture-free interception scenes intentionally keep their randomized
    procedural worktop.  When an owned physical fixture is present, however,
    it is the sole visible task surface; the lower table/counter structure
    remains as visual support.  Fail closed if the expected primary worktop
    is absent, or if a future external variant adds another top-level
    ``countertop`` work-surface that this owned cleanup did not classify.
    """

    if not scenario.requires_real_robocasa:
        return (), ()
    world = root.find("worldbody")
    if world is None:
        raise RuntimeError("external scene lacks worldbody")
    if _is_floor_rooted_interception(scenario.corpus_leaf_id):
        # The F1/F2a task/base stays on the room floor in both R0 and R1.
        # Every central support-table component is therefore false
        # appearance: the object may legitimately miss and fall through its
        # former volume.
        # Remove the entire named family, not only its collision-disabled top,
        # and bind the exact variant-dependent set into runtime metadata.
        candidates = tuple(
            geom
            for geom in world.findall("./geom")
            if str(geom.get("name") or "").startswith("robot_table_")
        )
        names = tuple(sorted(str(geom.get("name") or "") for geom in candidates))
        if "robot_table_top" not in names:
            raise RuntimeError("R1 F1 scene lacks its classified central robot table")
        for geom in candidates:
            world.remove(geom)
        if scenario.scene_profile == "robocasa_storage":
            marker = world.find("./geom[@name='storage_floor_marker']")
            if marker is None:
                raise RuntimeError(
                    "R1 F1 storage scene lacks its classified floor marker"
                )
            world.remove(marker)
            names = tuple(sorted((*names, "storage_floor_marker")))
        if (
            scenario.corpus_leaf_id == "F2f"
            and scenario.scene_profile == "robocasa_kitchen"
        ):
            # The yawed-barrier negative branch legitimately rolls to the
            # back of the room after its miss.  The procedural cabinet bulk
            # occupied that floor sweep even though its collision was off.
            # Remove only the intersecting visual bulk; the cabinet fronts,
            # toe kick, counter, backsplash, and licensed RoboCasa asset stay
            # in place, preserving an unmistakable kitchen background.
            cabinet_bulk = world.find("./geom[@name='back_counter_base']")
            if cabinet_bulk is None:
                raise RuntimeError(
                    "R1 F2f kitchen lacks its classified back-counter bulk"
                )
            world.remove(cabinet_bulk)
            names = tuple(sorted((*names, "back_counter_base")))
        remaining = tuple(
            str(element.get("name") or "")
            for element in root.iter()
            if str(element.get("name") or "").startswith("robot_table_")
        )
        if remaining:
            raise RuntimeError(
                "unclassified central robot-table element remains in R1 F1 scene: "
                + ", ".join(sorted(remaining))
            )
        return (), names

    # The overlap replacement below is the P0 passive-fixture contract.  F2
    # scenes may declare a contact plate or vertical rebound wall while the
    # procedural table still physically supports the robot; those fixtures
    # are not replacements for the worktop.
    if not scenario.corpus_leaf_id.startswith("P0") or not scenario.surfaces:
        return (), ()
    candidates = tuple(
        geom
        for geom in world.findall("./geom")
        if str(geom.get("material") or "") == "countertop"
        and str(geom.get("name") or "").endswith("_top")
    )
    names = tuple(sorted(str(geom.get("name") or "") for geom in candidates))
    if "robot_table_top" not in names:
        raise RuntimeError(
            "R1 physical-fixture scene lacks its classified procedural worktop"
        )
    for geom in candidates:
        world.remove(geom)
    remaining = tuple(
        str(geom.get("name") or "")
        for geom in world.findall("./geom")
        if str(geom.get("material") or "") == "countertop"
        and str(geom.get("name") or "").endswith("_top")
    )
    if remaining:
        raise RuntimeError(
            "unclassified procedural worktop remains in an R1 fixture scene: "
            + ", ".join(sorted(remaining))
        )
    return names, ()


def _relocate_external_visual_boundaries(
    root: ET.Element,
    scenario: SourceMujocoCompiledScenario,
) -> tuple[Mapping[str, Any], ...]:
    """Move visual-only left room boundaries outside every owned P0 fixture.

    The external room used x≈-0.95 for both ``left_wall`` and the kitchen
    ``backsplash_left_return`` while owned P0 supports extend as far as
    x=-1.40.  Keeping them in place creates a visible intersection even after
    contact is disabled.  Relocation preserves task physics and fixed seeds;
    the exact pre/post transforms are persisted for review.
    """

    if (
        not scenario.requires_real_robocasa
        or not scenario.corpus_leaf_id.startswith("P0")
        or not scenario.surfaces
    ):
        return ()
    world = root.find("worldbody")
    if world is None:
        raise RuntimeError("external scene lacks worldbody")
    left_wall = world.find("./geom[@name='left_wall']")
    if left_wall is None:
        raise RuntimeError("R1 P0 scene lacks its classified left visual boundary")
    result: list[Mapping[str, Any]] = []
    # The far face is at most -1.46 m, leaving 6 cm beyond the largest current
    # owned support bound (-1.40 m).  Keep Y/Z and geometry size unchanged.
    maximum_x = -1.46
    for name in ("left_wall", "backsplash_left_return"):
        geom = world.find(f"./geom[@name='{name}']")
        if geom is None:
            continue
        old = tuple(float(value) for value in str(geom.get("pos") or "").split())
        size = tuple(float(value) for value in str(geom.get("size") or "").split())
        if len(old) != 3 or len(size) < 1 or size[0] <= 0:
            raise RuntimeError(f"visual boundary {name} lacks a finite box transform")
        new = (maximum_x - size[0], old[1], old[2])
        geom.set("pos", " ".join(f"{value:.12g}" for value in new))
        result.append(
            {
                "name": name,
                "old_position_m": old,
                "new_position_m": new,
                "maximum_x_m": maximum_x,
                "reason": "clear_owned_p0_fixture_aabb",
            }
        )
    return tuple(result)


def _apply_owned_robot_base(
    root: ET.Element,
    scenario: SourceMujocoCompiledScenario,
) -> None:
    """Replace the external sample's implicit base transform with the contract."""

    if scenario.embodiment == "no_robot":
        return
    assert scenario.robot_base_position_m is not None
    assert scenario.robot_base_euler_rad is not None
    link0 = root.find("./worldbody/body[@name='link0']")
    if link0 is None:
        raise RuntimeError("external robot scene lacks link0")
    link0.set(
        "pos", " ".join(f"{value:.12g}" for value in scenario.robot_base_position_m)
    )
    link0.attrib.pop("quat", None)
    link0.attrib.pop("axisangle", None)
    link0.attrib.pop("xyaxes", None)
    link0.attrib.pop("zaxis", None)
    link0.set(
        "euler", " ".join(f"{value:.12g}" for value in scenario.robot_base_euler_rad)
    )


def _loaded_external_variants(source_root: Path) -> Any:
    """Return the already-loaded, hash-pinned private variants module."""

    module_name = f"{_EXTERNAL_PACKAGE}.variants"
    module = sys.modules.get(module_name)
    if module is None:
        raise RuntimeError("external scene builder did not load its pinned variants module")
    expected_path = (source_root / "scripts_mujoco/variants.py").resolve(strict=True)
    module_path = Path(module.__file__).resolve(strict=True)
    if module_path != expected_path:
        raise RuntimeError("external variants module cache points at another source root")
    importer = getattr(module, "append_robocasa_visual_model", None)
    if not callable(importer):
        raise RuntimeError("pinned variants module lacks its visual-only RoboCasa importer")
    return module


def _select_catalog_candidate(
    scenario: SourceMujocoCompiledScenario,
    dependency: RoboCasaDependency,
) -> tuple[RoboCasaCatalogAsset, str, tuple[float, float, float], float, Mapping[str, Any]]:
    """Resolve exactly one content-bound candidate for the fixed R1 profile."""

    catalog_profile = scenario.scene_profile.removeprefix("robocasa_")
    policy = load_robocasa_catalog_policy(
        _ROBOCASA_CATALOG,
        source_root_override=dependency.source_root,
    )
    if Path(policy.source_root).resolve(strict=True) != Path(
        dependency.source_root
    ).resolve(strict=True):
        raise RuntimeError("RoboCasa catalog resolved a different dependency root")
    if Path(policy.asset_root).resolve(strict=True) != Path(
        dependency.asset_root
    ).resolve(strict=True):
        raise RuntimeError("RoboCasa catalog resolved a different asset root")
    if policy.license_notice_sha256 != dependency.license_sha256:
        raise RuntimeError("RoboCasa catalog license binding differs from the dependency")
    candidates = tuple(
        asset for asset in policy.assets if catalog_profile in asset.profiles
    )
    if len(candidates) != 1:
        raise RuntimeError(
            f"R1 profile {catalog_profile!r} requires exactly one catalog candidate"
        )
    candidate = candidates[0]
    if (
        not candidate.review_candidate
        or not candidate.visual_only
        or candidate.collision_enabled
    ):
        raise RuntimeError(
            f"RoboCasa catalog candidate is not safe for review: {candidate.asset_id}"
        )
    try:
        original_position, yaw = _ROBOCASA_CATALOG_POSES[catalog_profile]
    except KeyError as error:
        raise RuntimeError(
            f"R1 profile lacks an owned supported background pose: {catalog_profile}"
        ) from error
    clearance_lift_m = (
        _ROBOCASA_P0_CLEARANCE_LIFT_M[catalog_profile]
        if scenario.corpus_leaf_id.startswith("P0")
        else 0.0
    )
    if scenario.rolling_island_scene is not None:
        # The old catalog pose was calibrated for a 0.74 m procedural table
        # and can intersect the much larger real island.  Put the visual-only
        # review accessory on the room floor in a remote task-frame corner;
        # the full RoboCasa layout remains the visible scene context.
        position = (1.55, 1.35, original_position[2] - 0.74)
    else:
        position = (
            original_position[0],
            original_position[1],
            original_position[2] + clearance_lift_m,
        )
    descriptor = Path(candidate.descriptor_path).resolve(strict=True)
    asset_root = Path(dependency.asset_root).resolve(strict=True)
    try:
        relative_descriptor = descriptor.relative_to(asset_root).as_posix()
    except ValueError as error:
        raise RuntimeError("RoboCasa catalog descriptor escapes the dependency asset root") from error
    catalog_metadata = {
        "schema_version": policy.schema_version,
        "catalog_version": policy.catalog_version,
        "catalog_sha256": policy.catalog_sha256,
        "catalog_release_ready": policy.release_ready,
        "catalog_asset_id": candidate.asset_id,
        "descriptor_sha256": candidate.descriptor_sha256,
        "referenced_file_sha256": dict(candidate.referenced_file_sha256),
        "manifest_sha256": candidate.manifest_sha256,
        "license_notice_sha256": candidate.license_notice_sha256,
        "scale_to_meters": candidate.scale_to_meters,
        "visual_only": candidate.visual_only,
        "collision_enabled": candidate.collision_enabled,
        "static_admission_blockers": list(candidate.blockers),
        "original_position_m": list(original_position),
        "p0_fixture_clearance_lift_m": clearance_lift_m,
        "position_transform_source": "owned_profile_level_p0_clearance/v1",
    }
    return candidate, relative_descriptor, position, yaw, catalog_metadata


def _inject_owned_robocasa_catalog_candidate(
    root: ET.Element,
    *,
    scenario: SourceMujocoCompiledScenario,
    dependency: RoboCasaDependency,
    source_root: Path,
) -> tuple[tuple[Mapping[str, Any], ...], int]:
    """Replace external random imports with the profile's catalog candidate."""

    removed = _strip_external_robocasa_selection(root)
    candidate, relative, position, yaw, catalog_metadata = _select_catalog_candidate(
        scenario, dependency
    )
    variants = _loaded_external_variants(source_root)
    catalog_profile = scenario.scene_profile.removeprefix("robocasa_")
    slot = f"catalog_{catalog_profile}"
    instance = f"rc_{slot}"
    environment = {
        "ROBOCASA_ROOT": dependency.source_root,
        "ROBOCASA_ASSETS_ROOT": dependency.asset_root,
    }
    with _temporary_environment(environment):
        imported = variants.append_robocasa_visual_model(
            root,
            relative,
            instance=instance,
            pos=position,
            euler=(0.0, 0.0, yaw),
        )
    if imported is not True:
        raise RuntimeError(
            f"failed to import RoboCasa catalog candidate: {candidate.asset_id}"
        )
    prefix = f"{instance}_"
    named_rc_elements = tuple(
        element
        for element in root.iter()
        if str(element.get("name") or "").startswith("rc_")
    )
    if not named_rc_elements or any(
        not str(element.get("name") or "").startswith(prefix)
        for element in named_rc_elements
    ):
        raise RuntimeError("arbitrary externally sampled RoboCasa elements remain")
    metadata = {
        "slot": slot,
        "model_xml": relative,
        "position": [float(value) for value in position],
        "yaw": float(yaw),
        "imported": True,
        "selection_source": "owned_content_bound_catalog",
        "supported_surface_height_m": 0.74,
        "external_random_elements_removed": removed,
        **catalog_metadata,
    }
    return (metadata,), removed


def _runtime_camera_metadata(
    camera_pose: Mapping[str, Any],
    runtime_assets: Sequence[Mapping[str, Any]],
    *,
    removed_count: int,
    removed_visual_work_surface_names: Sequence[str],
    removed_task_volume_background_names: Sequence[str],
    removed_fixture_intersection_background_names: Sequence[str],
    relocated_visual_backgrounds: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Replace stale external selection metadata with the executed selection."""

    result = dict(camera_pose)
    main = dict(result.get("main_camera") or {})
    serialized_assets = [dict(value) for value in runtime_assets]
    main["robocasa_background_assets"] = serialized_assets
    main["robocasa_background_selection_source"] = "owned_content_bound_catalog"
    main["external_random_robocasa_elements_removed"] = int(removed_count)
    main["removed_visual_work_surface_names"] = list(
        removed_visual_work_surface_names
    )
    main["removed_task_volume_background_names"] = list(
        removed_task_volume_background_names
    )
    main["removed_fixture_intersection_background_names"] = list(
        removed_fixture_intersection_background_names
    )
    main["relocated_visual_backgrounds"] = [
        dict(value) for value in relocated_visual_backgrounds
    ]
    randomization = main.get("scene_randomization")
    if isinstance(randomization, Mapping):
        randomization = dict(randomization)
        randomization["robocasa_background_assets"] = serialized_assets
        randomization["robocasa_background_selection_source"] = (
            "owned_content_bound_catalog"
        )
        main["scene_randomization"] = randomization
    result["main_camera"] = main
    return result


def _remove_robot(root: ET.Element) -> None:
    world = root.find("worldbody")
    if world is None:
        raise RuntimeError("external scene lacks worldbody")
    link0 = world.find("./body[@name='link0']")
    if link0 is None:
        raise RuntimeError("external Panda scene lacks link0")
    world.remove(link0)
    for section_name in ("actuator", "tendon", "equality", "contact"):
        section = root.find(section_name)
        if section is not None:
            root.remove(section)


def _add_secondary_camera(
    root: ET.Element,
    scenario: SourceMujocoCompiledScenario,
    height_offset: float,
) -> None:
    """Install the owned task-overview view before model initialization.

    The external camera was composed around the interception point.  That is
    useful as the primary review view, but it cannot show the complete outcome
    of a miss that reaches the room floor, a high projectile apex, or the end
    of a long rolling surface.  Keep the primary view untouched and give only
    the secondary stream a deterministic task envelope.  Camera changes do
    not alter task geometry, physics, RNG state, or runtime callbacks.
    """

    world = root.find("worldbody")
    if world is None:
        raise RuntimeError("external scene lacks worldbody")
    old = world.find("./camera[@name='secondary_camera']")
    if old is not None:
        world.remove(old)
    camera = ET.SubElement(
        world,
        "camera",
        name="secondary_camera",
    )
    if scenario.motion_kind in {
        "wall_rebound_interception",
        "arbitrary_surface_rebound_interception",
    }:
        anchor = scenario.physical_target_position_m
        if anchor is None:
            raise RuntimeError("interception camera lacks its physical target")
        # The rebound wall stands behind the workspace at +X, so the default
        # close right-side secondary would sit inside or behind it (in R1 it
        # measurably framed the inside of a cabinet).  A -Y flank view keeps
        # the launch, the tall wall face, and the catch in frame.
        _set_camera_look_at(
            camera,
            position_m=(anchor[0] - 0.10, anchor[1] - 1.60, anchor[2] + 0.60),
            target_m=(anchor[0] + 0.05, anchor[1], anchor[2] + 0.45),
            fovy_deg=62.0,
        )
    elif scenario.motion_kind in {
        "direct_free_contact_interception",
        "rolling_pickup_interception",
        "bounce_apex_pickup_interception",
        "table_bounce_apex_pickup_interception",
        "ramp_launch_pickup_interception",
        "ramp_launch_interception",
        "ordered_multi_rebound_pickup_interception",
        "random_plane_bounce_pickup_interception",
        "arbitrary_surface_rebound_interception",
    }:
        anchor = scenario.physical_target_position_m
        if anchor is None:
            raise RuntimeError("interception camera lacks its physical target")
        if (
            scenario.scene_profile == "robocasa_storage"
            and scenario.motion_kind == "direct_free_contact_interception"
        ):
            # A high deterministic overview covers the complete storage F1
            # envelope: release near z=1.50, free-contact interaction near the
            # gripper, and collision-derived failures reaching y=-1.61 on the
            # room floor.  The close view lost the last third of three fixed
            # storage/Panda trajectories.  Restrict this overview to storage:
            # other RoboCasa profiles contain tall appearance geometry that
            # can occlude an overhead camera.
            _set_camera_look_at(
                camera,
                position_m=(0.30, -1.20, 2.80),
                target_m=(0.90, -0.90, 0.65),
                fovy_deg=72.0,
            )
        elif scenario.motion_kind == "rolling_pickup_interception":
            # The folded arm occludes whichever rolling lane lies behind it:
            # with both views on the -Y side the declared +0.135 m lateral
            # negatives measured 60%/67% per-view visible fractions, while a
            # +Y secondary lost the arm-shifted timing negatives instead.
            # The secondary therefore sits on the side of the declared lane
            # away from the commanded arm, derived from the same fixed
            # intervention the scenario declares.
            lane_offset_y = float(scenario.object_initial_position_m[1]) - float(
                scenario.controller_target_position_m[1]
            )
            secondary_y = 0.55 if lane_offset_y > 0.025 else -0.35
            _set_camera_look_at(
                camera,
                position_m=(anchor[0] + 1.05, anchor[1] + secondary_y, anchor[2] + 0.45),
                target_m=(anchor[0], anchor[1], anchor[2] + 0.05),
                fovy_deg=55.0,
            )
        else:
            # The close right-side angle resolves the key-event target and
            # both opposed finger pads when the main view looks nearly along
            # their axis.
            _set_camera_look_at(
                camera,
                position_m=(anchor[0] + 1.05, anchor[1] - 0.35, anchor[2] + 0.45),
                target_m=(anchor[0], anchor[1], anchor[2] + 0.05),
                fovy_deg=55.0,
            )
    elif scenario.motion_kind == "passive_projectile":
        # P0b-review-02 reaches 1.395 m above its support and exceeded both old
        # views near the apex.  Express this overview relative to the support
        # height so the clean and RoboCasa scenes share identical framing.
        _set_camera_look_at(
            camera,
            position_m=(0.20, -1.80, height_offset + 1.20),
            target_m=(-0.15, -0.05, height_offset + 0.72),
            fovy_deg=62.0,
        )
    elif scenario.motion_kind in {
        "passive_slope_roll",
        "passive_straight_roll",
    }:
        # The P0d object stays on its declared support while traversing as far
        # as x=[-0.89, 1.07].  Aim at the physical support plane, not the old
        # interception-height target, to retain both endpoints.
        _set_camera_look_at(
            camera,
            position_m=(0.10, -1.80, height_offset + 0.90),
            target_m=(0.10, 0.0, height_offset + 0.04),
            fovy_deg=58.0,
        )
    elif scenario.motion_kind == "passive_wall_rebound":
        # A wall-normal complementary view makes the separated post-contact
        # trajectory legible instead of compressing it into the wall plane.
        # Keep this preset wall-only: the table-rebound secondary serialization
        # is already accepted and must not change as a side effect.
        _set_camera_look_at(
            camera,
            position_m=(-0.05, -1.70, height_offset + 0.76),
            target_m=(-0.05, 0.0, height_offset + 0.76),
            fovy_deg=58.0,
        )
    else:
        # Preserve the already-reviewed complementary view for other tasks,
        # including P0c table rebounds.
        camera.set("pos", f"-0.72 -1.18 {1.10 + height_offset:.8g}")
        camera.set("xyaxes", "0.853  -0.522 0 0.239 0.391 0.889")
        camera.set("fovy", "52")


def _set_camera_look_at(
    camera: ET.Element,
    *,
    position_m: tuple[float, float, float],
    target_m: tuple[float, float, float],
    fovy_deg: float,
) -> None:
    """Set a fixed MuJoCo camera from a world-space position and target."""

    position = np.asarray(position_m, dtype=np.float64)
    target = np.asarray(target_m, dtype=np.float64)
    direction = target - position
    direction /= np.linalg.norm(direction)
    camera_z = -direction
    camera_x = np.cross(np.asarray((0.0, 0.0, 1.0)), camera_z)
    camera_x /= np.linalg.norm(camera_x)
    camera_y = np.cross(camera_z, camera_x)
    camera.set("pos", " ".join(f"{value:.9g}" for value in position))
    camera.set(
        "xyaxes",
        " ".join(f"{value:.9g}" for value in (*camera_x, *camera_y)),
    )
    camera.set("fovy", f"{fovy_deg:.9g}")


def _repair_task_camera(
    root: ET.Element,
    scenario: SourceMujocoCompiledScenario,
    height_offset: float,
) -> None:
    """Keep owned physical fixtures from hiding the task in the main view."""

    if scenario.rolling_island_scene is not None:
        # The adapter installs Michael's arm-relative island camera in the
        # normalized task frame.  It is the requested visual reference for
        # F3b and must not be replaced by the former miniature-runway view.
        return

    if scenario.motion_kind not in {
        "direct_free_contact_interception",
        "rolling_pickup_interception",
        "wall_rebound_interception",
        "bounce_apex_pickup_interception",
        "table_bounce_apex_pickup_interception",
        "ramp_launch_pickup_interception",
        "ramp_launch_interception",
        "ordered_multi_rebound_pickup_interception",
        "random_plane_bounce_pickup_interception",
        "arbitrary_surface_rebound_interception",
        "passive_projectile",
        "passive_wall_rebound",
    }:
        return
    main = root.find(".//camera[@name='main_camera']")
    if main is None:
        raise RuntimeError("external scene lacks main_camera")
    if scenario.motion_kind in {
        "direct_free_contact_interception",
        "rolling_pickup_interception",
        "wall_rebound_interception",
        "bounce_apex_pickup_interception",
        "table_bounce_apex_pickup_interception",
        "ramp_launch_pickup_interception",
        "ramp_launch_interception",
        "ordered_multi_rebound_pickup_interception",
        "random_plane_bounce_pickup_interception",
        "arbitrary_surface_rebound_interception",
    }:
        anchor = scenario.physical_target_position_m or scenario.controller_target_position_m
        if anchor is None:
            raise RuntimeError("interception camera lacks its physical/controller target")
        _set_camera_look_at(
            main,
            position_m=(anchor[0] - 1.15, anchor[1] - 1.18, anchor[2] + 0.65),
            target_m=(anchor[0], anchor[1], anchor[2] + 0.17),
            fovy_deg=62.0,
        )
        return
    if scenario.motion_kind == "passive_projectile":
        # The external main camera clips the higher-speed fixed projectile at
        # its measured apex.  This deterministic task-envelope view preserves
        # at least an eight-pixel projected-radius margin for every canonical
        # 30 Hz frame in all six P0b review trajectories while retaining a
        # distinct azimuth from the owned secondary overview.
        _set_camera_look_at(
            main,
            position_m=(1.20, -1.15, height_offset + 1.15),
            target_m=(0.05, -0.10, height_offset + 0.85),
            fovy_deg=58.0,
        )
        return
    # The external main camera sits on the far side of supported_wall, making
    # the rebound invisible.  Keep the repaired camera on the incoming side,
    # but frame the complete fixed trajectory rather than only the contact
    # event.  The 64-degree vertical field of view and balanced look-at keep
    # the projected object (including its radius) at least eight pixels inside
    # all four image edges for P0c-review-01/03/05 at every 30 Hz timestamp.
    # This is a deterministic pre-initialization camera calibration: no seed,
    # physics state, fixture, or runtime callback is changed.
    _set_camera_look_at(
        main,
        position_m=(-0.90, 0.95, height_offset + 1.05),
        target_m=(-0.02, 0.0, height_offset + 0.84),
        fovy_deg=64.0,
    )


def _restore_collision_geometries(root: ET.Element) -> None:
    """Undo the source demo's broad robot-collision suppression."""

    for geom in root.findall(".//geom"):
        class_name = str(geom.get("class") or "")
        name = str(geom.get("name") or "")
        if "collision" in class_name and not class_name.endswith("/visual"):
            if geom.get("contype") == "0":
                geom.attrib.pop("contype", None)
            if geom.get("conaffinity") == "0":
                geom.attrib.pop("conaffinity", None)
        if name in {
            "rq_left_pad1",
            "rq_left_pad2",
            "rq_right_pad1",
            "rq_right_pad2",
        }:
            geom.attrib.pop("contype", None)
            geom.attrib.pop("conaffinity", None)


def _patch_calibrated_model(
    root: ET.Element,
    scenario: SourceMujocoCompiledScenario,
) -> None:
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("timestep", f"{1.0 / scenario.simulation_hz:.12g}")
    option.set("gravity", " ".join(f"{value:.12g}" for value in scenario.gravity_m_s2))
    option.set("integrator", "implicitfast")
    option.set("solver", "Newton")
    option.set("iterations", "100")
    option.set("cone", "elliptic")
    option.set("impratio", "3")

    ball = root.find(".//geom[@name='catch_ball_geom']")
    if ball is None:
        raise RuntimeError("external scene lacks the free task-object geom")
    ball.set("condim", "3")
    ball.set("friction", "0.9 0.005 0.0001")
    ball.set("solref", "0.012 0.7")
    ball.set("solimp", "0.94 0.995 0.001")

    world = root.find("worldbody")
    if world is None:
        raise RuntimeError("external scene lacks worldbody")
    declared_surfaces = {surface.name for surface in scenario.surfaces}
    clean_visual_ground_center_z = None
    if (
        not scenario.requires_real_robocasa
        and scenario.corpus_leaf_id.startswith("P0")
        and any(
            surface.role in {"floor", "table", "slope"}
            for surface in scenario.surfaces
        )
    ):
        # The external clean scene's room floor is coplanar with the owned P0
        # task surface.  Removing it leaves large black voids outside narrow
        # rolling/bounce fixtures.  Replace it with a thin, collision-disabled
        # visual box whose top is 1 mm below the lowest owned fixture.  It is
        # therefore visible but neither coplanar nor part of task contact.
        lowest_fixture_bottom = min(
            surface.position_m[2] - surface.half_size_m[2]
            for surface in scenario.surfaces
            if surface.role in {"floor", "table", "slope"}
        )
        clean_visual_ground_center_z = lowest_fixture_bottom - 0.006
    # Scene furniture/walls are appearance context, not undeclared task
    # fixtures.  Only the room floor (when distinct from the task plane) and
    # explicitly declared owned surfaces retain collision.
    for geom in list(world.findall("./geom")):
        name = str(geom.get("name") or "")
        if name == "floor" and clean_visual_ground_center_z is not None:
            geom.set("name", "clean_visual_ground")
            geom.set("type", "box")
            geom.set("size", "3 3 0.005")
            geom.set("pos", f"0 0 {clean_visual_ground_center_z:.9g}")
            geom.set("contype", "0")
            geom.set("conaffinity", "0")
            geom.set("condim", "3")
        elif name not in declared_surfaces and name != "floor":
            geom.set("contype", "0")
            geom.set("conaffinity", "0")
        elif name == "floor":
            # The room floor is a real anchored support, not appearance-only
            # furniture.  Give it the hard calibrated support profile so a
            # missed object cannot tunnel one radius into the plane.
            geom.set("contype", "1")
            geom.set("conaffinity", "1")
            geom.set("condim", "3")
            geom.set("friction", "0.9 0.005 0.0001")
            geom.set("solref", "0.003 1")
            geom.set("solimp", "0.94 0.995 0.001")
            geom.set("priority", "2")
            # A failed interception reaches the room support at substantially
            # higher speed than the passive calibration drops.  Four
            # millimetres of predictive contact margin keeps the geometric
            # depth below the 3 mm hard limit; this does not move the object or
            # create a hidden fixture.
            geom.set("margin", "0.004")

    if scenario.rolling_island_scene is not None:
        floor = root.find(".//geom[@name='floor']")
        if floor is None:
            raise RuntimeError("rolling-island scene lacks its physical room floor")
        floor.set("contype", "1")
        floor.set("conaffinity", "1")
        floor.set("condim", "3")
        floor.set("friction", "0.9 0.005 0.0001")
        floor.set("solref", "0.003 1")
        floor.set("solimp", "0.94 0.995 0.001")
        floor.set("priority", "2")
        floor.set("margin", "0.004")

    for surface in scenario.surfaces:
        geom = root.find(f".//geom[@name='{surface.name}']")
        if geom is None:
            raise RuntimeError(f"compiled scene omitted physical fixture {surface.name}")
        geom.set("contype", "1")
        geom.set("conaffinity", "1")
        geom.set("condim", "3")
        geom.set("friction", " ".join(f"{value:.9g}" for value in surface.friction))
        geom.set("solref", " ".join(f"{value:.9g}" for value in surface.solref))
        if surface.contact_profile == "rebound_wall":
            # Same calibrated mixed wall pair as P0c, but with the 4 mm
            # predictive margin the room floor and hand shells use: the F2d
            # arc measurably recorded a 3.51 mm geometric wall penetration
            # under the 2 mm margin, marginally past the 3 mm gate.
            geom.set(
                "margin",
                f"{RIGID_REVIEW_PROFILE.wall_rebound_contact_margin_m:.9g}",
            )
            continue
        if surface.contact_profile == "rebound_pad":
            # The calibrated bounce pad owns its full contact response like
            # the room floor and hand shells: mixed-pair plates measurably
            # either exceed the 3 mm geometric-penetration gate at the fixed
            # construction speed or inject energy through an underdamped
            # margin catapult.
            geom.set("solimp", "0.94 0.995 0.001")
            geom.set("priority", "2")
            geom.set(
                "margin",
                f"{RIGID_REVIEW_PROFILE.bounce_pad_contact_margin_m:.9g}",
            )
            continue
        if surface.role == "ramp":
            # F2b initializes at geometric rolling contact.  A predictive
            # margin would interpret that valid state as 2 mm of soft contact
            # compression and catapult the ball off the ramp for ~47 ms,
            # splitting one support interval into two artificial groups.
            # The ramp-normal speed is small and the reference 1200 Hz step
            # resolves it without a speculative margin.
            geom.set("margin", "0")
            continue
        # At 600 Hz a fast sphere can travel several millimetres per step.
        # A 2 mm predictive contact margin keeps geometric penetration within
        # the 3 mm hard limit without changing or scripting object motion.
        geom.set("margin", "0.002")

    if scenario.embodiment != "no_robot":
        _restore_collision_geometries(root)
        # A missed rolling ball is allowed to strike the real robot pedestal,
        # but the contact must still obey the strict 2 mm robot-penetration
        # gate.  Calibrate the restored link0 collision shell just like the
        # hand shells instead of suppressing this physically meaningful
        # contact or rounding a threshold violation away.
        link0 = root.find(".//body[@name='link0']")
        if link0 is None:
            raise RuntimeError("robot pedestal body is unavailable: link0")
        for geom in link0.findall("./geom"):
            if "collision" not in str(geom.get("class") or ""):
                continue
            geom.set("condim", "3")
            geom.set("friction", "0.9 0.005 0.0001")
            geom.set("solref", "0.003 1")
            geom.set("solimp", "0.94 0.995 0.001")
            geom.set("priority", "2")
            geom.set("margin", "0.004")
    if scenario.embodiment == FRANKA_HAND:
        for body_name in ("hand", "left_finger", "right_finger"):
            body = root.find(f".//body[@name='{body_name}']")
            if body is None:
                raise RuntimeError(f"Panda collision body is unavailable: {body_name}")
            for geom in body.findall(".//geom"):
                if geom.get("contype", "1") == "0":
                    continue
                geom.set("condim", "3")
                geom.set("friction", "0.9 0.005 0.0001")
                geom.set("solref", "0.003 1")
                geom.set("solimp", "0.94 0.995 0.001")
                geom.set("priority", "2")
                # A deflected interception can clip the rigid hand shell at
                # full ballistic speed.  Like the room floor's failed-catch
                # profile, four millimetres of predictive contact margin keeps
                # the geometric depth of that corner graze below the 2 mm
                # gripper admission limit without moving or scripting the
                # object.
                geom.set("margin", "0.004")
    if scenario.embodiment == ROBOTIQ_2F85_THICK_PAD:
        controller_plan = load_scenario_definition(
            scenario.corpus_leaf_id
        ).module.controller_plan
        for name in (
            "rq_left_pad_thick_collision_pad",
            "rq_right_pad_thick_collision_pad",
        ):
            pad = root.find(f".//geom[@name='{name}']")
            if pad is None:
                raise RuntimeError(f"Robotiq calibrated pad is unavailable: {name}")
            pad_size = [float(value) for value in str(pad.get("size") or "").split()]
            if len(pad_size) != 3:
                raise RuntimeError(f"Robotiq calibrated pad size is malformed: {name}")
            pad_size[0] = RIGID_REVIEW_PROFILE.robotiq_pad_half_depth_m
            pad.set("size", " ".join(f"{value:.9g}" for value in pad_size))
            pad.set("condim", str(RIGID_REVIEW_PROFILE.robotiq_pad_condim))
            pad.set(
                "friction",
                " ".join(
                    f"{value:.9g}" for value in RIGID_REVIEW_PROFILE.robotiq_pad_friction
                ),
            )
            pad.set(
                "solref",
                " ".join(
                    f"{value:.9g}" for value in RIGID_REVIEW_PROFILE.robotiq_pad_solref
                ),
            )
            pad.set("contype", "1")
            pad.set("conaffinity", "1")
        actuator = root.find(".//actuator/general[@name='rq_fingers_actuator']")
        if actuator is None:
            raise RuntimeError("Robotiq tendon actuator is unavailable")
        actuator_force_limit = (
            controller_plan.robotiq_actuator_force_limit_n
            if controller_plan.robotiq_actuator_force_limit_n is not None
            else 0.16
        )
        actuator.set(
            "forcerange",
            f"{-actuator_force_limit:.9g} {actuator_force_limit:.9g}",
        )
        actuator.set("forcelimited", "true")
        actuator.set("ctrlrange", "0 255")
        actuator.set("ctrllimited", "true")

    # Camera height follows the owned task coordinate frame.  The
    # floor-rooted interceptions (F1, F2a) are explicitly anchored from
    # their physical/controller targets and remain at local height zero in
    # R0 and R1; other current recipes retain their declared table/support
    # frame.
    height_offset = (
        scenario.rolling_island_scene.table_top_z_m
        if scenario.rolling_island_scene is not None
        else 0.0
        if _is_floor_rooted_interception(scenario.corpus_leaf_id)
        else 0.74
        if scenario.requires_real_robocasa
        else 0.0
    )
    _repair_task_camera(root, scenario, height_offset)
    _add_secondary_camera(root, scenario, height_offset)
    # Visual background geometry must never participate in contact.
    # RoboCasa's imported geoms are commonly unnamed, so the owning body
    # prefix is the authoritative classification boundary.
    for body in root.findall(".//body"):
        if str(body.get("name") or "").startswith("rc_"):
            for geom in body.findall(".//geom"):
                geom.set("contype", "0")
                geom.set("conaffinity", "0")
    for geom in root.findall(".//geom"):
        name = str(geom.get("name") or "")
        if name.startswith("rc_") or str(geom.get("group") or "") == "1" and name.startswith("robocasa_"):
            geom.set("contype", "0")
            geom.set("conaffinity", "0")

    equality = root.find("equality")
    if equality is not None:
        serialized = ET.tostring(equality, encoding="unicode").lower()
        if "catch_ball" in serialized or "ball_freejoint" in serialized:
            raise RuntimeError("object-linked equality/latch is forbidden")


def _name2id(mujoco: Any, model: Any, object_type: Any, name: str) -> int:
    index = int(mujoco.mj_name2id(model, object_type, name))
    if index < 0:
        raise RuntimeError(f"compiled MuJoCo object is unavailable: {name}")
    return index


def _resolve_model_ids(mujoco: Any, model: Any, scenario: SourceMujocoCompiledScenario) -> SourceModelIds:
    ball_joint = _name2id(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, "ball_freejoint")
    object_body = _name2id(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, "catch_ball")
    object_geom = _name2id(mujoco, model, mujoco.mjtObj.mjOBJ_GEOM, "catch_ball_geom")
    surface_ids = {
        surface.name: _name2id(mujoco, model, mujoco.mjtObj.mjOBJ_GEOM, surface.name)
        for surface in scenario.surfaces
    }
    structural_support_ids = tuple(
        surface_ids[surface.name]
        for surface in scenario.surfaces
        if not surface.expected_task_contact
    )
    object_linked_equalities: list[int] = []
    for equality_id in range(model.neq):
        equality_type = int(model.eq_type[equality_id])
        object_ids = {
            int(model.eq_obj1id[equality_id]),
            int(model.eq_obj2id[equality_id]),
        }
        if equality_type in {
            int(mujoco.mjtEq.mjEQ_CONNECT),
            int(mujoco.mjtEq.mjEQ_WELD),
        } and object_body in object_ids:
            object_linked_equalities.append(equality_id)
        elif equality_type == int(mujoco.mjtEq.mjEQ_JOINT) and ball_joint in object_ids:
            object_linked_equalities.append(equality_id)
        elif equality_type == int(mujoco.mjtEq.mjEQ_DISTANCE) and object_geom in object_ids:
            object_linked_equalities.append(equality_id)
    if scenario.embodiment == "no_robot":
        return SourceModelIds(
            object_body=object_body,
            object_geom=object_geom,
            object_qpos_adr=int(model.jnt_qposadr[ball_joint]),
            object_qvel_adr=int(model.jnt_dofadr[ball_joint]),
            robot_joint_ids=(),
            robot_qpos_adrs=(),
            robot_qvel_adrs=(),
            actuator_ids=(),
            left_gripper_geom_ids=(),
            right_gripper_geom_ids=(),
            left_gripper_body=None,
            right_gripper_body=None,
            hand_body=None,
            grasp_site=None,
            surface_geom_ids=surface_ids,
            structural_support_geom_ids=structural_support_ids,
            object_linked_equality_ids=tuple(object_linked_equalities),
        )

    arm_names = tuple(f"joint{index}" for index in range(1, 8))
    if scenario.embodiment == FRANKA_HAND:
        joint_names = arm_names + ("finger_joint1", "finger_joint2")
        actuator_names = tuple(f"actuator{index}" for index in range(1, 9))
        left_body = _name2id(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
        right_body = _name2id(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
        left_geoms = tuple(
            index for index in range(model.ngeom) if int(model.geom_bodyid[index]) == left_body
        )
        right_geoms = tuple(
            index for index in range(model.ngeom) if int(model.geom_bodyid[index]) == right_body
        )
        hand_body = _name2id(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        grasp_site = None
    else:
        joint_names = arm_names + (
            "rq_right_driver_joint",
            "rq_right_coupler_joint",
            "rq_right_spring_link_joint",
            "rq_right_follower_joint",
            "rq_left_driver_joint",
            "rq_left_coupler_joint",
            "rq_left_spring_link_joint",
            "rq_left_follower_joint",
        )
        actuator_names = tuple(f"actuator{index}" for index in range(1, 8)) + (
            "rq_fingers_actuator",
        )
        left_body = _name2id(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, "rq_left_pad")
        right_body = _name2id(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, "rq_right_pad")
        left_geoms = tuple(
            _name2id(mujoco, model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in (
                "rq_left_pad1",
                "rq_left_pad2",
                "rq_left_pad_thick_collision_pad",
            )
        )
        right_geoms = tuple(
            _name2id(mujoco, model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in (
                "rq_right_pad1",
                "rq_right_pad2",
                "rq_right_pad_thick_collision_pad",
            )
        )
        hand_body = _name2id(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, "rq_base")
        grasp_site = _name2id(
            mujoco,
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            "robotiq_grasp_center_site",
        )
    joint_ids = tuple(
        _name2id(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names
    )
    actuator_ids = tuple(
        _name2id(mujoco, model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        for name in actuator_names
    )
    return SourceModelIds(
        object_body=object_body,
        object_geom=object_geom,
        object_qpos_adr=int(model.jnt_qposadr[ball_joint]),
        object_qvel_adr=int(model.jnt_dofadr[ball_joint]),
        robot_joint_ids=joint_ids,
        robot_qpos_adrs=tuple(int(model.jnt_qposadr[index]) for index in joint_ids),
        robot_qvel_adrs=tuple(int(model.jnt_dofadr[index]) for index in joint_ids),
        actuator_ids=actuator_ids,
        left_gripper_geom_ids=left_geoms,
        right_gripper_geom_ids=right_geoms,
        left_gripper_body=left_body,
        right_gripper_body=right_body,
        hand_body=hand_body,
        grasp_site=grasp_site,
        surface_geom_ids=surface_ids,
        structural_support_geom_ids=structural_support_ids,
        object_linked_equality_ids=tuple(object_linked_equalities),
    )


def _compiled_robot_base_pose(
    mujoco: Any,
    model: Any,
    data: Any,
    scenario: SourceMujocoCompiledScenario,
) -> tuple[
    tuple[float, float, float] | None,
    tuple[float, float, float, float] | None,
]:
    """Validate and return the compiled world pose of the owned robot base."""

    if scenario.embodiment == "no_robot":
        if int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link0")) >= 0:
            raise RuntimeError("no_robot scene retained link0")
        return None, None
    assert scenario.robot_base_position_m is not None
    assert scenario.robot_base_euler_rad is not None
    body_id = _name2id(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, "link0")
    mujoco.mj_forward(model, data)
    actual_position = tuple(float(value) for value in data.xpos[body_id])
    actual_quaternion = tuple(float(value) for value in data.xquat[body_id])
    expected_quaternion_array = np.empty(4, dtype=np.float64)
    mujoco.mju_euler2Quat(
        expected_quaternion_array,
        np.asarray(scenario.robot_base_euler_rad, dtype=np.float64),
        "xyz",
    )
    expected_quaternion = tuple(float(value) for value in expected_quaternion_array)
    if not np.allclose(
        actual_position,
        scenario.robot_base_position_m,
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError(
            "compiled link0 position differs from owned robot-base contract: "
            f"actual={actual_position}, expected={scenario.robot_base_position_m}"
        )
    if not np.allclose(
        actual_quaternion,
        expected_quaternion,
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError(
            "compiled link0 orientation differs from owned robot-base contract"
        )
    return actual_position, actual_quaternion


def _background_geometry_contract(
    mujoco: Any,
    model: Any,
    scenario: SourceMujocoCompiledScenario,
    robocasa_assets: Sequence[Mapping[str, Any]],
) -> tuple[tuple[Mapping[str, Any], ...], Mapping[str, str]]:
    """Classify every procedural/catalog background geom and enforce no contact.

    Worldbody geoms are owned task support only when explicitly excluded here;
    every other top-level geom is procedural appearance.  Catalog membership
    is resolved through body ancestry because imported mesh geoms are commonly
    unnamed.  Stable IDs remain independent of a worker or rollout process.
    """

    exclusions: dict[str, str] = {}
    if int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")) >= 0:
        exclusions["floor"] = "physical_room_support"
    exclusions.update(
        {
            surface.name: (
                "owned_structural_support"
                if not surface.expected_task_contact
                else f"owned_task_fixture:{surface.role}"
            )
            for surface in scenario.surfaces
        }
    )
    descriptors: list[Mapping[str, Any]] = []
    classified_geom_ids: set[int] = set()
    for geom_id in range(model.ngeom):
        if int(model.geom_bodyid[geom_id]) != 0:
            continue
        name = str(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        )
        if name in exclusions:
            continue
        if not name:
            raise RuntimeError("unnamed top-level procedural background geom")
        descriptors.append(
            {
                "stable_id": f"procedural:{name}",
                "geom_id": geom_id,
                "classification": "procedural",
                "source_name": name,
                "catalog_slot": None,
            }
        )
        classified_geom_ids.add(geom_id)

    rolling_wrapper_id = int(
        mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, ROLLING_ISLAND_WRAPPER_NAME
        )
    )
    if scenario.rolling_island_scene is not None and rolling_wrapper_id < 0:
        raise RuntimeError("rolling-island scenario omitted its scene wrapper")
    if scenario.rolling_island_scene is None and rolling_wrapper_id >= 0:
        raise RuntimeError("non-F3b scenario unexpectedly contains a rolling island")
    if rolling_wrapper_id >= 0:
        assert scenario.rolling_island_scene is not None
        counter_name = scenario.rolling_island_scene.counter_name
        counter_suffix = counter_name.removeprefix("island_")
        rolling_body_ids: set[int] = set()
        for body_id in range(1, model.nbody):
            cursor = body_id
            while cursor > 0:
                if cursor == rolling_wrapper_id:
                    rolling_body_ids.add(body_id)
                    break
                cursor = int(model.body_parentid[cursor])
        ordinal = 0
        for geom_id in range(model.ngeom):
            if int(model.geom_bodyid[geom_id]) not in rolling_body_ids:
                continue
            name = str(
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
                or ""
            )
            if name in exclusions:
                continue
            if geom_id in classified_geom_ids:
                raise RuntimeError("background geom received multiple classifications")
            suffix = name or f"geom:{ordinal:04d}"
            is_fixture_support = bool(
                name
                and (
                    name.endswith(counter_name)
                    or name.endswith(counter_suffix)
                )
            )
            descriptors.append(
                {
                    "stable_id": f"rolling_island:{suffix}",
                    "geom_id": geom_id,
                    "classification": (
                        "rolling_island_fixture_support"
                        if is_fixture_support
                        else "rolling_island"
                    ),
                    "source_name": name or None,
                    "catalog_slot": None,
                    "fixture_support_id": (
                        scenario.surfaces[0].name if is_fixture_support else None
                    ),
                }
            )
            classified_geom_ids.add(geom_id)
            ordinal += 1

    for item in robocasa_assets:
        slot = str(item.get("slot") or "")
        if not slot:
            raise RuntimeError("compiled RoboCasa asset lacks its catalog slot")
        prefix = f"rc_{slot}_"
        owned_body_ids: set[int] = set()
        for body_id in range(1, model.nbody):
            cursor = body_id
            while cursor > 0:
                body_name = str(
                    mujoco.mj_id2name(
                        model, mujoco.mjtObj.mjOBJ_BODY, cursor
                    )
                    or ""
                )
                if body_name.startswith(prefix):
                    owned_body_ids.add(body_id)
                    break
                cursor = int(model.body_parentid[cursor])
        geom_ids = tuple(
            geom_id
            for geom_id in range(model.ngeom)
            if int(model.geom_bodyid[geom_id]) in owned_body_ids
        )
        if not geom_ids:
            raise RuntimeError(f"catalog background has no compiled geoms: {slot}")
        for ordinal, geom_id in enumerate(geom_ids):
            if geom_id in classified_geom_ids:
                raise RuntimeError("background geom received multiple classifications")
            name = str(
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
                or ""
            )
            suffix = name or f"geom:{ordinal:04d}"
            descriptors.append(
                {
                    "stable_id": f"catalog:{slot}:{suffix}",
                    "geom_id": geom_id,
                    "classification": "catalog",
                    "source_name": name or None,
                    "catalog_slot": slot,
                }
            )
            classified_geom_ids.add(geom_id)

    # v4 makes the fixture relationship part of every descriptor, including
    # an explicit null for ordinary visual backgrounds.  Omitting the key on
    # some rows made the planned static hash differ from the independently
    # normalized runtime replay even though the geometry itself was identical.
    for descriptor in descriptors:
        descriptor.setdefault("fixture_support_id", None)
    descriptors.sort(key=lambda value: str(value["stable_id"]))
    if len({str(value["stable_id"]) for value in descriptors}) != len(descriptors):
        raise RuntimeError("background geometry stable IDs are not unique")
    for descriptor in descriptors:
        geom_id = int(descriptor["geom_id"])
        if (
            int(model.geom_contype[geom_id]) != 0
            or int(model.geom_conaffinity[geom_id]) != 0
        ):
            raise RuntimeError(
                "background geometry collision is enabled: "
                + str(descriptor["stable_id"])
            )
    return tuple(descriptors), dict(sorted(exclusions.items()))


def _compiled_world_geom_aabb(
    model: Any,
    data: Any,
    geom_id: int,
) -> tuple[float, float, float, float, float, float]:
    local = np.asarray(model.geom_aabb[int(geom_id)], dtype=np.float64)
    center_local = local[:3]
    half_local = local[3:]
    rotation = np.asarray(data.geom_xmat[int(geom_id)], dtype=np.float64).reshape(3, 3)
    center_world = np.asarray(data.geom_xpos[int(geom_id)], dtype=np.float64) + (
        rotation @ center_local
    )
    half_world = np.abs(rotation) @ half_local
    return tuple(float(value) for value in (*center_world - half_world, *center_world + half_world))  # type: ignore[return-value]


def _positive_aabb_overlap(
    left: Sequence[float],
    right: Sequence[float],
    *,
    tolerance_m: float = 1e-6,
) -> bool:
    return all(
        min(float(left[index + 3]), float(right[index + 3]))
        - max(float(left[index]), float(right[index]))
        > tolerance_m
        for index in range(3)
    )


def _remove_procedural_fixture_intersections(
    mujoco: Any,
    root: ET.Element,
    scenario: SourceMujocoCompiledScenario,
) -> tuple[str, ...]:
    """Remove R1 appearance geoms with positive owned-fixture volume overlap.

    This is a single geometry rule across P0 profiles, evaluated from MuJoCo's
    compiled AABBs after all deterministic relocation and support normalization.
    Face contact is retained; only positive volume intersection is removed.
    Catalog descendants are governed by their profile-level support lift and
    remain fail-closed in runtime clearance.
    """

    if (
        not scenario.surfaces
        or (
            not scenario.requires_real_robocasa
            and scenario.corpus_leaf_id not in {"F2f", "F3b"}
        )
    ):
        return ()
    provisional_xml = ET.tostring(root, encoding="unicode")
    provisional_model = mujoco.MjModel.from_xml_string(provisional_xml)
    provisional_data = mujoco.MjData(provisional_model)
    mujoco.mj_forward(provisional_model, provisional_data)
    fixture_ids = tuple(
        _name2id(mujoco, provisional_model, mujoco.mjtObj.mjOBJ_GEOM, surface.name)
        for surface in scenario.surfaces
    )
    fixture_aabbs = tuple(
        _compiled_world_geom_aabb(provisional_model, provisional_data, geom_id)
        for geom_id in fixture_ids
    )
    remove_names: list[str] = []
    exclusions = {"floor", *(surface.name for surface in scenario.surfaces)}
    for geom_id in range(provisional_model.ngeom):
        if int(provisional_model.geom_bodyid[geom_id]) != 0:
            continue
        name = str(
            mujoco.mj_id2name(
                provisional_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
            )
            or ""
        )
        if name in exclusions:
            continue
        aabb = _compiled_world_geom_aabb(
            provisional_model, provisional_data, geom_id
        )
        if any(_positive_aabb_overlap(aabb, fixture) for fixture in fixture_aabbs):
            if not name:
                raise RuntimeError(
                    "unnamed procedural background intersects an owned fixture"
                )
            remove_names.append(name)
    world = root.find("worldbody")
    if world is None:
        raise RuntimeError("external scene lacks worldbody")
    for name in sorted(set(remove_names)):
        geom = world.find(f"./geom[@name='{name}']")
        if geom is None:
            raise RuntimeError(
                f"classified fixture-intersecting procedural geom is not top-level: {name}"
            )
        world.remove(geom)
    return tuple(sorted(set(remove_names)))


def compile_source_model(
    scenario: SourceMujocoCompiledScenario,
    *,
    source_dependency: SourceDependencyManifest,
    robocasa_dependency: RoboCasaDependency | None,
    rolling_island_dependency: RollingIslandDependencyManifest | None = None,
) -> CompiledSourceModel:
    """Compile a model; every state/model mutation happens before initialization."""

    scenario.validate()
    if scenario.requires_real_robocasa and robocasa_dependency is None:
        raise RuntimeError("R1 review requires the real licensed RoboCasa dependency")
    if scenario.rolling_island_scene is not None and rolling_island_dependency is None:
        raise RuntimeError("rolling-island review requires its pinned scene dependency")
    try:
        import mujoco
    except ImportError as error:
        raise RuntimeError("source_mujoco requires the project MuJoCo extra") from error
    source_root = Path(source_dependency.source_root).resolve(strict=True)
    scene_builder = _load_external_scene_builder(source_root)
    sample = _build_external_sample(scene_builder, scenario)
    environment = {}
    if robocasa_dependency is not None:
        environment = {
            "ROBOCASA_ROOT": robocasa_dependency.source_root,
            "ROBOCASA_ASSETS_ROOT": robocasa_dependency.asset_root,
        }
    with _temporary_environment(environment):
        if scenario.embodiment == ROBOTIQ_2F85_THICK_PAD:
            bundle = scene_builder.build_robotiq_thick_pad_episode(sample)
        else:
            bundle = scene_builder.build_episode(sample)
    root = ET.fromstring(bundle.xml)
    stripped = _strip_robotwin(root)
    rolling_camera_pose: Mapping[str, Any] = {}
    if scenario.rolling_island_scene is not None:
        assert rolling_island_dependency is not None
        assert robocasa_dependency is not None
        rolling_camera_pose = install_rolling_island_scene(
            root,
            scenario.rolling_island_scene,
            dependency=rolling_island_dependency,
            robocasa=robocasa_dependency,
            seed=int(scenario.rng_subseeds["assets"]),
            lighting_intensity=float(sample.lighting_intensity),
            camera_jitter=tuple(float(value) for value in sample.camera_jitter),
            object_initial_position_m=scenario.object_initial_position_m,
            physical_target_position_m=(
                scenario.physical_target_position_m
                if scenario.physical_target_position_m is not None
                else scenario.controller_target_position_m
            ),
        )
    robocasa_assets: tuple[Mapping[str, Any], ...] = ()
    stripped_external_robocasa = 0
    if scenario.requires_real_robocasa:
        if robocasa_dependency is None:  # guarded above; keeps type/state fail-closed
            raise RuntimeError("R1 review requires the real licensed RoboCasa dependency")
        robocasa_assets, stripped_external_robocasa = (
            _inject_owned_robocasa_catalog_candidate(
                root,
                scenario=scenario,
                dependency=robocasa_dependency,
                source_root=source_root,
            )
        )
    (
        removed_visual_work_surfaces,
        removed_task_volume_backgrounds,
    ) = _remove_external_visual_work_surfaces(root, scenario)
    relocated_visual_backgrounds = _relocate_external_visual_boundaries(
        root, scenario
    )
    _apply_owned_robot_base(root, scenario)
    if scenario.embodiment == "no_robot":
        _remove_robot(root)
    _patch_calibrated_model(root, scenario)
    removed_fixture_intersections = _remove_procedural_fixture_intersections(
        mujoco, root, scenario
    )
    xml = ET.tostring(root, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    ids = _resolve_model_ids(mujoco, model, scenario)
    robot_base_position, robot_base_quaternion = _compiled_robot_base_pose(
        mujoco, model, data, scenario
    )
    if scenario.requires_real_robocasa and not robocasa_assets:
        raise RuntimeError("R1 scene imported no content-bound RoboCasa catalog asset")
    if scenario.requires_real_robocasa and any(
        str(item.get("model_xml") or "").strip() == "" for item in robocasa_assets
    ):
        raise RuntimeError("R1 RoboCasa selection lacks source descriptor identity")
    asset_manifest = referenced_asset_manifest(
        xml,
        allowed_roots=(
            source_dependency.source_root,
            *(
                (robocasa_dependency.asset_root,)
                if robocasa_dependency is not None
                else ()
            ),
        ),
    )
    background_geom_descriptors, background_geom_exclusions = (
        _background_geometry_contract(
            mujoco,
            model,
            scenario,
            robocasa_assets,
        )
    )
    return CompiledSourceModel(
        model=model,
        data=data,
        xml=xml,
        xml_sha256=hashlib.sha256(xml.encode("utf-8")).hexdigest(),
        ids=ids,
        camera_names={"main": "main_camera", "secondary": "secondary_camera"},
        source_asset_sha256=asset_manifest,
        robocasa_assets=robocasa_assets,
        stripped_robotwin_asset_count=stripped,
        stripped_external_robocasa_element_count=stripped_external_robocasa,
        removed_visual_work_surface_names=removed_visual_work_surfaces,
        removed_task_volume_background_names=removed_task_volume_backgrounds,
        removed_fixture_intersection_background_names=(
            removed_fixture_intersections
        ),
        relocated_visual_backgrounds=relocated_visual_backgrounds,
        background_geom_descriptors=background_geom_descriptors,
        background_geom_exclusions=background_geom_exclusions,
        robot_base_position_m=robot_base_position,
        robot_base_quaternion_wxyz=robot_base_quaternion,
        external_camera_metadata=_runtime_camera_metadata(
            {
                **(
                    dict(bundle.camera_pose)
                    if isinstance(bundle.camera_pose, Mapping)
                    else {}
                ),
                **dict(rolling_camera_pose),
            },
            robocasa_assets,
            removed_count=stripped_external_robocasa,
            removed_visual_work_surface_names=removed_visual_work_surfaces,
            removed_task_volume_background_names=(
                removed_task_volume_backgrounds
            ),
            removed_fixture_intersection_background_names=(
                removed_fixture_intersections
            ),
            relocated_visual_backgrounds=relocated_visual_backgrounds,
        ),
    )


__all__ = [
    "CompiledSourceModel",
    "SourceModelIds",
    "compile_source_model",
]
