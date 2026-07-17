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
from .compiler import PhysicalSurface, SourceMujocoCompiledScenario
from .profiles import RIGID_REVIEW_PROFILE
from .provenance import (
    RoboCasaDependency,
    SourceDependencyManifest,
    referenced_asset_manifest,
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
    external_camera_metadata: Mapping[str, Any]


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
        "material": "source_mujoco_task_fixture",
        "material_rgba": (0.20, 0.36, 0.58, 1.0),
        "roughness": 0.62,
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
    tabletop_height = 0.74 if scenario.requires_real_robocasa else None
    contact_groups = tuple(
        {"name": surface.role, "geoms": (surface.name,)} for surface in scenario.surfaces
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
        catch_center_z=(scenario.controller_target_position_m or (0.0, 0.0, 0.5))[2],
        tabletop_height=tabletop_height,
        planned_intercept_time_s=scenario.ballistic_event_time_s,
        planned_intercept_position=scenario.controller_target_position_m,
        interception_subfamily=scenario.subfamily,
        expected_contact_sequence=tuple(surface.role for surface in scenario.surfaces),
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
        position, yaw = _ROBOCASA_CATALOG_POSES[catalog_profile]
    except KeyError as error:
        raise RuntimeError(
            f"R1 profile lacks an owned supported background pose: {catalog_profile}"
        ) from error
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
) -> dict[str, Any]:
    """Replace stale external selection metadata with the executed selection."""

    result = dict(camera_pose)
    main = dict(result.get("main_camera") or {})
    serialized_assets = [dict(value) for value in runtime_assets]
    main["robocasa_background_assets"] = serialized_assets
    main["robocasa_background_selection_source"] = "owned_content_bound_catalog"
    main["external_random_robocasa_elements_removed"] = int(removed_count)
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


def _add_secondary_camera(root: ET.Element, height_offset: float) -> None:
    world = root.find("worldbody")
    if world is None:
        raise RuntimeError("external scene lacks worldbody")
    old = world.find("./camera[@name='secondary_camera']")
    if old is not None:
        world.remove(old)
    ET.SubElement(
        world,
        "camera",
        name="secondary_camera",
        pos=f"-0.72 -1.18 {1.10 + height_offset:.8g}",
        xyaxes="0.853  -0.522 0 0.239 0.391 0.889",
        fovy="52",
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
    # Scene furniture/walls are appearance context, not undeclared task
    # fixtures.  Only the room floor (when distinct from the task plane) and
    # explicitly declared owned surfaces retain collision.
    for geom in list(world.findall("./geom")):
        name = str(geom.get("name") or "")
        if name == "floor" and not scenario.requires_real_robocasa and any(
            surface.role in {"floor", "table", "slope"}
            for surface in scenario.surfaces
        ):
            world.remove(geom)
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

    for surface in scenario.surfaces:
        geom = root.find(f".//geom[@name='{surface.name}']")
        if geom is None:
            raise RuntimeError(f"compiled scene omitted physical fixture {surface.name}")
        geom.set("contype", "1")
        geom.set("conaffinity", "1")
        geom.set("condim", "3")
        geom.set("friction", " ".join(f"{value:.9g}" for value in surface.friction))
        geom.set("solref", " ".join(f"{value:.9g}" for value in surface.solref))
        # At 600 Hz a fast sphere can travel several millimetres per step.
        # A 2 mm predictive contact margin keeps geometric penetration within
        # the 3 mm hard limit without changing or scripting object motion.
        geom.set("margin", "0.002")

    if scenario.embodiment != "no_robot":
        _restore_collision_geometries(root)
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
                geom.set("margin", "0.002")
    if scenario.embodiment == ROBOTIQ_2F85_THICK_PAD:
        for name in (
            "rq_left_pad_thick_collision_pad",
            "rq_right_pad_thick_collision_pad",
        ):
            pad = root.find(f".//geom[@name='{name}']")
            if pad is None:
                raise RuntimeError(f"Robotiq calibrated pad is unavailable: {name}")
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
        actuator.set("forcerange", "-0.16 0.16")
        actuator.set("forcelimited", "true")
        actuator.set("ctrlrange", "0 255")
        actuator.set("ctrllimited", "true")

    height_offset = 0.74 if scenario.requires_real_robocasa else 0.0
    _add_secondary_camera(root, height_offset)
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
        object_linked_equality_ids=tuple(object_linked_equalities),
    )


def compile_source_model(
    scenario: SourceMujocoCompiledScenario,
    *,
    source_dependency: SourceDependencyManifest,
    robocasa_dependency: RoboCasaDependency | None,
) -> CompiledSourceModel:
    """Compile a model; every state/model mutation happens before initialization."""

    scenario.validate()
    if scenario.requires_real_robocasa and robocasa_dependency is None:
        raise RuntimeError("R1 review requires the real licensed RoboCasa dependency")
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
    if scenario.embodiment == "no_robot":
        _remove_robot(root)
    _patch_calibrated_model(root, scenario)
    xml = ET.tostring(root, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    ids = _resolve_model_ids(mujoco, model, scenario)
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
        external_camera_metadata=_runtime_camera_metadata(
            bundle.camera_pose if isinstance(bundle.camera_pose, Mapping) else {},
            robocasa_assets,
            removed_count=stripped_external_robocasa,
        ),
    )


__all__ = [
    "CompiledSourceModel",
    "SourceModelIds",
    "compile_source_model",
]
