"""Fail-closed review-case compiler for the owned rigid MuJoCo backend."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Any, Mapping, Sequence

import numpy as np

from ...common.corpus_registry import load_corpus_registry
from ...common.embodiments import FRANKA_HAND, ROBOTIQ_2F85_THICK_PAD
from ...common.hashing import sha256_json
from ...common.review import event_strip_frame_indices
from ...common.synchronization import fixed_duration_frame_timestamps
from .profiles import RIGID_REVIEW_PROFILE
from .provenance import (
    RoboCasaDependency,
    RollingIslandDependencyManifest,
    resolve_robocasa_dependency,
    resolve_rolling_island_dependency,
)
from .rolling_island import (
    ROLLING_ISLAND_SURFACE_NAME,
    RollingIslandScenePlan,
    resolve_rolling_island_plan,
)


SOURCE_MUJOCO_COMPILED_SCHEMA = "dynamic-robot-source-mujoco-compiled/v4"
SOURCE_MUJOCO_BACKEND_VERSION = "0.13.0-review"


class SourceMujocoUnsupported(ValueError):
    """The requested leaf/variant cannot be represented honestly yet."""


_SCENE_VARIANTS = {
    "clean_R0": "clean_lab",
    "robocasa_lab": "robocasa_lab",
    "robocasa_kitchen": "robocasa_kitchen",
    "robocasa_workbench": "robocasa_workbench",
    "robocasa_storage": "robocasa_storage",
    "robocasa_tabletop": "robocasa_tabletop",
}
_PASSIVE_VARIATION_PROFILES = (
    "nominal",
    "lower_initial_speed",
    "higher_initial_speed",
    "initial_spin",
    "lower_admitted_contact_parameter",
    "higher_admitted_contact_parameter",
)

# World-origin-centered supports derived from the complete fixed-six persisted
# high-rate object sweeps, expanded by the largest P0 sphere radius (25.5 mm)
# and a 100 mm visual/physics safety margin, then rounded outward.
_P0_SUPPORT_HALF_XY_M: Mapping[str, tuple[float, float]] = {
    # sweep x=[-0.150, 0.0361], y=-0.100
    "P0a": (0.28, 0.23),
    # sweep x=[-0.700, 0.4041], y=[-0.150, 0.1221]
    "P0b": (0.83, 0.28),
    # sweep x=[-0.450, 0.3707], y=[-0.0265, 0.0001]
    "P0c": (0.58, 0.16),
}

# This is intentionally narrower than the taxonomy registry.  Registry support
# means ownership; this table means an executable, non-proxy review attempt is
# implemented today.  Unsupported variants remain blocked instead of falling
# back to a visually similar task.
IMPLEMENTED_REVIEW_VARIANTS: Mapping[str, tuple[str, ...]] = {
    "P0a": ("nominal_freefall", "lateral_freefall"),
    "P0b": ("ballistic_projectile", "angled_projectile"),
    "P0c": ("table_bounce", "wall_rebound"),
    "P0d": ("straight_roll", "slope_roll"),
    "F1a": ("catch_retain", "catch_transport"),
    "F1b": ("off_center_catch", "off_center_near_miss"),
    "F1c": ("drift_catch", "drift_near_miss"),
    "F1d": ("mild_projectile_catch", "mild_projectile_near_miss"),
    "F2a": ("direct_catch", "direct_deflection"),
    "F2d": ("wall_rebound",),
    "F3b": ("rolling_pickup", "rolling_pickup_transport"),
}


def _finite_tuple(values: Sequence[Any], size: int, label: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or len(values) != size:
        raise SourceMujocoUnsupported(f"{label} must contain {size} values")
    result = tuple(float(value) for value in values)
    if any(not math.isfinite(value) for value in result):
        raise SourceMujocoUnsupported(f"{label} contains a non-finite value")
    return result


@dataclass(frozen=True, slots=True)
class PhysicalSurface:
    name: str
    role: str
    position_m: tuple[float, float, float]
    half_size_m: tuple[float, float, float]
    euler_rad: tuple[float, float, float] = (0.0, 0.0, 0.0)
    friction: tuple[float, float, float] = (0.9, 0.005, 0.0001)
    solref: tuple[float, float] = (0.012, 0.7)
    expected_task_contact: bool = True
    supports_fixture_id: str | None = None
    grounded_plane_z_m: float | None = None
    support_interface_maximum_mismatch_m: float | None = None

    def validate(self) -> None:
        if not self.name or self.role not in {
            "floor",
            "table",
            "wall",
            "ramp",
            "slope",
            "structural_support",
        }:
            raise SourceMujocoUnsupported("physical surface identity/role is invalid")
        _finite_tuple(self.position_m, 3, "surface position")
        size = _finite_tuple(self.half_size_m, 3, "surface size")
        if any(value <= 0 for value in size):
            raise SourceMujocoUnsupported("physical surface size must be positive")
        _finite_tuple(self.euler_rad, 3, "surface Euler rotation")
        friction = _finite_tuple(self.friction, 3, "surface friction")
        if any(value < 0 for value in friction):
            raise SourceMujocoUnsupported("surface friction cannot be negative")
        solref = _finite_tuple(self.solref, 2, "surface solref")
        if solref[0] <= 0 or solref[1] <= 0:
            raise SourceMujocoUnsupported("surface solref must be positive")
        if self.role == "structural_support":
            if self.expected_task_contact:
                raise SourceMujocoUnsupported(
                    "structural supports cannot be expected task contacts"
                )
            if not self.supports_fixture_id or self.grounded_plane_z_m is None:
                raise SourceMujocoUnsupported(
                    "structural support lacks its grounded fixture relationship"
                )
            if self.support_interface_maximum_mismatch_m is None:
                raise SourceMujocoUnsupported(
                    "structural support lacks its interface tolerance evidence"
                )
            ground = float(self.grounded_plane_z_m)
            mismatch = float(self.support_interface_maximum_mismatch_m)
            if (
                not math.isfinite(ground)
                or not math.isfinite(mismatch)
                or mismatch < 0.0
                or mismatch > 0.002 + 1e-12
            ):
                raise SourceMujocoUnsupported(
                    "structural support interface exceeds the 2 mm contract"
                )
        elif (
            not self.expected_task_contact
            or self.supports_fixture_id is not None
            or self.grounded_plane_z_m is not None
            or self.support_interface_maximum_mismatch_m is not None
        ):
            raise SourceMujocoUnsupported(
                "task fixtures cannot carry structural-support semantics"
            )


@dataclass(frozen=True, slots=True)
class SourceMujocoCompiledScenario:
    case_id: str
    episode_uuid: str
    corpus_leaf_id: str
    family: str
    subfamily: str
    task_variant: str
    embodiment: str
    branch_role: str
    intended_outcome: str
    scene_profile: str
    scene_variant: str
    randomization_level: str
    requires_real_robocasa: bool
    robot_base_position_m: tuple[float, float, float] | None
    robot_base_euler_rad: tuple[float, float, float] | None
    passive_variation_profile: str | None
    duration_s: float
    simulation_hz: int
    control_hz: int
    video_hz: int
    motion_kind: str
    object_radius_m: float
    object_mass_kg: float
    object_initial_position_m: tuple[float, float, float]
    object_initial_linear_velocity_m_s: tuple[float, float, float]
    object_initial_angular_velocity_rad_s: tuple[float, float, float]
    gravity_m_s2: tuple[float, float, float]
    key_event_time_s: float
    ballistic_event_time_s: float | None
    physical_target_position_m: tuple[float, float, float] | None
    controller_target_position_m: tuple[float, float, float] | None
    controller_transport_position_m: tuple[float, float, float] | None
    surfaces: tuple[PhysicalSurface, ...]
    rng_subseeds: Mapping[str, int]
    evaluator: str
    rolling_island_scene: RollingIslandScenePlan | None
    backend_version: str = SOURCE_MUJOCO_BACKEND_VERSION
    schema_version: str = SOURCE_MUJOCO_COMPILED_SCHEMA
    production_eligible: bool = False
    release_state: str = "blocked"

    def validate(self) -> None:
        if self.schema_version != SOURCE_MUJOCO_COMPILED_SCHEMA:
            raise SourceMujocoUnsupported("unsupported compiled scenario schema")
        if self.backend_version != SOURCE_MUJOCO_BACKEND_VERSION:
            raise SourceMujocoUnsupported("unsupported source_mujoco backend version")
        if self.production_eligible or self.release_state != "blocked":
            raise SourceMujocoUnsupported("review compilation cannot activate production")
        allowed = IMPLEMENTED_REVIEW_VARIANTS.get(self.corpus_leaf_id, ())
        if self.task_variant not in allowed:
            raise SourceMujocoUnsupported(
                f"{self.corpus_leaf_id}/{self.task_variant} has no honest review implementation"
            )
        if self.corpus_leaf_id.startswith("P0"):
            if self.embodiment != "no_robot":
                raise SourceMujocoUnsupported("passive P0 scenarios require no_robot")
            if self.passive_variation_profile not in _PASSIVE_VARIATION_PROFILES:
                raise SourceMujocoUnsupported(
                    "passive P0 scenario lacks its fixed variation profile"
                )
        elif self.embodiment not in {FRANKA_HAND, ROBOTIQ_2F85_THICK_PAD}:
            raise SourceMujocoUnsupported("actuated source_mujoco review requires a real gripper")
        elif self.passive_variation_profile is not None:
            raise SourceMujocoUnsupported(
                "actuated review cannot claim a passive variation profile"
            )
        if (
            self.simulation_hz
            not in {
                RIGID_REVIEW_PROFILE.simulation_hz,
                RIGID_REVIEW_PROFILE.comparison_simulation_hz,
            }
            or (self.control_hz, self.video_hz) != (60, 30)
        ):
            raise SourceMujocoUnsupported(
                "rigid review execution is restricted to the calibrated "
                "600/1200 Hz timestep pair at 60/30 Hz control/video"
            )
        if not math.isfinite(self.duration_s) or self.duration_s <= 0:
            raise SourceMujocoUnsupported("duration_s must be finite and positive")
        if self.scene_profile not in _SCENE_VARIANTS:
            raise SourceMujocoUnsupported(f"unknown fixed review scene {self.scene_profile!r}")
        if self.requires_real_robocasa != (self.scene_profile != "clean_R0"):
            raise SourceMujocoUnsupported("R1 scene/profile admission flags disagree")
        if self.randomization_level != ("R0" if self.scene_profile == "clean_R0" else "R1"):
            raise SourceMujocoUnsupported("scene profile uses the wrong randomization level")
        if self.corpus_leaf_id == "F3b" and self.requires_real_robocasa:
            if self.rolling_island_scene is None:
                raise SourceMujocoUnsupported(
                    "R1 F3b requires its audited RoboCasa rolling-island scene"
                )
            try:
                self.rolling_island_scene.validate()
            except ValueError as error:
                raise SourceMujocoUnsupported(str(error)) from error
            if self.rolling_island_scene.scene_profile != self.scene_profile:
                raise SourceMujocoUnsupported(
                    "rolling-island plan/profile identities disagree"
                )
        elif self.rolling_island_scene is not None:
            raise SourceMujocoUnsupported(
                "rolling-island scenes are restricted to randomized F3b review cases"
            )
        if self.embodiment == "no_robot":
            if self.robot_base_position_m is not None or self.robot_base_euler_rad is not None:
                raise SourceMujocoUnsupported("no_robot scenario cannot declare a robot base pose")
        elif self.robot_base_position_m is None or self.robot_base_euler_rad is None:
            raise SourceMujocoUnsupported("actuated scenario lacks its owned robot base pose")
        else:
            _finite_tuple(self.robot_base_position_m, 3, "robot base position")
            _finite_tuple(self.robot_base_euler_rad, 3, "robot base Euler rotation")
        _finite_tuple(self.object_initial_position_m, 3, "object position")
        _finite_tuple(self.object_initial_linear_velocity_m_s, 3, "object velocity")
        _finite_tuple(self.object_initial_angular_velocity_rad_s, 3, "object angular velocity")
        _finite_tuple(self.gravity_m_s2, 3, "gravity")
        if self.object_radius_m <= 0 or self.object_mass_kg <= 0:
            raise SourceMujocoUnsupported("object radius and mass must be positive")
        for optional_name, optional_value in (
            ("physical target", self.physical_target_position_m),
            ("controller target", self.controller_target_position_m),
            ("transport target", self.controller_transport_position_m),
        ):
            if optional_value is not None:
                _finite_tuple(optional_value, 3, optional_name)
        if self.embodiment != "no_robot" and (
            self.ballistic_event_time_s is None or self.controller_target_position_m is None
        ):
            raise SourceMujocoUnsupported("actuated interception lacks a ballistic/controller target")
        if self.ballistic_event_time_s is not None and (
            not math.isfinite(self.ballistic_event_time_s)
            or not 0 < self.ballistic_event_time_s < self.duration_s
        ):
            raise SourceMujocoUnsupported("ballistic event time lies outside the rollout")
        if (
            not math.isfinite(self.key_event_time_s)
            or self.key_event_time_s < 0.1
        ):
            raise SourceMujocoUnsupported(
                "key event must retain at least 0.1 s pre-event and 0.3 s post-event context"
            )
        try:
            strip = event_strip_frame_indices(
                fixed_duration_frame_timestamps(self.duration_s, self.video_hz),
                self.key_event_time_s,
            )
        except ValueError as error:
            raise SourceMujocoUnsupported(str(error)) from error
        if len(set(strip.values())) != len(strip):
            raise SourceMujocoUnsupported(
                "event strip frames, including final, must be distinct"
            )
        for surface in self.surfaces:
            surface.validate()
        surface_by_name = {surface.name: surface for surface in self.surfaces}
        if len(surface_by_name) != len(self.surfaces):
            raise SourceMujocoUnsupported("physical fixture names must be unique")
        if self.corpus_leaf_id == "F3b":
            task_surfaces = tuple(
                surface
                for surface in self.surfaces
                if surface.role != "structural_support"
            )
            if len(task_surfaces) != 1 or task_surfaces[0].role != "table":
                raise SourceMujocoUnsupported(
                    "F3b requires one real table surface and no runway/backstop"
                )
            if task_surfaces[0].name != ROLLING_ISLAND_SURFACE_NAME:
                raise SourceMujocoUnsupported(
                    "F3b table surface lacks its stable rolling-island identity"
                )
        structural_supports = tuple(
            surface
            for surface in self.surfaces
            if surface.role == "structural_support"
        )
        requires_owned_supports = bool(
            self.requires_real_robocasa
            and self.corpus_leaf_id in {"P0c", "P0d"}
        )
        if requires_owned_supports and len(structural_supports) != 4:
            raise SourceMujocoUnsupported(
                "elevated R1 P0c/P0d fixtures require four owned grounded supports"
            )
        if not requires_owned_supports and structural_supports:
            raise SourceMujocoUnsupported(
                "owned elevated structural supports are restricted to R1 P0c/P0d"
            )
        for support in structural_supports:
            target = surface_by_name.get(str(support.supports_fixture_id))
            if target is None or target.role not in {"floor", "table", "slope"}:
                raise SourceMujocoUnsupported(
                    "structural support targets an unavailable horizontal fixture"
                )
            assert support.grounded_plane_z_m is not None
            bottom_z = support.position_m[2] - support.half_size_m[2]
            if abs(bottom_z - support.grounded_plane_z_m) > 1e-12:
                raise SourceMujocoUnsupported(
                    "structural support bottom does not contact the room floor"
                )
            expected_top_z, maximum_mismatch = _support_interface_geometry(
                target,
                world_x_m=support.position_m[0],
                support_half_x_m=support.half_size_m[0],
            )
            actual_top_z = support.position_m[2] + support.half_size_m[2]
            if abs(actual_top_z - expected_top_z) > 1e-12:
                raise SourceMujocoUnsupported(
                    "structural support top is not constructed against its task surface"
                )
            if (
                support.support_interface_maximum_mismatch_m is None
                or abs(
                    support.support_interface_maximum_mismatch_m
                    - maximum_mismatch
                )
                > 1e-12
            ):
                raise SourceMujocoUnsupported(
                    "structural support interface evidence differs from geometry"
                )
        required_rng = {
            "physics",
            "initial_state",
            "camera",
            "assets",
            "controller",
            "scene_construction",
        }
        if set(self.rng_subseeds) != required_rng:
            raise SourceMujocoUnsupported("review case lacks independent RNG streams")
        values = [int(self.rng_subseeds[name]) for name in sorted(required_rng)]
        if len(set(values)) != len(values) or any(value < 0 or value >= 2**64 for value in values):
            raise SourceMujocoUnsupported("review RNG sub-seeds are invalid or coupled")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @property
    def scenario_sha256(self) -> str:
        return sha256_json(self.to_dict())


def _case_mapping(case: Mapping[str, Any] | Any) -> Mapping[str, Any]:
    if isinstance(case, Mapping):
        return case
    if hasattr(case, "to_dict"):
        value = case.to_dict()
        if isinstance(value, Mapping):
            return value
    raise TypeError("review case must be a mapping or expose to_dict()")


def _rng_mapping(value: Any) -> dict[str, int]:
    if hasattr(value, "__dataclass_fields__"):
        value = asdict(value)
    if not isinstance(value, Mapping):
        raise SourceMujocoUnsupported("review case rng_subseeds must be a mapping")
    return {str(key): int(item) for key, item in value.items()}


def _sphere_mass(radius_m: float, density_kg_m3: float = 44.0) -> float:
    return 4.0 / 3.0 * math.pi * radius_m**3 * density_kg_m3


def _surface_lower_face_z_at_world_x(
    surface: PhysicalSurface,
    world_x_m: float,
) -> float:
    """Return the lower box face height for the owned X/Z fixture profiles.

    Current P0 fixtures are either horizontal or pitched about MuJoCo's Y
    Euler axis.  Keeping this calculation in the compiler makes each support
    position a deterministic consequence of the task fixture, not a visual
    afterthought in the renderer.
    """

    roll, pitch, yaw = surface.euler_rad
    if abs(roll) > 1e-12 or abs(yaw) > 1e-12:
        raise SourceMujocoUnsupported(
            "owned structural supports require an X/Z-aligned task fixture"
        )
    cosine = math.cos(pitch)
    if abs(cosine) <= 1e-9:
        raise SourceMujocoUnsupported(
            "owned structural support cannot resolve a vertical task fixture"
        )
    sine = math.sin(pitch)
    local_x = (
        float(world_x_m)
        - surface.position_m[0]
        + sine * surface.half_size_m[2]
    ) / cosine
    if abs(local_x) > surface.half_size_m[0] + 1e-12:
        raise SourceMujocoUnsupported(
            "owned structural support lies outside its task fixture"
        )
    return (
        surface.position_m[2]
        - sine * local_x
        - cosine * surface.half_size_m[2]
    )


def _support_interface_geometry(
    surface: PhysicalSurface,
    *,
    world_x_m: float,
    support_half_x_m: float,
) -> tuple[float, float]:
    """Return center contact height and worst top/underside mismatch."""

    center = _surface_lower_face_z_at_world_x(surface, world_x_m)
    edges = (
        _surface_lower_face_z_at_world_x(
            surface, float(world_x_m) - float(support_half_x_m)
        ),
        _surface_lower_face_z_at_world_x(
            surface, float(world_x_m) + float(support_half_x_m)
        ),
    )
    return center, max(abs(value - center) for value in edges)


def _surface(
    name: str,
    role: str,
    position: Sequence[float],
    half_size: Sequence[float],
    *,
    euler: Sequence[float] = (0.0, 0.0, 0.0),
    table_rebound: bool = False,
    expected_task_contact: bool = True,
    supports_fixture_id: str | None = None,
    grounded_plane_z_m: float | None = None,
    support_interface_maximum_mismatch_m: float | None = None,
) -> PhysicalSurface:
    if role == "wall":
        solref = RIGID_REVIEW_PROFILE.wall_solref
    elif table_rebound:
        if role != "table":
            raise SourceMujocoUnsupported(
                "the table rebound contact profile requires a table surface"
            )
        solref = RIGID_REVIEW_PROFILE.table_rebound_solref
    else:
        solref = (0.003, 1.0)
    return PhysicalSurface(
        name=name,
        role=role,
        position_m=_finite_tuple(position, 3, "surface position"),  # type: ignore[arg-type]
        half_size_m=_finite_tuple(half_size, 3, "surface size"),  # type: ignore[arg-type]
        euler_rad=_finite_tuple(euler, 3, "surface rotation"),  # type: ignore[arg-type]
        solref=solref,
        expected_task_contact=expected_task_contact,
        supports_fixture_id=supports_fixture_id,
        grounded_plane_z_m=grounded_plane_z_m,
        support_interface_maximum_mismatch_m=(
            None
            if support_interface_maximum_mismatch_m is None
            else float(support_interface_maximum_mismatch_m)
        ),
    )


def _owned_grounded_structural_supports(
    *,
    leaf_id: str,
    tabletop_height_m: float,
    task_surfaces: Sequence[PhysicalSurface],
) -> tuple[PhysicalSurface, ...]:
    """Construct the fixed four-leg frame beneath elevated P0c/P0d fixtures.

    The legs are world-fixed collision geoms whose bottoms meet the room floor
    at z=0.  They sit near the task fixture's lateral edges, safely outside the
    complete fixed-six object corridor around y=0.  P0d uses narrow X extents
    so a vertical leg follows a pitched underside to within 2 mm.
    """

    if tabletop_height_m <= 0.0 or leaf_id not in {"P0c", "P0d"}:
        return ()
    target = next(
        (
            surface
            for surface in task_surfaces
            if surface.role in {"floor", "table", "slope"}
            and surface.expected_task_contact
        ),
        None,
    )
    if target is None:
        raise SourceMujocoUnsupported(
            "elevated P0 structural frame lacks its task surface"
        )
    if leaf_id == "P0c":
        local_x_positions = (-0.50, 0.50)
        support_y_positions = (-0.125, 0.125)
        support_half_x = 0.04
        support_half_y = 0.025
    else:
        local_x_positions = (-1.0, 1.0)
        support_y_positions = (-0.47, 0.47)
        support_half_x = 0.01
        support_half_y = 0.04
    pitch = target.euler_rad[1]
    cosine = math.cos(pitch)
    sine = math.sin(pitch)
    result: list[PhysicalSurface] = []
    for x_index, local_x in enumerate(local_x_positions):
        # Transform the selected local underside point into world X.  The
        # center top of the vertical support meets that exact point.
        world_x = (
            target.position_m[0]
            + cosine * local_x
            - sine * target.half_size_m[2]
        )
        top_z, mismatch = _support_interface_geometry(
            target,
            world_x_m=world_x,
            support_half_x_m=support_half_x,
        )
        if top_z <= 0.0:
            raise SourceMujocoUnsupported(
                "owned structural support has no positive grounded height"
            )
        for y_index, world_y in enumerate(support_y_positions):
            result.append(
                _surface(
                    f"owned_structural_leg_x{x_index}_y{y_index}",
                    "structural_support",
                    (world_x, world_y, top_z / 2.0),
                    (support_half_x, support_half_y, top_z / 2.0),
                    expected_task_contact=False,
                    supports_fixture_id=target.name,
                    grounded_plane_z_m=0.0,
                    support_interface_maximum_mismatch_m=mismatch,
                )
            )
    return tuple(result)


def _recipe(
    leaf_id: str,
    task_variant: str,
    embodiment: str,
    branch_role: str,
    *,
    seed: int,
    tabletop_height_m: float,
    rolling_island_scene: RollingIslandScenePlan | None = None,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    gravity = (0.0, 0.0, -9.81)
    radius = (
        float(rng.uniform(*RIGID_REVIEW_PROFILE.panda_ball_radius_range_m))
        if embodiment == FRANKA_HAND
        else 0.0245
    )
    mass = _sphere_mass(radius)
    catch_z = tabletop_height_m + 0.50
    close_z = catch_z + 0.055
    r1_support_half_height = 0.02 if tabletop_height_m > 0.0 else 0.04
    r1_support_center_z = tabletop_height_m - r1_support_half_height
    negative = branch_role not in {"nominal_success", "passive_observation"}
    duration = 2.0
    base: dict[str, Any] = {
        "duration_s": duration,
        "motion_kind": "unknown",
        "object_radius_m": radius,
        "object_mass_kg": mass,
        "object_initial_angular_velocity_rad_s": (0.0, 0.0, 0.0),
        "gravity_m_s2": gravity,
        "key_event_time_s": 0.5,
        "ballistic_event_time_s": None,
        "physical_target_position_m": None,
        "controller_target_position_m": None,
        "controller_transport_position_m": None,
        "surfaces": (),
    }

    if leaf_id == "P0a":
        support_xy = (
            _P0_SUPPORT_HALF_XY_M[leaf_id]
            if tabletop_height_m > 0.0
            else (1.4, 1.4)
        )
        lateral = 0.28 if task_variant == "lateral_freefall" else 0.0
        base.update(
            duration_s=0.8,
            key_event_time_s=0.3,
            motion_kind="passive_freefall",
            object_initial_position_m=(-0.15, -0.10, tabletop_height_m + 0.47),
            object_initial_linear_velocity_m_s=(lateral, 0.0, 0.0),
            surfaces=(_surface("supported_floor", "floor", (0, 0, r1_support_center_z), (*support_xy, r1_support_half_height)),),
        )
    elif leaf_id == "P0b":
        support_xy = (
            _P0_SUPPORT_HALF_XY_M[leaf_id]
            if tabletop_height_m > 0.0
            else (1.4, 1.4)
        )
        vy = 0.34 if task_variant == "angled_projectile" else 0.0
        base.update(
            duration_s=0.8,
            key_event_time_s=0.4,
            motion_kind="passive_projectile",
            object_initial_position_m=(-0.70, -0.15, tabletop_height_m + 0.50),
            object_initial_linear_velocity_m_s=(1.15, vy, 3.50),
            surfaces=(_surface("supported_floor", "floor", (0, 0, r1_support_center_z), (*support_xy, r1_support_half_height)),),
        )
    elif leaf_id == "P0c" and task_variant == "table_bounce":
        support_xy = (
            _P0_SUPPORT_HALF_XY_M[leaf_id]
            if tabletop_height_m > 0.0
            else (1.2, 0.8)
        )
        base.update(
            duration_s=1.3,
            key_event_time_s=0.31,
            motion_kind="passive_table_bounce",
            object_initial_position_m=(-0.30, 0.0, tabletop_height_m + 0.75),
            object_initial_linear_velocity_m_s=(0.55, 0.0, -0.85),
            surfaces=(
                _surface(
                    "supported_bounce_table",
                    "table",
                    (0, 0, r1_support_center_z),
                    (*support_xy, r1_support_half_height),
                    table_rebound=True,
                ),
            ),
        )
    elif leaf_id == "P0c" and task_variant == "wall_rebound":
        support_xy = (
            _P0_SUPPORT_HALF_XY_M[leaf_id]
            if tabletop_height_m > 0.0
            else (1.4, 1.4)
        )
        wall_half_y = support_xy[1] if tabletop_height_m > 0.0 else 0.65
        # Reach the wall after a complete airborne arc.  The earlier horizontal-only
        # launch struck the support floor after 0.35 s and merely rolled toward
        # the wall, so its nominal "wall rebound" never occurred.  The wall
        # face/contact-center geometry gives the deterministic 0.6868 s event.
        wall_event_time = (0.35 - 0.02 - radius + 0.45) / 1.10
        base.update(
            duration_s=1.3,
            key_event_time_s=wall_event_time,
            motion_kind="passive_wall_rebound",
            object_initial_position_m=(-0.45, 0.0, tabletop_height_m + 0.62),
            object_initial_linear_velocity_m_s=(
                1.10,
                0.0,
                0.5 * abs(gravity[2]) * wall_event_time,
            ),
            surfaces=(
                _surface("supported_floor", "floor", (0, 0, r1_support_center_z), (*support_xy, r1_support_half_height)),
                _surface("supported_wall", "wall", (0.35, 0.0, tabletop_height_m + 0.62), (0.02, wall_half_y, 0.62)),
            ),
        )
    elif leaf_id == "P0d":
        slope = 0.10 if task_variant == "slope_roll" else 0.0
        speed = 0.72
        base.update(
            key_event_time_s=0.5,
            motion_kind="passive_slope_roll" if slope else "passive_straight_roll",
            object_initial_position_m=(-0.65, 0.0, tabletop_height_m + radius + 0.002),
            object_initial_linear_velocity_m_s=(speed, 0.0, 0.0),
            object_initial_angular_velocity_rad_s=(0.0, speed / radius, 0.0),
            surfaces=(
                _surface(
                    "supported_rolling_surface",
                    "slope" if slope else "table",
                    (0, 0, tabletop_height_m - 0.02),
                    (1.2, 0.55, 0.02),
                    euler=(0.0, -slope, 0.0),
                ),
            ),
        )
    elif leaf_id.startswith("F1") or leaf_id == "F2a":
        event_time = 0.44
        # The ballistic state is constructed so the ball center reaches
        # ``close_z`` at the closure event.  The physical/controller target
        # must be that same point; the previous catch_z target placed the palm
        # 55 mm below the actual event and let the ball sink into the hand.
        target = np.array((0.47, 0.0, close_z), dtype=np.float64)
        start_xy = target[:2].copy()
        if leaf_id == "F1b":
            target[1] = 0.035
            start_xy = target[:2].copy()
        elif leaf_id == "F1c":
            start_xy += np.array((-0.09, 0.05))
        elif leaf_id in {"F1d", "F2a"}:
            start_xy += np.array((-0.12, -0.055))
        # Freeze the nominal physical velocity before applying a negative
        # initial-state intervention.  Recomputing it after shifting the start
        # position silently steered the object back to the grasp target and
        # turned every intended initial-state failure into a success.
        velocity_xy = (target[:2] - start_xy) / event_time
        controller_target = target.copy()
        if negative:
            # Keep the physical initial state and intended difficult seed.  The
            # negative branch changes only its declared intervention stream.
            if "initial_state" in branch_role:
                # A smaller 85 mm shift missed the fingers but let the fixed
                # Panda review seeds clip link5/link6 on the way down.  Keep
                # the intervention and seed fixed, but make the declared
                # lateral state perturbation a clean 135 mm physical miss.
                start_xy[1] += 0.135
            elif "controller" in branch_role or "near_miss" in task_variant:
                controller_target[1] += 0.10
            else:
                controller_target[0] -= 0.10
        start_z = float(target[2]) + 0.5 * 9.81 * event_time**2
        transport = (
            (float(controller_target[0] + 0.12), float(controller_target[1]), catch_z + 0.06)
            if task_variant == "catch_transport"
            else None
        )
        base.update(
            key_event_time_s=event_time,
            motion_kind="direct_free_contact_interception",
            object_initial_position_m=(float(start_xy[0]), float(start_xy[1]), float(start_z)),
            object_initial_linear_velocity_m_s=(float(velocity_xy[0]), float(velocity_xy[1]), 0.0),
            ballistic_event_time_s=event_time,
            physical_target_position_m=tuple(float(value) for value in target),
            controller_target_position_m=tuple(float(value) for value in controller_target),
            controller_transport_position_m=transport,
            surfaces=(),
        )
    elif leaf_id == "F3b":
        base["duration_s"] = 2.5
        event_time = RIGID_REVIEW_PROFILE.rolling_pickup_event_time_s
        speed = RIGID_REVIEW_PROFILE.rolling_pickup_speed_m_s
        if rolling_island_scene is None:
            # R0 uses one neutral, full table.  It deliberately has neither
            # the former blue miniature runway nor its artificial backstop.
            table_top_z = 0.10
            table_position = (0.60, 0.0, table_top_z / 2.0)
            table_half_size = (0.65, 0.40, table_top_z / 2.0)
            table_yaw = 0.0
        else:
            table_top_z = rolling_island_scene.table_top_z_m
            table_position = rolling_island_scene.surface_position_task_m
            table_half_size = rolling_island_scene.surface_half_size_m
            table_yaw = rolling_island_scene.counter_yaw_task_rad
        # The intercept lies on the actual counter surface.  The object is
        # initialized rolling without slip; all subsequent robot motion is
        # produced through actuators and all ball motion through contact.
        target = np.array((0.47, 0.0, table_top_z + radius), dtype=np.float64)
        start_xy = np.array((float(target[0]) + speed * event_time, 0.0))
        controller_target = target.copy()
        if embodiment == ROBOTIQ_2F85_THICK_PAD:
            # The 2f85 finger structure extends below the thick-pad midpoint;
            # commanding the midpoint to ball-center height bottoms the
            # knuckles out on the runway.  Stand off by the measured
            # clearance and pinch the upper hemisphere instead.
            controller_target[2] += RIGID_REVIEW_PROFILE.robotiq_pickup_standoff_m
        if negative:
            # Keep the physical rolling state and intended difficult seed;
            # negatives change only their declared intervention stream.
            if "initial_state" in branch_role:
                # The declared lateral perturbation is the same clean 135 mm
                # physical miss the falling-catch negatives use.
                start_xy[1] += 0.135
            elif "controller" in branch_role:
                controller_target[1] += 0.10
            else:
                controller_target[0] -= 0.10
        lift_x = (
            float(controller_target[0])
            if task_variant == "rolling_pickup"
            else float(
                controller_target[0]
                + RIGID_REVIEW_PROFILE.pickup_transport_lateral_m
            )
        )
        if embodiment == ROBOTIQ_2F85_THICK_PAD:
            lift_x += RIGID_REVIEW_PROFILE.robotiq_pickup_capture_followthrough_x_m
        transport = (
            lift_x,
            float(controller_target[1]),
            float(target[2]) + RIGID_REVIEW_PROFILE.pickup_lift_height_m,
        )
        if negative:
            # Preserve the failed rollout instead of commanding a gratuitous
            # empty-hand lift after the measured miss.  Successful branches
            # still require the full contact-supported displacement evidence.
            transport = None
        base.update(
            key_event_time_s=event_time,
            motion_kind="rolling_pickup_interception",
            object_initial_position_m=(
                float(start_xy[0]),
                float(start_xy[1]),
                float(table_top_z + radius),
            ),
            object_initial_linear_velocity_m_s=(-speed, 0.0, 0.0),
            object_initial_angular_velocity_rad_s=(0.0, -speed / radius, 0.0),
            ballistic_event_time_s=event_time,
            physical_target_position_m=tuple(float(value) for value in target),
            controller_target_position_m=tuple(
                float(value) for value in controller_target
            ),
            controller_transport_position_m=transport,
            surfaces=(
                _surface(
                    ROLLING_ISLAND_SURFACE_NAME,
                    "table",
                    table_position,
                    table_half_size,
                    euler=(0.0, 0.0, table_yaw),
                ),
            ),
        )
    elif leaf_id == "F2c":
        event_time = 0.68
        bounce_time = 0.25
        target = np.array((0.50, 0.0, catch_z), dtype=np.float64)
        plate_top = tabletop_height_m + 0.008
        contact_z = plate_top + radius
        post_time = event_time - bounce_time
        outgoing_vz = (close_z - contact_z + 0.5 * 9.81 * post_time**2) / post_time
        incoming_vz = -outgoing_vz / RIGID_REVIEW_PROFILE.wall_effective_restitution
        start_z = contact_z - incoming_vz * bounce_time + 0.5 * 9.81 * bounce_time**2
        start_x = 0.22
        vx = (target[0] - start_x) / event_time
        controller_target = target + (np.array((0.0, 0.10, 0.0)) if negative else 0.0)
        base.update(
            key_event_time_s=event_time,
            motion_kind="floor_rebound_interception",
            object_initial_position_m=(float(start_x), 0.0, float(start_z)),
            object_initial_linear_velocity_m_s=(float(vx), 0.0, float(incoming_vz)),
            ballistic_event_time_s=event_time,
            physical_target_position_m=tuple(float(value) for value in target),
            controller_target_position_m=tuple(float(value) for value in controller_target),
            surfaces=(
                _surface(
                    "supported_floor_bounce_plate",
                    "table",
                    (0.32, 0.0, tabletop_height_m + 0.004),
                    (0.30, 0.28, 0.008),
                ),
            ),
        )
    elif leaf_id == "F2d" and task_variant == "wall_rebound":
        event_time = 0.62
        wall_time = 0.22
        target = np.array((0.47, 0.0, catch_z), dtype=np.float64)
        wall_x = 0.18
        face_x = wall_x + 0.02 + radius
        post_time = event_time - wall_time
        vx_post = (target[0] - face_x) / post_time
        vx_pre = -vx_post / RIGID_REVIEW_PROFILE.wall_effective_restitution
        start_x = face_x - vx_pre * wall_time
        start_z = close_z + 0.5 * 9.81 * event_time**2
        controller_target = target + (np.array((0.0, 0.10, 0.0)) if negative else 0.0)
        base.update(
            key_event_time_s=event_time,
            motion_kind="wall_rebound_interception",
            object_initial_position_m=(float(start_x), 0.0, float(start_z)),
            object_initial_linear_velocity_m_s=(float(vx_pre), 0.0, 0.0),
            ballistic_event_time_s=event_time,
            physical_target_position_m=tuple(float(value) for value in target),
            controller_target_position_m=tuple(float(value) for value in controller_target),
            surfaces=(
                _surface(
                    "supported_wall_rebound_barrier",
                    "wall",
                    (wall_x, 0.0, tabletop_height_m + 0.58),
                    (0.02, 0.50, 0.58),
                ),
            ),
        )
    else:
        raise SourceMujocoUnsupported(
            f"no physical source_mujoco recipe for {leaf_id}/{task_variant}"
        )
    return base


def _apply_passive_variation(
    recipe: Mapping[str, Any],
    *,
    leaf_id: str,
    task_variant: str,
    profile: str,
) -> dict[str, Any]:
    """Apply the declared fixed P0 variation without changing its seed.

    Speed changes preserve the physical recipe rather than merely changing an
    appearance seed.  Contact variants modify only the owned fixture material
    before initialization.  The event time is recomputed when the variation
    changes a deterministic rebound intercept.
    """

    if profile not in _PASSIVE_VARIATION_PROFILES:
        raise SourceMujocoUnsupported(f"unknown passive variation profile {profile!r}")
    result = dict(recipe)
    if profile == "nominal":
        return result

    if profile in {"lower_initial_speed", "higher_initial_speed"}:
        factor = 0.8 if profile == "lower_initial_speed" else 1.2
        linear = np.asarray(result["object_initial_linear_velocity_m_s"], dtype=np.float64)
        angular = np.asarray(result["object_initial_angular_velocity_rad_s"], dtype=np.float64)
        linear *= factor
        angular *= factor
        if leaf_id == "P0a":
            position = np.asarray(result["object_initial_position_m"], dtype=np.float64)
            position[2] += -0.06 if factor < 1.0 else 0.06
            result["object_initial_position_m"] = tuple(float(value) for value in position)
        if leaf_id == "P0c" and task_variant == "wall_rebound":
            wall = next(surface for surface in result["surfaces"] if surface.role == "wall")
            start_x = float(result["object_initial_position_m"][0])
            contact_x = wall.position_m[0] - wall.half_size_m[0] - float(
                result["object_radius_m"]
            )
            event_time = (contact_x - start_x) / float(linear[0])
            if not 0.1 < event_time < float(result["duration_s"]) - 0.3:
                raise SourceMujocoUnsupported(
                    "passive wall speed variation loses required review context"
                )
            linear[2] = 0.5 * abs(float(result["gravity_m_s2"][2])) * event_time
            result["key_event_time_s"] = event_time
        elif leaf_id == "P0c" and task_variant == "table_bounce":
            surface = result["surfaces"][0]
            contact_z = (
                surface.position_m[2]
                + surface.half_size_m[2]
                + float(result["object_radius_m"])
            )
            height = float(result["object_initial_position_m"][2]) - contact_z
            gravity = abs(float(result["gravity_m_s2"][2]))
            vertical = float(linear[2])
            event_time = (vertical + math.sqrt(vertical**2 + 2.0 * gravity * height)) / gravity
            result["key_event_time_s"] = event_time
        result["object_initial_linear_velocity_m_s"] = tuple(
            float(value) for value in linear
        )
        result["object_initial_angular_velocity_rad_s"] = tuple(
            float(value) for value in angular
        )
        return result

    if profile == "initial_spin":
        angular = np.asarray(result["object_initial_angular_velocity_rad_s"], dtype=np.float64)
        if leaf_id == "P0d":
            # For a rolling leaf, yaw spin is not an admissible variation: it
            # intentionally creates gross lateral slip.  Vary spin about the
            # physical roll axis instead, retaining a bounded 10% overspin
            # that the frictional contact resolves without scripted motion.
            linear = np.asarray(
                result["object_initial_linear_velocity_m_s"], dtype=np.float64
            )
            angular[:] = (0.0, 1.10 * linear[0] / float(result["object_radius_m"]), 0.0)
        else:
            angular[2] += 8.0
        result["object_initial_angular_velocity_rad_s"] = tuple(
            float(value) for value in angular
        )
        return result

    contact_factor = (
        0.8 if profile == "lower_admitted_contact_parameter" else 1.2
    )
    result["surfaces"] = tuple(
        replace(
            surface,
            friction=tuple(value * contact_factor for value in surface.friction),
            solref=(
                # P0c's v7 rebound calibration owns one normal-contact
                # profile at both solver rates.  The fixed contact-parameter
                # counterfactual varies tangential friction only; changing
                # solref here made review-04 pass at 600 Hz but exceed the
                # 3 mm penetration limit at the 1200 Hz reference.
                surface.solref[0]
                if leaf_id == "P0c" and task_variant == "table_bounce"
                else surface.solref[0]
                * (1.15 if contact_factor < 1.0 else 0.85),
                surface.solref[1],
            ),
        )
        for surface in result["surfaces"]
    )
    return result


def compile_review_case(
    case: Mapping[str, Any] | Any,
    *,
    rolling_island_dependency: RollingIslandDependencyManifest | None = None,
    robocasa_dependency: RoboCasaDependency | None = None,
) -> SourceMujocoCompiledScenario:
    """Compile one fixed review case without making a release claim."""

    value = _case_mapping(case)
    leaf_id = str(value.get("corpus_leaf_id") or "")
    task_variant = str(value.get("task_variant") or "")
    embodiment = str(value.get("embodiment") or "")
    allowed = IMPLEMENTED_REVIEW_VARIANTS.get(leaf_id)
    if allowed is None or task_variant not in allowed:
        raise SourceMujocoUnsupported(
            f"source_mujoco has no honest review implementation for {leaf_id}/{task_variant}"
        )
    if str(value.get("backend") or "") != "source_mujoco":
        raise SourceMujocoUnsupported("review case is assigned to a different backend")
    corpus = load_corpus_registry()
    leaf = corpus.resolve(leaf_id, embodiment=embodiment, task_variant=task_variant)
    scene_profile = str(value.get("scene_profile") or "")
    if scene_profile not in _SCENE_VARIANTS:
        raise SourceMujocoUnsupported(f"unknown review scene profile {scene_profile!r}")
    randomization_level = str(value.get("randomization_level") or "")
    requires_real_robocasa = bool(value.get("requires_real_robocasa"))
    rng_subseeds = _rng_mapping(value.get("rng_subseeds"))
    rolling_island_scene: RollingIslandScenePlan | None = None
    if leaf_id == "F3b" and requires_real_robocasa:
        rolling_dependency = (
            rolling_island_dependency or resolve_rolling_island_dependency()
        )
        robocasa = robocasa_dependency or resolve_robocasa_dependency()
        rolling_island_scene = resolve_rolling_island_plan(
            scene_profile,
            dependency=rolling_dependency,
            robocasa=robocasa,
            seed=rng_subseeds["assets"],
        )
    external_tabletop_height = (
        rolling_island_scene.table_top_z_m
        if rolling_island_scene is not None
        else (0.74 if requires_real_robocasa else 0.0)
    )
    # F1 is a free-space interception rooted at the room floor.  The external
    # R1 scene historically raised both robot and task by 0.74 m to sit on a
    # procedural table whose collision was later disabled.  That made R0 and
    # R1 different physical tasks and let failed balls pass through visible
    # furniture.  Keep the owned local F1 task/base pose invariant and treat
    # R1 furniture solely as remote appearance context.  F2a shares F1's
    # fixtureless ballistic interception and therefore its floor rooting:
    # counter-rooted F2a misses measurably fell 2.2 m out of frame and into
    # background furniture, failing the rendered visibility/clearance QC that
    # the F1 rooting was introduced to satisfy.
    task_height = (
        0.0
        if leaf_id.startswith("F1") or leaf_id == "F2a"
        else external_tabletop_height
    )
    robot_base_position = (
        None if embodiment == "no_robot" else (0.0, 0.0, task_height)
    )
    robot_base_euler = None if embodiment == "no_robot" else (0.0, 0.0, 0.0)
    branch_role = str(value.get("branch_role") or "")
    passive_variation_profile = (
        str(value.get("passive_variation_profile") or "")
        if leaf_id.startswith("P0")
        else None
    )
    recipe = _recipe(
        leaf_id,
        task_variant,
        embodiment,
        branch_role,
        seed=rng_subseeds["initial_state"],
        tabletop_height_m=task_height,
        rolling_island_scene=rolling_island_scene,
    )
    if leaf_id.startswith("P0"):
        recipe = _apply_passive_variation(
            recipe,
            leaf_id=leaf_id,
            task_variant=task_variant,
            profile=str(passive_variation_profile),
        )
    task_surfaces = tuple(recipe.get("surfaces", ()))
    recipe["surfaces"] = (
        *task_surfaces,
        *_owned_grounded_structural_supports(
            leaf_id=leaf_id,
            tabletop_height_m=task_height,
            task_surfaces=task_surfaces,
        ),
    )
    # The 600 Hz candidate is admitted only for fixed cases that preserve the
    # strict contact/rebound result at the 1200 Hz reference.  The lower-speed
    # P0c wall case exceeds 3 mm at 600 Hz, and the Panda F1a/F1b
    # negative-timing drops reach the room floor at about 5 m/s.  Under the
    # v9 reaching controller the F1a/F1d Robotiq nominal catches keep
    # outcome/QC/replay agreement but shift the first-bilateral-contact
    # sample past the 1 cm gate (12.3 mm and 21.8 mm measured).  Compile only
    # those measured exception classes directly at the calibrated reference
    # rate instead of weakening QC.
    requires_reference_rate = (
        (
            leaf_id == "P0c"
            and passive_variation_profile == "lower_initial_speed"
        )
        or (
            leaf_id in {"F1a", "F1b", "F2a"}
            and embodiment == FRANKA_HAND
            and "controller" in branch_role
        )
        or (
            leaf_id in {"F1a", "F1d", "F2a"}
            and embodiment == ROBOTIQ_2F85_THICK_PAD
            and branch_role == "nominal_success"
        )
    )
    result = SourceMujocoCompiledScenario(
        case_id=str(value.get("case_id") or ""),
        episode_uuid=str(value.get("episode_uuid") or ""),
        corpus_leaf_id=leaf_id,
        family=leaf.family,
        subfamily=leaf.subfamily,
        task_variant=task_variant,
        embodiment=embodiment,
        branch_role=branch_role,
        intended_outcome=str(value.get("intended_outcome") or ""),
        scene_profile=scene_profile,
        scene_variant=_SCENE_VARIANTS[scene_profile],
        randomization_level=randomization_level,
        requires_real_robocasa=requires_real_robocasa,
        robot_base_position_m=robot_base_position,
        robot_base_euler_rad=robot_base_euler,
        passive_variation_profile=passive_variation_profile,
        simulation_hz=(
            RIGID_REVIEW_PROFILE.comparison_simulation_hz
            if requires_reference_rate
            else RIGID_REVIEW_PROFILE.simulation_hz
        ),
        control_hz=RIGID_REVIEW_PROFILE.control_hz,
        video_hz=RIGID_REVIEW_PROFILE.video_hz,
        rng_subseeds=rng_subseeds,
        evaluator=str(value.get("evaluator") or leaf.evaluator),
        rolling_island_scene=rolling_island_scene,
        **recipe,
    )
    result.validate()
    return result


__all__ = [
    "IMPLEMENTED_REVIEW_VARIANTS",
    "PhysicalSurface",
    "SOURCE_MUJOCO_BACKEND_VERSION",
    "SOURCE_MUJOCO_COMPILED_SCHEMA",
    "SourceMujocoCompiledScenario",
    "SourceMujocoUnsupported",
    "compile_review_case",
]
