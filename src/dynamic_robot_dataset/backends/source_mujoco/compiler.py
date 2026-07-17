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


SOURCE_MUJOCO_COMPILED_SCHEMA = "dynamic-robot-source-mujoco-compiled/v1"
SOURCE_MUJOCO_BACKEND_VERSION = "0.4.0-review"


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
    "F2a": ("direct_catch",),
    "F2d": ("wall_rebound",),
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

    def validate(self) -> None:
        if not self.name or self.role not in {"floor", "table", "wall", "ramp", "slope"}:
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
        if self.corpus_leaf_id.startswith("F3"):
            raise SourceMujocoUnsupported("F3 scenes are not implemented by this rigid review backend")
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


def _surface(
    name: str,
    role: str,
    position: Sequence[float],
    half_size: Sequence[float],
    *,
    euler: Sequence[float] = (0.0, 0.0, 0.0),
) -> PhysicalSurface:
    solref = (
        RIGID_REVIEW_PROFILE.wall_solref
        if role == "wall"
        else (0.003, 1.0)
    )
    return PhysicalSurface(
        name=name,
        role=role,
        position_m=_finite_tuple(position, 3, "surface position"),  # type: ignore[arg-type]
        half_size_m=_finite_tuple(half_size, 3, "surface size"),  # type: ignore[arg-type]
        euler_rad=_finite_tuple(euler, 3, "surface rotation"),  # type: ignore[arg-type]
        solref=solref,
    )


def _recipe(
    leaf_id: str,
    task_variant: str,
    embodiment: str,
    branch_role: str,
    *,
    seed: int,
    tabletop_height_m: float,
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
        lateral = 0.28 if task_variant == "lateral_freefall" else 0.0
        base.update(
            duration_s=0.8,
            key_event_time_s=0.3,
            motion_kind="passive_freefall",
            object_initial_position_m=(-0.15, -0.10, tabletop_height_m + 0.47),
            object_initial_linear_velocity_m_s=(lateral, 0.0, 0.0),
            surfaces=(_surface("supported_floor", "floor", (0, 0, tabletop_height_m - 0.04), (1.4, 1.4, 0.04)),),
        )
    elif leaf_id == "P0b":
        vy = 0.34 if task_variant == "angled_projectile" else 0.0
        base.update(
            duration_s=0.8,
            key_event_time_s=0.4,
            motion_kind="passive_projectile",
            object_initial_position_m=(-0.70, -0.15, tabletop_height_m + 0.50),
            object_initial_linear_velocity_m_s=(1.15, vy, 3.50),
            surfaces=(_surface("supported_floor", "floor", (0, 0, tabletop_height_m - 0.04), (1.4, 1.4, 0.04)),),
        )
    elif leaf_id == "P0c" and task_variant == "table_bounce":
        base.update(
            duration_s=1.3,
            key_event_time_s=0.31,
            motion_kind="passive_table_bounce",
            object_initial_position_m=(-0.30, 0.0, tabletop_height_m + 0.75),
            object_initial_linear_velocity_m_s=(0.55, 0.0, -0.85),
            surfaces=(_surface("supported_bounce_table", "table", (0, 0, tabletop_height_m - 0.04), (1.2, 0.8, 0.04)),),
        )
    elif leaf_id == "P0c" and task_variant == "wall_rebound":
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
                _surface("supported_floor", "floor", (0, 0, tabletop_height_m - 0.04), (1.4, 1.4, 0.04)),
                _surface("supported_wall", "wall", (0.35, 0.0, tabletop_height_m + 0.62), (0.02, 0.65, 0.62)),
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
                surface.solref[0] * (1.15 if contact_factor < 1.0 else 0.85),
                surface.solref[1],
            ),
        )
        for surface in result["surfaces"]
    )
    return result


def compile_review_case(
    case: Mapping[str, Any] | Any,
) -> SourceMujocoCompiledScenario:
    """Compile one fixed review case without making a release claim."""

    value = _case_mapping(case)
    leaf_id = str(value.get("corpus_leaf_id") or "")
    task_variant = str(value.get("task_variant") or "")
    embodiment = str(value.get("embodiment") or "")
    if leaf_id.startswith("F3"):
        raise SourceMujocoUnsupported(
            f"{leaf_id} requires a dedicated dynamic-handoff/fluid implementation"
        )
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
    tabletop_height = 0.74 if requires_real_robocasa else 0.0
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
        tabletop_height_m=tabletop_height,
    )
    if leaf_id.startswith("P0"):
        recipe = _apply_passive_variation(
            recipe,
            leaf_id=leaf_id,
            task_variant=task_variant,
            profile=str(passive_variation_profile),
        )
    # The 600 Hz candidate is admitted only for fixed cases that preserve the
    # strict contact/rebound result at the 1200 Hz reference.  These cases
    # demonstrably do not: fast P0c speed variants lose rebound separation or
    # exceed 3 mm, and the Panda F1a/F1b negative-timing drops reach the room
    # floor at about 5 m/s.  Compile those cases directly at the calibrated
    # reference rate instead of weakening QC.
    requires_reference_rate = (
        leaf_id == "P0c"
        and passive_variation_profile
        in {"lower_initial_speed", "higher_initial_speed"}
    ) or (
        leaf_id in {"F1a", "F1b"}
        and embodiment == FRANKA_HAND
        and "controller" in branch_role
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
