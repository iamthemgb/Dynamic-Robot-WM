"""Typed simulation-backend lifecycle.

The original family adapters combine planning and a lightweight diagnostic
simulation.  Production backends deliberately split those responsibilities:
an immutable :class:`ScenarioSpec` is converted to the existing stable
``EpisodePlan`` identity contract, then a backend compiles, simulates, and
renders that plan.  This module contains no MuJoCo import so inventory, schema,
and dry-run commands remain usable in base-only environments.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from enum import Enum
import math
from typing import Any, Mapping, Sequence

from ..common.cameras import CameraCalibration
from ..families.base import EpisodePlan, SimulationResult, stable_hash


NATIVE_SCENARIO_VERSION = "dynamic-robot-native-scenario/v1"


class NativeBackendError(RuntimeError):
    """A native backend could not compile or execute an episode."""


class NativeFamily(str, Enum):
    FALLING_CATCH = "falling_catch"
    ROLLING_INTERCEPTION = "rolling_interception"
    PROJECTILE_REBOUND = "projectile_rebound"


class RigidScenario(str, Enum):
    CENTERED_DROP = "centered_vertical_drop"
    OFF_CENTER_DROP = "off_center_drop"
    DRIFTED_DROP = "drifted_drop"
    CATCH_RETAIN = "catch_retain"
    CATCH_TRANSPORT = "catch_transport"
    CATCH_BRAKE = "catch_abrupt_brake"
    CATCH_TILT = "catch_tray_tilt"
    CATCH_EDGE_RECOVERY = "catch_edge_recovery"
    STRAIGHT_ROLL = "straight_rolling"
    STRAIGHT_SLIDE = "straight_sliding"
    ROLLING_SLIDING_TRANSITION = "rolling_sliding_transition"
    SMALL_SLOPE = "small_slope"
    RAMP_TO_TABLE = "ramp_to_table"
    ROLL_OFF_EDGE = "roll_off_table_edge"
    PADDLE_BLOCK = "paddle_block"
    REDIRECT_TO_TARGET = "redirect_to_target"
    CONTAINER_RECEIVE = "container_receive"
    OCCLUDED_INTERCEPTION = "occluded_interception"
    DIRECT_PROJECTILE = "direct_projectile"
    TABLE_BOUNCE = "table_bounce"
    WALL_REBOUND = "wall_rebound"
    ANGLED_BARRIER_REBOUND = "angled_barrier_rebound"
    RAMP_LAUNCH = "ramp_launch"
    PROJECTILE_ROLL_OFF_EDGE = "projectile_roll_off_edge"
    FLIGHT_TO_TABLE_BOUNCE = "flight_to_table_bounce"
    FLOOR_TO_WALL = "floor_to_wall"
    PADDLE_DEFLECTION = "paddle_deflection"
    BOUNCE_TO_INTERCEPTION = "bounce_to_robot_interception"


class RigidShape(str, Enum):
    SPHERE = "sphere"
    PUCK = "puck"
    CUBE = "cube"


class ToolKind(str, Enum):
    SHALLOW_TRAY = "shallow_tray"
    DEEP_TRAY = "deep_tray"
    SMALL_BIN = "small_bin"
    FLAT_PADDLE = "flat_paddle"
    ANGLED_PADDLE = "angled_paddle"


class IntendedBranch(str, Enum):
    SUCCESS = "success_seeking"
    NEAR_MISS = "near_miss"
    CONTACT_FAILURE = "contact_failure"
    NO_OP = "no_op"
    WRONG_ACTION = "wrong_action"


@dataclass(frozen=True)
class RigidObjectSpec:
    shape: RigidShape = RigidShape.SPHERE
    radius_m: float = 0.04
    half_extents_m: tuple[float, float, float] = (0.04, 0.04, 0.04)
    mass_kg: float = 0.075
    friction: tuple[float, float, float] = (0.45, 0.01, 0.001)
    effective_restitution_target: float = 0.2
    rgba: tuple[float, float, float, float] = (0.92, 0.25, 0.08, 1.0)
    density_kg_m3: float | None = None

    def validate(self) -> None:
        if self.radius_m <= 0 or self.mass_kg <= 0:
            raise ValueError("object radius and mass must be positive")
        if len(self.half_extents_m) != 3 or any(value <= 0 for value in self.half_extents_m):
            raise ValueError("object half extents must contain three positive values")
        if len(self.friction) != 3 or any(value < 0 for value in self.friction):
            raise ValueError("MuJoCo friction must contain three non-negative values")
        if not 0.0 <= self.effective_restitution_target <= 1.0:
            raise ValueError("effective restitution target must be in [0, 1]")
        if len(self.rgba) != 4 or any(not math.isfinite(value) for value in self.rgba):
            raise ValueError("object RGBA must contain four finite values")
        if self.density_kg_m3 is not None:
            if not math.isfinite(self.density_kg_m3) or self.density_kg_m3 <= 0:
                raise ValueError("object density must be finite and positive")
            if self.shape == RigidShape.SPHERE:
                volume = 4.0 / 3.0 * math.pi * self.radius_m**3
            elif self.shape == RigidShape.PUCK:
                volume = math.pi * self.half_extents_m[0] ** 2 * (2.0 * self.half_extents_m[2])
            else:
                volume = 8.0 * math.prod(self.half_extents_m)
            implied_mass = self.density_kg_m3 * volume
            if abs(implied_mass - self.mass_kg) > max(1e-8, 1e-6 * self.mass_kg):
                raise ValueError("object mass, geometry, and density are inconsistent")


@dataclass(frozen=True)
class InitialStateSpec:
    position_m: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    linear_velocity_m_s: tuple[float, float, float] = (0.0, 0.0, 0.0)
    angular_velocity_rad_s: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def validate(self) -> None:
        vectors: tuple[Sequence[float], ...] = (
            self.position_m,
            self.quaternion_wxyz,
            self.linear_velocity_m_s,
            self.angular_velocity_rad_s,
        )
        if any(any(not math.isfinite(float(value)) for value in vector) for vector in vectors):
            raise ValueError("initial state contains a non-finite value")
        if len(self.position_m) != 3 or len(self.quaternion_wxyz) != 4:
            raise ValueError("initial pose must use XYZ and WXYZ")
        if len(self.linear_velocity_m_s) != 3 or len(self.angular_velocity_rad_s) != 3:
            raise ValueError("initial velocity vectors must have length three")
        norm = math.sqrt(sum(float(value) ** 2 for value in self.quaternion_wxyz))
        if abs(norm - 1.0) > 1e-5:
            raise ValueError("initial quaternion must be normalized")


@dataclass(frozen=True)
class ToolSpec:
    kind: ToolKind
    start_position_m: tuple[float, float, float]
    half_extents_m: tuple[float, float, float] = (0.16, 0.16, 0.012)
    wall_height_m: float = 0.07
    rgba: tuple[float, float, float, float] = (0.12, 0.22, 0.32, 1.0)

    def validate(self) -> None:
        if len(self.start_position_m) != 3 or any(not math.isfinite(value) for value in self.start_position_m):
            raise ValueError("tool start position must contain three finite values")
        if len(self.half_extents_m) != 3 or any(value <= 0 for value in self.half_extents_m):
            raise ValueError("tool half extents must contain three positive values")
        if self.wall_height_m < 0:
            raise ValueError("tool wall height cannot be negative")


@dataclass(frozen=True)
class ControllerPhase:
    name: str
    start_s: float
    end_s: float
    target_position_m: tuple[float, float, float]
    target_rpy_rad: tuple[float, float, float] = (0.0, 0.0, 0.0)
    interpolation: str = "smoothstep"

    def validate(self) -> None:
        if not self.name or self.start_s < 0 or self.end_s <= self.start_s:
            raise ValueError("controller phases require a name and a positive interval")
        if self.interpolation not in {"hold", "linear", "smoothstep"}:
            raise ValueError(f"unsupported interpolation {self.interpolation!r}")
        if len(self.target_position_m) != 3 or len(self.target_rpy_rad) != 3:
            raise ValueError("controller targets must be XYZ and RPY triples")
        if any(
            not math.isfinite(value)
            for value in (*self.target_position_m, *self.target_rpy_rad)
        ):
            raise ValueError("controller targets must be finite")


@dataclass(frozen=True)
class CameraSpec:
    name: str
    position_m: tuple[float, float, float]
    look_at_m: tuple[float, float, float]
    up_world: tuple[float, float, float] = (0.0, 0.0, 1.0)
    fovy_deg: float = 48.0
    role: str = "main_three_quarter"

    def validate(self) -> None:
        if not self.name or not 10.0 <= self.fovy_deg <= 120.0:
            raise ValueError("camera needs a name and a plausible vertical field of view")
        if any(
            not math.isfinite(value)
            for value in (*self.position_m, *self.look_at_m, *self.up_world)
        ):
            raise ValueError("camera vectors must be finite")
        if math.dist(self.position_m, self.look_at_m) < 1e-6:
            raise ValueError("camera position and look-at point must differ")


@dataclass(frozen=True)
class PhysicsRangeProvenance:
    range_version: str = "native-rigid-ranges/v1"
    partition: str = "nominal"
    calibrated: bool = False
    calibration_artifact: str | None = None
    calibration_artifact_path: str | None = None

    def validate(self) -> None:
        if self.partition not in {"nominal", "train_id", "validation_id", "test_ood"}:
            raise ValueError(f"unknown physics partition {self.partition!r}")
        if self.calibrated and (
            not self.calibration_artifact or not self.calibration_artifact_path
        ):
            raise ValueError(
                "calibrated physics ranges require an artifact hash and resolvable path"
            )


@dataclass(frozen=True)
class ScenarioSpec:
    """Complete, immutable input to a native simulation backend."""

    scenario_id: str
    family: NativeFamily
    scenario: RigidScenario
    branch: IntendedBranch
    seed: int
    object: RigidObjectSpec
    initial_state: InitialStateSpec
    tool: ToolSpec
    phases: tuple[ControllerPhase, ...]
    cameras: tuple[CameraSpec, CameraSpec]
    scene_style: str = "clean_franka_lab"
    gravity_m_s2: tuple[float, float, float] = (0.0, 0.0, -9.81)
    surface_friction: tuple[float, float, float] = (0.55, 0.01, 0.001)
    sim_hz: int = 240
    control_hz: int = 60
    video_hz: int = 30
    maximum_duration_s: float = 4.0
    minimum_terminal_context_s: float = 0.5
    controller_latency_s: float = 0.0
    camera_latency_s: float = 0.0
    robot_start_joint_offsets_rad: tuple[float, ...] = (0.0,) * 7
    robot_model: str = "franka_panda"
    tool_calibration_id: str = "procedural_tool_v1"
    physics_provenance: PhysicsRangeProvenance = field(default_factory=PhysicsRangeProvenance)
    expected_contact_sequence: tuple[str, ...] = ()
    max_task_contacts: int = 2
    counterfactual_bundle_id: str | None = None
    physics_counterfactual_family_id: str | None = None
    split_group_id: str | None = None
    extras: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = NATIVE_SCENARIO_VERSION

    def validate(self) -> None:
        if self.schema_version != NATIVE_SCENARIO_VERSION:
            raise ValueError(f"unsupported native scenario schema {self.schema_version!r}")
        if not self.scenario_id or not self.scene_style:
            raise ValueError("scenario_id and scene_style are required")
        if self.sim_hz <= 0 or self.control_hz <= 0 or self.video_hz <= 0:
            raise ValueError("simulation, control, and video rates must be positive")
        if self.sim_hz % self.control_hz or self.sim_hz % self.video_hz:
            raise ValueError("sim_hz must be divisible by control_hz and video_hz")
        if self.maximum_duration_s <= 0 or self.minimum_terminal_context_s < 0:
            raise ValueError("duration values must be non-negative and maximum positive")
        if self.controller_latency_s < 0 or self.camera_latency_s < 0:
            raise ValueError("controller and camera latency cannot be negative")
        if self.camera_latency_s * self.video_hz > 1.0 + 1e-9:
            raise ValueError("camera latency cannot exceed one encoded frame")
        if (
            len(self.robot_start_joint_offsets_rad) != 7
            or any(
                not math.isfinite(value) or abs(value) > 0.1
                for value in self.robot_start_joint_offsets_rad
            )
        ):
            raise ValueError(
                "robot_start_joint_offsets_rad must contain seven finite values within +/-0.1 rad"
            )
        if len(self.gravity_m_s2) != 3 or any(not math.isfinite(value) for value in self.gravity_m_s2):
            raise ValueError("gravity must be a finite XYZ vector")
        if len(self.surface_friction) != 3 or any(
            not math.isfinite(value) or value < 0 for value in self.surface_friction
        ):
            raise ValueError("surface friction must contain three finite non-negative values")
        if self.family == NativeFamily.PROJECTILE_REBOUND:
            if self.max_task_contacts not in {1, 2}:
                raise ValueError(
                    "production projectile/rebound scenarios permit one or two task contacts"
                )
        elif self.max_task_contacts not in {1, 2, 3}:
            raise ValueError(
                "production falling/rolling scenarios permit at most three declared task contacts"
            )
        expected_family = {
            RigidScenario.CENTERED_DROP: NativeFamily.FALLING_CATCH,
            RigidScenario.OFF_CENTER_DROP: NativeFamily.FALLING_CATCH,
            RigidScenario.DRIFTED_DROP: NativeFamily.FALLING_CATCH,
            RigidScenario.CATCH_RETAIN: NativeFamily.FALLING_CATCH,
            RigidScenario.CATCH_TRANSPORT: NativeFamily.FALLING_CATCH,
            RigidScenario.CATCH_BRAKE: NativeFamily.FALLING_CATCH,
            RigidScenario.CATCH_TILT: NativeFamily.FALLING_CATCH,
            RigidScenario.CATCH_EDGE_RECOVERY: NativeFamily.FALLING_CATCH,
            RigidScenario.STRAIGHT_ROLL: NativeFamily.ROLLING_INTERCEPTION,
            RigidScenario.STRAIGHT_SLIDE: NativeFamily.ROLLING_INTERCEPTION,
            RigidScenario.ROLLING_SLIDING_TRANSITION: NativeFamily.ROLLING_INTERCEPTION,
            RigidScenario.SMALL_SLOPE: NativeFamily.ROLLING_INTERCEPTION,
            RigidScenario.RAMP_TO_TABLE: NativeFamily.ROLLING_INTERCEPTION,
            RigidScenario.ROLL_OFF_EDGE: NativeFamily.ROLLING_INTERCEPTION,
            RigidScenario.PADDLE_BLOCK: NativeFamily.ROLLING_INTERCEPTION,
            RigidScenario.REDIRECT_TO_TARGET: NativeFamily.ROLLING_INTERCEPTION,
            RigidScenario.CONTAINER_RECEIVE: NativeFamily.ROLLING_INTERCEPTION,
            RigidScenario.OCCLUDED_INTERCEPTION: NativeFamily.ROLLING_INTERCEPTION,
            RigidScenario.DIRECT_PROJECTILE: NativeFamily.PROJECTILE_REBOUND,
            RigidScenario.TABLE_BOUNCE: NativeFamily.PROJECTILE_REBOUND,
            RigidScenario.WALL_REBOUND: NativeFamily.PROJECTILE_REBOUND,
            RigidScenario.ANGLED_BARRIER_REBOUND: NativeFamily.PROJECTILE_REBOUND,
            RigidScenario.RAMP_LAUNCH: NativeFamily.PROJECTILE_REBOUND,
            RigidScenario.PROJECTILE_ROLL_OFF_EDGE: NativeFamily.PROJECTILE_REBOUND,
            RigidScenario.FLIGHT_TO_TABLE_BOUNCE: NativeFamily.PROJECTILE_REBOUND,
            RigidScenario.FLOOR_TO_WALL: NativeFamily.PROJECTILE_REBOUND,
            RigidScenario.PADDLE_DEFLECTION: NativeFamily.PROJECTILE_REBOUND,
            RigidScenario.BOUNCE_TO_INTERCEPTION: NativeFamily.PROJECTILE_REBOUND,
        }[self.scenario]
        if expected_family != self.family:
            raise ValueError(f"scenario {self.scenario.value} belongs to {expected_family.value}")
        self.object.validate()
        self.initial_state.validate()
        self.tool.validate()
        self.physics_provenance.validate()
        if len(self.cameras) != 2 or {camera.name for camera in self.cameras} != {"main", "secondary"}:
            raise ValueError("native episodes require main and secondary cameras")
        for camera in self.cameras:
            camera.validate()
        previous_end = -math.inf
        for phase in self.phases:
            phase.validate()
            if phase.start_s < previous_end - 1e-9:
                raise ValueError("controller phases cannot overlap")
            previous_end = phase.end_s
        if self.phases and self.phases[-1].end_s > self.maximum_duration_s + 1e-9:
            raise ValueError("controller phase extends beyond maximum duration")

    def to_dict(self) -> dict[str, Any]:
        self.validate()

        def convert(value: Any) -> Any:
            if isinstance(value, Enum):
                return value.value
            if isinstance(value, Mapping):
                return {str(key): convert(item) for key, item in value.items()}
            if isinstance(value, (tuple, list)):
                return [convert(item) for item in value]
            return value

        return convert(asdict(self))

    @property
    def spec_hash(self) -> str:
        return stable_hash(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ScenarioSpec":
        data = dict(value)
        data["family"] = NativeFamily(data["family"])
        data["scenario"] = RigidScenario(data["scenario"])
        data["branch"] = IntendedBranch(data["branch"])
        object_data = dict(data["object"])
        object_data["shape"] = RigidShape(object_data["shape"])
        for name in ("half_extents_m", "friction", "rgba"):
            object_data[name] = tuple(object_data[name])
        data["object"] = RigidObjectSpec(**object_data)
        initial = dict(data["initial_state"])
        for name in (
            "position_m",
            "quaternion_wxyz",
            "linear_velocity_m_s",
            "angular_velocity_rad_s",
        ):
            initial[name] = tuple(initial[name])
        data["initial_state"] = InitialStateSpec(**initial)
        tool = dict(data["tool"])
        tool["kind"] = ToolKind(tool["kind"])
        for name in ("start_position_m", "half_extents_m", "rgba"):
            tool[name] = tuple(tool[name])
        data["tool"] = ToolSpec(**tool)
        phases = []
        for raw in data.get("phases", ()):
            item = dict(raw)
            item["target_position_m"] = tuple(item["target_position_m"])
            item["target_rpy_rad"] = tuple(item.get("target_rpy_rad", (0.0, 0.0, 0.0)))
            phases.append(ControllerPhase(**item))
        data["phases"] = tuple(phases)
        cameras = []
        for raw in data["cameras"]:
            item = dict(raw)
            for name in ("position_m", "look_at_m", "up_world"):
                item[name] = tuple(item[name])
            cameras.append(CameraSpec(**item))
        data["cameras"] = tuple(cameras)
        physics = dict(data.get("physics_provenance") or {})
        data["physics_provenance"] = PhysicsRangeProvenance(**physics)
        data["gravity_m_s2"] = tuple(data.get("gravity_m_s2", (0.0, 0.0, -9.81)))
        data["surface_friction"] = tuple(
            data.get("surface_friction", (0.55, 0.01, 0.001))
        )
        data["robot_start_joint_offsets_rad"] = tuple(
            data.get("robot_start_joint_offsets_rad", (0.0,) * 7)
        )
        data["expected_contact_sequence"] = tuple(data.get("expected_contact_sequence", ()))
        data["extras"] = dict(data.get("extras") or {})
        spec = cls(**data)
        spec.validate()
        return spec


@dataclass(frozen=True)
class NativeEpisodePlan:
    """Backend compilation unit retaining the canonical episode identity."""

    episode_plan: EpisodePlan
    scenario: ScenarioSpec
    compiled_scenario_hash: str
    fixed_field_hash: str
    action_replay_hash: str

    def __post_init__(self) -> None:
        self.scenario.validate()
        if self.episode_plan.episode_uuid == "":
            raise ValueError("episode plan identity is empty")
        if self.compiled_scenario_hash != self.scenario.spec_hash:
            raise ValueError("compiled scenario hash does not match the scenario")


@dataclass
class BackendRunResult:
    """Native simulation plus synchronized render/write sidecars."""

    simulation: SimulationResult
    frames_by_camera: Mapping[str, Sequence[Any]]
    camera_calibrations: Mapping[str, CameraCalibration]
    frame_rows: Sequence[Mapping[str, Any]]
    high_rate_rows: Sequence[Mapping[str, Any]]
    event_rows: Sequence[Mapping[str, Any]]
    object_state_rows: Sequence[Mapping[str, Any]]
    transition_events: Sequence[Mapping[str, Any]]
    backend_provenance: Mapping[str, Any]
    quality_flags: tuple[str, ...] = ()
    visibility_qc: Mapping[str, Any] = field(default_factory=dict)

    @property
    def production_eligible(self) -> bool:
        return bool(self.backend_provenance.get("production_eligible", False)) and not self.quality_flags

    def as_renderer_payload(self) -> dict[str, Any]:
        """Return the mapping consumed by the existing atomic writer path."""
        metrics = dict(self.simulation.outcome.metrics)
        task_events = [
            row
            for row in self.event_rows
            if str(row.get("contact_role", "")) == "task_contact"
        ]
        if task_events:
            key_event_time_s = min(float(row["timestamp"]) for row in task_events)
            key_event_name = "first_task_contact"
            key_event_semantics = "first_non_fixture_task_contact"
        elif self.event_rows:
            first_event = min(self.event_rows, key=lambda row: float(row["timestamp"]))
            key_event_time_s = float(first_event["timestamp"])
            key_event_name = f"first_{first_event.get('object_b', 'physics')}_contact"
            key_event_semantics = "first_saved_physics_contact"
        else:
            key_event_time_s = (
                float(self.frame_rows[-1]["timestamp"]) if self.frame_rows else 0.0
            )
            key_event_name = "terminal_observation"
            key_event_semantics = "terminal_observation_without_contact"
        return {
            "videos": dict(self.frames_by_camera),
            "cameras": dict(self.camera_calibrations),
            "frame_rows": list(self.frame_rows),
            "high_rate_rows": list(self.high_rate_rows),
            "event_rows": list(self.event_rows),
            "object_state_rows": list(self.object_state_rows),
            "transition_rows": list(self.transition_events),
            "renderer": str(self.backend_provenance.get("renderer", "mujoco.Renderer")),
            "backend_provenance": dict(self.backend_provenance),
            "production_eligible": self.production_eligible,
            "quality_flags": list(self.quality_flags),
            "visibility_qc": dict(self.visibility_qc),
            "task_event_time_s": (
                key_event_time_s if key_event_semantics == "first_non_fixture_task_contact" else None
            ),
            "key_event_time_s": key_event_time_s,
            "key_event_name": key_event_name,
            "key_event_semantics": key_event_semantics,
            "objective_evaluator_id": metrics.get(
                "objective_evaluator_id", "native_rigid_state_event"
            ),
            "objective_evaluator_version": metrics.get(
                "objective_evaluator_version", "1.2.0"
            ),
            "objective_threshold_set_hash": metrics.get(
                "objective_threshold_set_hash", "unknown"
            ),
            "objective_evidence": {
                "stored_objective_success": self.simulation.outcome.task_success,
                "independently_recomputed": True,
                "source": "registered_persisted_state_event_evaluator",
                "frame_count": len(self.frame_rows),
                "event_count": len(self.event_rows),
            },
        }


class SimulationBackend(ABC):
    """Interface implemented by integrated native simulation/render backends."""

    name: str
    version: str

    @abstractmethod
    def compile(self, plan: EpisodePlan | NativeEpisodePlan) -> NativeEpisodePlan:
        raise NotImplementedError

    @abstractmethod
    def run(self, plan: EpisodePlan | NativeEpisodePlan, *, render: bool = True) -> BackendRunResult:
        raise NotImplementedError
