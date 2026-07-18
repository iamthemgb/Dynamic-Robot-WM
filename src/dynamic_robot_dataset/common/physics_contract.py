"""Strict, backend-independent physics acceptance contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Any, Mapping, Sequence


STRICT_RIGID_QC_SCHEMA = "dynamic-robot-strict-rigid-qc/v1"


@dataclass(frozen=True, slots=True)
class StrictRigidThresholds:
    maximum_gripper_penetration_m: float = 0.002
    maximum_task_surface_penetration_m: float = 0.003
    maximum_effective_restitution: float = 1.05
    maximum_free_flight_energy_drift_fraction: float = 0.05
    maximum_timestep_event_shift_s: float = 1.0 / 30.0
    maximum_timestep_key_position_shift_m: float = 0.01

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


STRICT_RIGID_THRESHOLDS = StrictRigidThresholds()


class ContactClass(str, Enum):
    GRIPPER = "gripper"
    TASK_SURFACE = "task_surface"
    BACKGROUND = "background"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class StrictPhysicsResult:
    passed: bool
    failures: tuple[str, ...]
    metrics: Mapping[str, Any]
    schema_version: str = STRICT_RIGID_QC_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_GRIPPER_TOKENS = (
    "finger",
    "fingertip",
    "gripper",
    "robotiq",
    "panda_hand",
    "hand_collision",
    "pad",
    "left_jaw",
    "right_jaw",
)
_TASK_SURFACE_TOKENS = (
    "table",
    "surface",
    "ramp",
    "wall",
    "barrier",
    "platform",
    "floor",
    "fixture",
    "bin",
    "container",
    "support",
    "edge",
)


def classify_contact(row: Mapping[str, Any]) -> ContactClass:
    """Classify an object contact using explicit canonical role first."""

    explicit = str(
        row.get("contact_category")
        or row.get("contact.category")
        or row.get("counterpart_category")
        or ""
    ).strip().lower()
    aliases = {
        "robot_tool": ContactClass.GRIPPER,
        "robot_gripper": ContactClass.GRIPPER,
        "gripper": ContactClass.GRIPPER,
        # Robot-link impacts are physical, classified robot contacts.  They
        # use the stricter 2 mm object/robot penetration limit rather than
        # being silently reported as unknown contacts.
        "robot_arm": ContactClass.GRIPPER,
        "task_surface": ContactClass.TASK_SURFACE,
        "fixture": ContactClass.TASK_SURFACE,
        "support": ContactClass.TASK_SURFACE,
        "background": ContactClass.BACKGROUND,
        "visual_background": ContactClass.BACKGROUND,
    }
    if explicit in aliases:
        return aliases[explicit]
    pair_text = " ".join(
        str(row.get(name) or "").lower()
        for name in ("object_a", "object_b", "geom_a", "geom_b", "counterpart")
    )
    if any(token in pair_text for token in _GRIPPER_TOKENS):
        return ContactClass.GRIPPER
    if any(token in pair_text for token in _TASK_SURFACE_TOKENS):
        return ContactClass.TASK_SURFACE
    return ContactClass.UNKNOWN


def strict_contact_penetration_check(
    events: Sequence[Mapping[str, Any]],
    *,
    thresholds: StrictRigidThresholds = STRICT_RIGID_THRESHOLDS,
    require_classification: bool = True,
) -> StrictPhysicsResult:
    """Apply separate gripper and physical-fixture penetration limits."""

    maxima = {
        ContactClass.GRIPPER: 0.0,
        ContactClass.TASK_SURFACE: 0.0,
        ContactClass.BACKGROUND: 0.0,
        ContactClass.UNKNOWN: 0.0,
    }
    counts = {value: 0 for value in ContactClass}
    failures: list[str] = []
    for index, event in enumerate(events):
        category = classify_contact(event)
        counts[category] += 1
        if "penetration_depth_m" not in event:
            failures.append(f"contact {index} lacks measured penetration depth")
            continue
        try:
            depth = float(event["penetration_depth_m"])
        except (TypeError, ValueError):
            failures.append(f"contact {index} has non-numeric penetration depth")
            continue
        if not math.isfinite(depth) or depth < 0:
            failures.append(f"contact {index} has invalid penetration depth")
            continue
        maxima[category] = max(maxima[category], depth)
    if maxima[ContactClass.GRIPPER] > thresholds.maximum_gripper_penetration_m:
        failures.append(
            "object-gripper penetration exceeds "
            f"{thresholds.maximum_gripper_penetration_m:g} m"
        )
    if maxima[ContactClass.TASK_SURFACE] > thresholds.maximum_task_surface_penetration_m:
        failures.append(
            "object-task-surface penetration exceeds "
            f"{thresholds.maximum_task_surface_penetration_m:g} m"
        )
    if counts[ContactClass.BACKGROUND]:
        failures.append("task object contacted a visual-only background asset")
    if require_classification and counts[ContactClass.UNKNOWN]:
        failures.append(
            f"{counts[ContactClass.UNKNOWN]} contact(s) lack gripper/task-surface classification"
        )
    metrics = {
        "maximum_gripper_penetration_m": maxima[ContactClass.GRIPPER],
        "maximum_task_surface_penetration_m": maxima[ContactClass.TASK_SURFACE],
        "maximum_background_penetration_m": maxima[ContactClass.BACKGROUND],
        "classified_gripper_contact_count": counts[ContactClass.GRIPPER],
        "classified_task_surface_contact_count": counts[ContactClass.TASK_SURFACE],
        "background_contact_count": counts[ContactClass.BACKGROUND],
        "unclassified_contact_count": counts[ContactClass.UNKNOWN],
        "thresholds": thresholds.to_dict(),
    }
    return StrictPhysicsResult(not failures, tuple(failures), metrics)


_REQUIRED_RUNTIME_ZERO_FIELDS = (
    "object_state_writes_after_initialization",
    "direct_robot_state_writes_after_initialization",
    "mocap_writes_after_initialization",
    "applied_force_writes_after_initialization",
    "object_linked_equality_changes_after_initialization",
    "model_physics_mutations_after_initialization",
    "mutation_boundary_violations",
    "solver_warning_count",
    "non_finite_state_count",
    "tunneling_event_count",
    "unexplained_velocity_discontinuity_count",
)


def strict_runtime_audit_failures(
    audit: Mapping[str, Any],
    *,
    require_control_updates: bool = True,
) -> list[str]:
    """Require explicit zero evidence for every forbidden runtime mechanism."""

    failures: list[str] = []
    for name in _REQUIRED_RUNTIME_ZERO_FIELDS:
        if name not in audit:
            failures.append(f"runtime audit lacks {name}")
            continue
        try:
            value = int(audit[name])
        except (TypeError, ValueError):
            failures.append(f"runtime audit {name} is not an integer count")
            continue
        if value != 0:
            failures.append(f"runtime audit has nonzero {name}: {value}")
    try:
        updates = int(audit.get("control_updates", 0))
    except (TypeError, ValueError):
        updates = 0
    if require_control_updates and updates <= 0:
        failures.append("runtime audit has no applied actuator-control updates")
    return failures


def strict_persisted_physics_failures(
    physics_qc: Mapping[str, Any],
    *,
    thresholds: StrictRigidThresholds = STRICT_RIGID_THRESHOLDS,
) -> list[str]:
    """Validate common numerical evidence emitted by any rigid backend."""

    failures: list[str] = []
    checks = physics_qc.get("checks")
    if not isinstance(checks, Mapping):
        return ["strict physics QC lacks a checks mapping"]
    required_true = (
        "finite_state",
        "no_solver_warnings",
        "no_tunneling",
        "no_unexplained_velocity_discontinuity",
        "no_mutation_boundary_violation",
        "no_applied_forces",
        "no_object_linked_equality_or_latch_assistance",
        "actuator_forces_within_model_limits",
        "joint_motion_within_model_limits",
    )
    for name in required_true:
        if checks.get(name) is not True:
            failures.append(f"strict physics check is absent or false: {name}")

    def finite_metric(name: str) -> float | None:
        value = checks.get(name, physics_qc.get(name))
        try:
            number = float(value)
        except (TypeError, ValueError):
            failures.append(f"strict physics metric is missing or non-numeric: {name}")
            return None
        if not math.isfinite(number):
            failures.append(f"strict physics metric is non-finite: {name}")
            return None
        return number

    energy_applicable = checks.get("free_flight_energy_applicable")
    if not isinstance(energy_applicable, bool):
        failures.append("strict physics applicability is missing: free_flight_energy_applicable")
    elif energy_applicable:
        energy = finite_metric("free_flight_relative_energy_drift")
        if (
            energy is not None
            and abs(energy) > thresholds.maximum_free_flight_energy_drift_fraction
        ):
            failures.append(
                "free-flight energy drift exceeds "
                f"{thresholds.maximum_free_flight_energy_drift_fraction:.1%}"
            )
    restitution_applicable = checks.get("static_restitution_applicable")
    if not isinstance(restitution_applicable, bool):
        failures.append("strict physics applicability is missing: static_restitution_applicable")
    elif restitution_applicable:
        restitution = finite_metric("maximum_measured_effective_restitution")
        if restitution is not None:
            if restitution < 0.0:
                failures.append("measured effective restitution is negative")
            elif restitution > thresholds.maximum_effective_restitution:
                failures.append(
                    "measured effective restitution exceeds "
                    f"{thresholds.maximum_effective_restitution:g}"
                )
    if physics_qc.get("physics_qc_pass") is not True:
        failures.append("backend physics_qc_pass is not true")
    return failures


def timestep_halving_failures(
    coarse: Mapping[str, Any],
    fine: Mapping[str, Any],
    *,
    thresholds: StrictRigidThresholds = STRICT_RIGID_THRESHOLDS,
) -> list[str]:
    """Compare a candidate rigid timestep with its halved-timestep replay."""

    failures: list[str] = []
    if coarse.get("outcome") != fine.get("outcome"):
        failures.append("timestep halving changes the measured outcome")
    try:
        event_shift = abs(float(coarse["key_event_time_s"]) - float(fine["key_event_time_s"]))
    except (KeyError, TypeError, ValueError):
        failures.append("timestep comparison lacks finite key-event times")
    else:
        if not math.isfinite(event_shift) or event_shift > thresholds.maximum_timestep_event_shift_s:
            failures.append("timestep key-event shift exceeds one video frame")
    try:
        coarse_position = tuple(float(value) for value in coarse["key_event_position_m"])
        fine_position = tuple(float(value) for value in fine["key_event_position_m"])
        position_shift = math.dist(coarse_position, fine_position)
    except (KeyError, TypeError, ValueError):
        failures.append("timestep comparison lacks finite key-event positions")
    else:
        if (
            len(coarse_position) != 3
            or len(fine_position) != 3
            or not math.isfinite(position_shift)
            or position_shift > thresholds.maximum_timestep_key_position_shift_m
        ):
            failures.append("timestep key-event position shift exceeds 1 cm")
    return failures


def deformable_release_failures(evidence: Mapping[str, Any]) -> list[str]:
    """Fail closed until cloth/rope use physical grasps and replayable checks."""

    required_true = (
        "native_deformable_physics",
        "frictional_finger_contact_grasp",
        "no_equality_or_connect_proxy_attachment",
        "strain_within_declared_limits",
        "segment_length_consistent",
        "self_contact_and_intersection_valid",
        "grasp_slip_bounded",
        "saved_artifact_objective_replay_matches",
    )
    return [
        f"deformable release evidence is absent or false: {name}"
        for name in required_true
        if evidence.get(name) is not True
    ]


def fluid_release_failures(evidence: Mapping[str, Any]) -> list[str]:
    """Reject pouring-emitter and visual-particle stand-ins for F3c current pickup."""

    required_true = (
        "particle_fluid_solver_active",
        "fluid_mass_and_particle_conservation_pass",
        "current_field_measured",
        "object_drag_and_buoyancy_solver_derived",
        "no_scripted_object_motion",
        "no_pouring_emitter_proxy",
        "no_visual_particle_fallback",
        "saved_artifact_objective_replay_matches",
    )
    return [
        f"fluid release evidence is absent or false: {name}"
        for name in required_true
        if evidence.get(name) is not True
    ]


def rigid_task_evidence_failures(
    *,
    family: str,
    subfamily: str,
    task_variant: str,
    evidence: Mapping[str, Any],
    task_success: bool | None = None,
) -> list[str]:
    """Validate family-specific evidence against the *measured* outcome.

    Negative branches are valid data.  They must carry replayable evidence that
    the persisted evaluator measured the same failure; they must not be forced
    to exhibit the bilateral retention evidence required of a successful catch.
    Branch intent is deliberately not consulted here.
    """

    identity = " ".join((family, subfamily, task_variant)).lower()
    motion_identity = " ".join((subfamily, task_variant)).lower()
    required: list[str] = []
    if "catch" in identity or family == "falling_catch":
        if task_success is False:
            required.extend(
                (
                    "measured_failure_matches_persisted_label",
                    "saved_artifact_objective_replay_matches",
                )
            )
        else:
            required.extend(
                (
                    "sustained_opposing_bilateral_contacts",
                    "stable_object_to_grasp_transform",
                )
            )
            if "transport" in identity or "handoff" in identity or "pickup" in identity:
                required.append("displacement_physically_supported_by_contacts")
    if family == "rolling_interception" or any(
        token in motion_identity for token in ("rolling", "sliding")
    ):
        required.extend(
            ("rolling_or_sliding_slip_within_limit", "friction_deceleration_consistent")
        )
    if any(token in motion_identity for token in ("bounce", "rebound")):
        required.extend(
            (
                "separated_pre_post_contact_samples",
                "measured_contact_normal",
                "effective_restitution_within_limit",
                "no_unexplained_contact_energy_gain",
            )
        )
    if any(token in identity for token in ("ramp", "edge")):
        required.extend(
            (
                "ordered_support_contact_termination",
                "post_support_ballistic_motion",
            )
        )
    if "deflect" in identity or "redirect" in identity:
        required.append("velocity_change_matches_measured_contact_impulse")
    if not required:
        required.append("family_specific_physics_evaluator_passed")
    return [
        f"rigid task evidence is absent or false: {name}"
        for name in dict.fromkeys(required)
        if evidence.get(name) is not True
    ]


__all__ = [
    "ContactClass",
    "STRICT_RIGID_QC_SCHEMA",
    "STRICT_RIGID_THRESHOLDS",
    "StrictPhysicsResult",
    "StrictRigidThresholds",
    "classify_contact",
    "deformable_release_failures",
    "fluid_release_failures",
    "rigid_task_evidence_failures",
    "strict_contact_penetration_check",
    "strict_persisted_physics_failures",
    "strict_runtime_audit_failures",
    "timestep_halving_failures",
]
