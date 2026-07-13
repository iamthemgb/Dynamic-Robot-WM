"""Deterministic native scenario factories and EpisodePlan conversion."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import random
from typing import Any, Mapping, Sequence

from ..base import (
    CameraSpec,
    ControllerPhase,
    InitialStateSpec,
    IntendedBranch,
    NativeEpisodePlan,
    NativeFamily,
    PhysicsRangeProvenance,
    RigidObjectSpec,
    RigidScenario,
    RigidShape,
    ScenarioSpec,
    ToolKind,
    ToolSpec,
)
from ...families.base import (
    SCHEMA_VERSION as FAMILY_SCHEMA_VERSION,
    EpisodePlan,
    deterministic_seed,
    deterministic_uuid,
    physics_field,
    stable_hash,
)
from .model import (
    OBJECT_CONTACT_PRIORITY,
    RESTITUTION_SOLVER_PROFILE_VERSION,
    contact_damping_for_target,
)


_FALLING_SCENARIOS = {
    RigidScenario.CENTERED_DROP,
    RigidScenario.OFF_CENTER_DROP,
    RigidScenario.DRIFTED_DROP,
    RigidScenario.CATCH_RETAIN,
    RigidScenario.CATCH_TRANSPORT,
    RigidScenario.CATCH_BRAKE,
    RigidScenario.CATCH_TILT,
    RigidScenario.CATCH_EDGE_RECOVERY,
}
_ROLLING_SCENARIOS = {
    RigidScenario.STRAIGHT_ROLL,
    RigidScenario.STRAIGHT_SLIDE,
    RigidScenario.ROLLING_SLIDING_TRANSITION,
    RigidScenario.SMALL_SLOPE,
    RigidScenario.RAMP_TO_TABLE,
    RigidScenario.ROLL_OFF_EDGE,
    RigidScenario.PADDLE_BLOCK,
    RigidScenario.REDIRECT_TO_TARGET,
    RigidScenario.CONTAINER_RECEIVE,
    RigidScenario.OCCLUDED_INTERCEPTION,
}


def scenario_family(scenario: RigidScenario) -> NativeFamily:
    if scenario in _FALLING_SCENARIOS:
        return NativeFamily.FALLING_CATCH
    if scenario in _ROLLING_SCENARIOS:
        return NativeFamily.ROLLING_INTERCEPTION
    return NativeFamily.PROJECTILE_REBOUND


@dataclass(frozen=True)
class ScenarioAlias:
    scenario: RigidScenario
    shape: RigidShape = RigidShape.SPHERE
    direction: int | None = None
    tool_kind: ToolKind | None = None


_SUBFAMILY_ALIASES: dict[tuple[str, str], ScenarioAlias] = {
    ("falling_catch", "default"): ScenarioAlias(RigidScenario.CENTERED_DROP),
    ("falling_catch", "centered_vertical_drop"): ScenarioAlias(RigidScenario.CENTERED_DROP),
    ("falling_catch", "off_center_drop"): ScenarioAlias(RigidScenario.OFF_CENTER_DROP),
    ("falling_catch", "drifted_drop"): ScenarioAlias(RigidScenario.DRIFTED_DROP),
    ("falling_catch", "direct_projectile"): ScenarioAlias(RigidScenario.DRIFTED_DROP),
    ("falling_catch", "catch_retain"): ScenarioAlias(RigidScenario.CATCH_RETAIN),
    ("falling_catch", "catch_transport"): ScenarioAlias(RigidScenario.CATCH_TRANSPORT),
    ("falling_catch", "catch_brake"): ScenarioAlias(RigidScenario.CATCH_BRAKE),
    ("falling_catch", "catch_abrupt_brake"): ScenarioAlias(RigidScenario.CATCH_BRAKE),
    ("falling_catch", "catch_tilt"): ScenarioAlias(RigidScenario.CATCH_TILT),
    ("falling_catch", "catch_tray_tilt"): ScenarioAlias(RigidScenario.CATCH_TILT),
    ("falling_catch", "catch_recovery"): ScenarioAlias(RigidScenario.CATCH_EDGE_RECOVERY),
    ("falling_catch", "catch_edge_recovery"): ScenarioAlias(RigidScenario.CATCH_EDGE_RECOVERY),
    ("falling_catch", "shallow_tray"): ScenarioAlias(
        RigidScenario.CATCH_RETAIN, tool_kind=ToolKind.SHALLOW_TRAY
    ),
    ("falling_catch", "deep_tray"): ScenarioAlias(
        RigidScenario.CATCH_RETAIN, tool_kind=ToolKind.DEEP_TRAY
    ),
    ("rolling_interception", "default"): ScenarioAlias(RigidScenario.PADDLE_BLOCK),
    ("rolling_interception", "rolling_island"): ScenarioAlias(RigidScenario.REDIRECT_TO_TARGET),
    ("rolling_interception", "rolling_ball_interception"): ScenarioAlias(RigidScenario.PADDLE_BLOCK),
    ("rolling_interception", "straight_ball_left_to_right"): ScenarioAlias(
        RigidScenario.STRAIGHT_ROLL, direction=1
    ),
    ("rolling_interception", "straight_ball_right_to_left"): ScenarioAlias(
        RigidScenario.STRAIGHT_ROLL, direction=-1
    ),
    ("rolling_interception", "rolling_sliding_transition"): ScenarioAlias(
        RigidScenario.ROLLING_SLIDING_TRANSITION
    ),
    ("rolling_interception", "sliding_puck"): ScenarioAlias(
        RigidScenario.STRAIGHT_SLIDE, shape=RigidShape.PUCK
    ),
    ("rolling_interception", "sliding_cube"): ScenarioAlias(
        RigidScenario.STRAIGHT_SLIDE, shape=RigidShape.CUBE
    ),
    ("rolling_interception", "small_slope"): ScenarioAlias(RigidScenario.SMALL_SLOPE),
    ("rolling_interception", "ramp_to_table"): ScenarioAlias(RigidScenario.RAMP_TO_TABLE),
    ("rolling_interception", "table_edge_fall"): ScenarioAlias(RigidScenario.ROLL_OFF_EDGE),
    # Sliding friction is not identifiable from an ideal no-slip rolling
    # sphere.  Use a puck with zero initial spin so the five-point sweep has a
    # directly measurable stopping response.
    ("rolling_interception", "friction_sweep"): ScenarioAlias(
        RigidScenario.STRAIGHT_SLIDE, shape=RigidShape.PUCK
    ),
    ("rolling_interception", "temporary_occlusion"): ScenarioAlias(RigidScenario.OCCLUDED_INTERCEPTION),
    ("rolling_interception", "container_receive"): ScenarioAlias(RigidScenario.CONTAINER_RECEIVE),
    ("rolling_interception", "paddle_redirect"): ScenarioAlias(RigidScenario.REDIRECT_TO_TARGET),
    ("rolling_interception", "paddle_block"): ScenarioAlias(RigidScenario.PADDLE_BLOCK),
    ("projectile_rebound", "default"): ScenarioAlias(RigidScenario.BOUNCE_TO_INTERCEPTION),
    ("projectile_rebound", "free_contact_rebound"): ScenarioAlias(RigidScenario.TABLE_BOUNCE),
    ("projectile_rebound", "projectile_interception"): ScenarioAlias(RigidScenario.PADDLE_DEFLECTION),
    ("projectile_rebound", "bounce_sweep"): ScenarioAlias(RigidScenario.TABLE_BOUNCE),
    ("projectile_rebound", "direct_interception"): ScenarioAlias(RigidScenario.DIRECT_PROJECTILE),
    ("projectile_rebound", "table_bounce"): ScenarioAlias(RigidScenario.TABLE_BOUNCE),
    ("projectile_rebound", "wall_rebound"): ScenarioAlias(RigidScenario.WALL_REBOUND),
    ("projectile_rebound", "angled_barrier_rebound"): ScenarioAlias(RigidScenario.ANGLED_BARRIER_REBOUND),
    ("projectile_rebound", "paddle_deflection"): ScenarioAlias(RigidScenario.PADDLE_DEFLECTION),
    ("projectile_rebound", "gravity_sweep"): ScenarioAlias(RigidScenario.TABLE_BOUNCE),
    ("projectile_rebound", "restitution_sweep"): ScenarioAlias(RigidScenario.TABLE_BOUNCE),
    ("projectile_rebound", "ramp_launch"): ScenarioAlias(RigidScenario.RAMP_LAUNCH),
    ("projectile_rebound", "roll_off_edge"): ScenarioAlias(RigidScenario.PROJECTILE_ROLL_OFF_EDGE),
    ("projectile_rebound", "floor_to_wall"): ScenarioAlias(RigidScenario.FLOOR_TO_WALL),
    ("projectile_rebound", "flight_to_table_bounce"): ScenarioAlias(RigidScenario.FLIGHT_TO_TABLE_BOUNCE),
    ("projectile_rebound", "bounce_to_robot_interception"): ScenarioAlias(RigidScenario.BOUNCE_TO_INTERCEPTION),
}


def resolve_scenario_alias(family: str, subfamily: str) -> ScenarioAlias:
    """Resolve every rigid acceptance-suite name without cross-family coercion."""

    key = (family.strip().lower(), subfamily.strip().lower())
    try:
        return _SUBFAMILY_ALIASES[key]
    except KeyError as error:
        supported = sorted(name for (family_name, name) in _SUBFAMILY_ALIASES if family_name == key[0])
        raise ValueError(
            f"unsupported native rigid subfamily {family}/{subfamily}; supported for {family}: {supported}"
        ) from error


def make_scenario_spec_for_subfamily(
    family: str,
    subfamily: str,
    *,
    seed: int,
    branch: IntendedBranch | str,
    scene_style: str,
    physics_variant: str = "nominal",
    split_group_id: str | None = None,
    counterfactual_bundle_id: str | None = None,
    physics_counterfactual_family_id: str | None = None,
    options: Mapping[str, Any] | None = None,
) -> ScenarioSpec:
    """Build one exact suite case, including shape/direction/sweep overrides."""

    alias = resolve_scenario_alias(family, subfamily)
    extras = {
        **dict(options or {}),
        "suite_subfamily": subfamily,
        "physics_variant": physics_variant,
        "scene_group_id": split_group_id or f"{family}:{subfamily}:{seed}",
    }
    spec = make_scenario_spec(
        alias.scenario,
        seed=seed,
        branch=branch,
        scene_style=scene_style,
        shape=alias.shape,
        direction=alias.direction,
        scenario_id=split_group_id or f"{family}:{subfamily}:{seed}",
        extras=extras,
    )
    if alias.tool_kind is not None:
        wall_height = 0.055 if alias.tool_kind == ToolKind.SHALLOW_TRAY else 0.11
        spec = replace(
            spec,
            tool=replace(spec.tool, kind=alias.tool_kind, wall_height_m=wall_height),
        )
    if subfamily == "restitution_sweep":
        # The versioned solver-to-effective-restitution regression is valid at
        # this declared impact-speed regime.  Keep the family seed-dependent
        # appearance and lateral state, but normalize the vertical drop so the
        # five interventions test restitution rather than an undeclared change
        # in velocity-dependent soft-contact response.  The 0.8 s fixed family
        # horizon contains the first separated rebound sample and intentionally
        # ends before secondary bounces can contaminate the one-contact sweep.
        calibrated_drop_height_m = 0.6754658151889741 - 0.03964438134293399
        spec = replace(
            spec,
            initial_state=replace(
                spec.initial_state,
                position_m=(
                    spec.initial_state.position_m[0],
                    spec.initial_state.position_m[1],
                    spec.object.radius_m + calibrated_drop_height_m,
                ),
                linear_velocity_m_s=(
                    spec.initial_state.linear_velocity_m_s[0],
                    spec.initial_state.linear_velocity_m_s[1],
                    0.10034442119281872,
                ),
            ),
            maximum_duration_s=0.8,
        )
    sweep_values = {
        "gravity": (4.905, 7.3575, 9.81, 12.2625, 14.715),
        "friction": (0.05, 0.15, 0.30, 0.60, 1.00),
        "restitution": (0.05, 0.25, 0.50, 0.75, 0.95),
    }
    prefix, separator, raw_index = physics_variant.rpartition("_")
    if separator and prefix in sweep_values and raw_index.isdigit():
        index = int(raw_index)
        if not 0 <= index < len(sweep_values[prefix]):
            raise ValueError(f"physics variant index out of range: {physics_variant}")
        value = sweep_values[prefix][index]
        if prefix == "gravity":
            spec = replace(spec, gravity_m_s2=(0.0, 0.0, -value))
        elif prefix == "friction":
            spec = replace(
                spec,
                # MuJoCo combines the friction values of both contacting
                # geoms.  Vary both implementation proxies so a lower surface
                # value is not silently masked by the unchanged object geom.
                object=replace(
                    spec.object,
                    friction=(
                        value,
                        spec.object.friction[1],
                        spec.object.friction[2],
                    ),
                ),
                surface_friction=(
                    value,
                    spec.surface_friction[1],
                    spec.surface_friction[2],
                ),
            )
        else:
            spec = replace(
                spec,
                object=replace(spec.object, effective_restitution_target=value),
            )
    spec = replace(
        spec,
        counterfactual_bundle_id=counterfactual_bundle_id,
        physics_counterfactual_family_id=physics_counterfactual_family_id,
        split_group_id=split_group_id,
    )
    spec.validate()
    return spec


def _camera_pair(family: NativeFamily) -> tuple[CameraSpec, CameraSpec]:
    main = CameraSpec(
        name="main",
        position_m=(1.65, -1.75, 1.22),
        look_at_m=(0.25, 0.0, 0.38),
        role="main_three_quarter",
    )
    if family == NativeFamily.ROLLING_INTERCEPTION:
        secondary = CameraSpec(
            name="secondary",
            position_m=(0.15, -0.15, 2.25),
            look_at_m=(0.05, 0.0, 0.0),
            up_world=(0.0, 1.0, 0.0),
            fovy_deg=50.0,
            role="top_oblique",
        )
    else:
        secondary = CameraSpec(
            name="secondary",
            position_m=(1.72, 0.10, 0.88),
            look_at_m=(0.35, 0.0, 0.40),
            fovy_deg=46.0,
            role="side",
        )
    return main, secondary


def _phases_for_falling(
    scenario: RigidScenario,
    branch: IntendedBranch,
    landing: tuple[float, float, float],
    start: tuple[float, float, float],
) -> tuple[ControllerPhase, ...]:
    target = landing
    if branch == IntendedBranch.NEAR_MISS:
        target = (landing[0], landing[1] + 0.30, landing[2])
    elif branch == IntendedBranch.CONTACT_FAILURE:
        target = (landing[0], landing[1] + 0.115, landing[2])
    elif branch == IntendedBranch.WRONG_ACTION:
        target = (landing[0] - 0.24, landing[1] - 0.20, landing[2] + 0.08)
    elif branch == IntendedBranch.NO_OP:
        return (
            ControllerPhase("no_op", 0.0, 2.5, start, interpolation="hold"),
        )
    phases = [
        ControllerPhase("approach", 0.0, 0.34, target),
        ControllerPhase("catch", 0.34, 0.70, target, interpolation="hold"),
        ControllerPhase("retain", 0.70, 1.15, target, interpolation="hold"),
    ]
    if scenario == RigidScenario.CATCH_TRANSPORT:
        phases.extend(
            (
                ControllerPhase("transport", 1.15, 1.85, (target[0] + 0.16, target[1], target[2])),
                ControllerPhase("terminal_hold", 1.85, 2.50, (target[0] + 0.16, target[1], target[2]), interpolation="hold"),
            )
        )
    elif scenario == RigidScenario.CATCH_BRAKE:
        phases.extend(
            (
                ControllerPhase("accelerate", 1.15, 1.45, (target[0] + 0.18, target[1], target[2])),
                ControllerPhase("abrupt_brake", 1.45, 1.55, (target[0] + 0.18, target[1], target[2]), interpolation="hold"),
                ControllerPhase("terminal_hold", 1.55, 2.50, (target[0] + 0.18, target[1], target[2]), interpolation="hold"),
            )
        )
    elif scenario == RigidScenario.CATCH_TILT:
        phases.extend(
            (
                ControllerPhase("tilt", 1.15, 1.70, target, target_rpy_rad=(0.0, 0.28, 0.0)),
                ControllerPhase("terminal_hold", 1.70, 2.50, target, target_rpy_rad=(0.0, 0.28, 0.0), interpolation="hold"),
            )
        )
    elif scenario == RigidScenario.CATCH_EDGE_RECOVERY:
        phases.extend(
            (
                ControllerPhase("edge_disturbance", 1.15, 1.45, (target[0], target[1] + 0.11, target[2])),
                ControllerPhase("edge_recovery", 1.45, 1.90, (target[0], target[1] - 0.05, target[2])),
                ControllerPhase("terminal_hold", 1.90, 2.50, (target[0], target[1] - 0.05, target[2]), interpolation="hold"),
            )
        )
    else:
        phases.append(ControllerPhase("terminal_hold", 1.15, 2.50, target, interpolation="hold"))
    return tuple(phases)


def _phases_for_paddle(
    branch: IntendedBranch,
    target: tuple[float, float, float],
    *,
    duration_s: float,
    start: tuple[float, float, float] = (0.205, -0.220, 0.425),
    approach_start_s: float = 0.0,
    approach_duration_s: float = 0.55,
    near_miss_offset_m: float = 0.22,
    contact_failure_offset_m: float = 0.09,
) -> tuple[ControllerPhase, ...]:
    if branch == IntendedBranch.NEAR_MISS:
        target = (target[0], target[1] + near_miss_offset_m, target[2])
    elif branch == IntendedBranch.CONTACT_FAILURE:
        target = (
            target[0],
            target[1] + contact_failure_offset_m,
            target[2],
        )
    elif branch == IntendedBranch.WRONG_ACTION:
        target = (target[0] - 0.25, target[1] + 0.25, target[2] + 0.15)
    elif branch == IntendedBranch.NO_OP:
        return (ControllerPhase("no_op", 0.0, duration_s, start, interpolation="hold"),)
    approach_end_s = approach_start_s + approach_duration_s
    intercept_end_s = min(approach_end_s + 0.85, duration_s - 1e-6)
    phases: list[ControllerPhase] = []
    if approach_start_s > 0.0:
        phases.append(
            ControllerPhase(
                "pre_interaction_hold",
                0.0,
                approach_start_s,
                start,
                interpolation="hold",
            )
        )
    phases.extend(
        (
            ControllerPhase("approach", approach_start_s, approach_end_s, target),
            ControllerPhase(
                "intercept",
                approach_end_s,
                intercept_end_s,
                target,
                interpolation="hold",
            ),
            ControllerPhase(
                "terminal_hold",
                intercept_end_s,
                duration_s,
                target,
                interpolation="hold",
            ),
        )
    )
    return tuple(phases)


def make_scenario_spec(
    scenario: RigidScenario | str,
    *,
    seed: int,
    branch: IntendedBranch | str = IntendedBranch.SUCCESS,
    scene_style: str = "clean_franka_lab",
    shape: RigidShape | str = RigidShape.SPHERE,
    direction: int | None = None,
    physics_partition: str = "nominal",
    scenario_id: str | None = None,
    extras: Mapping[str, Any] | None = None,
) -> ScenarioSpec:
    """Create one deterministic, realistically bounded native rigid scenario."""

    scenario = RigidScenario(scenario)
    branch = IntendedBranch(branch)
    shape = RigidShape(shape)
    family = scenario_family(scenario)
    rng = random.Random(deterministic_seed("native-scenario", seed, scenario.value))
    radius = rng.uniform(0.028, 0.045)
    density = rng.uniform(350.0, 950.0)
    if shape == RigidShape.SPHERE:
        mass = 4.0 / 3.0 * math.pi * radius**3 * density
        half_extents = (radius, radius, radius)
    elif shape == RigidShape.PUCK:
        half_extents = (radius * 1.25, radius * 1.25, radius * 0.42)
        mass = math.pi * half_extents[0] ** 2 * (2 * half_extents[2]) * density
    else:
        half_extents = (radius, radius, radius)
        mass = 8.0 * radius**3 * density
    mass = max(0.012, min(0.22, mass))
    if shape == RigidShape.SPHERE:
        volume = 4.0 / 3.0 * math.pi * radius**3
    elif shape == RigidShape.PUCK:
        volume = math.pi * half_extents[0] ** 2 * (2.0 * half_extents[2])
    else:
        volume = 8.0 * math.prod(half_extents)
    effective_density = mass / volume
    object_spec = RigidObjectSpec(
        shape=shape,
        radius_m=radius,
        half_extents_m=half_extents,
        mass_kg=mass,
        friction=(rng.uniform(0.32, 0.72), 0.01, 0.001),
        effective_restitution_target=rng.uniform(0.08, 0.32),
        rgba=(rng.uniform(0.15, 0.92), rng.uniform(0.15, 0.82), rng.uniform(0.12, 0.75), 1.0),
        density_kg_m3=effective_density,
    )
    if family == NativeFamily.PROJECTILE_REBOUND and object_spec.mass_kg > 0.18:
        # Keep high-speed paddle impacts inside the first Franka acceptance
        # envelope.  Heavier projectiles remain a later calibrated/OOD range;
        # they produced joint-acceleration impulses above the declared limit.
        object_spec = replace(
            object_spec,
            mass_kg=0.18,
            density_kg_m3=0.18 / volume,
        )
    # Bundle-seeded embodiment/controller variation is independent of branch
    # intent and therefore remains fixed across action and physics siblings.
    controller_latency_s = rng.uniform(0.0, 0.035)
    # Use a separate deterministic stream so adding observation latency never
    # perturbs any physical/controller draw from the scenario RNG.
    camera_latency_s = (
        1.0 / 30.0
        if deterministic_seed(seed, "camera-latency-v1") % 2
        else 0.0
    )
    robot_start_joint_offsets_rad = tuple(rng.uniform(-0.025, 0.025) for _ in range(7))
    extra_values = dict(extras or {})
    expected: tuple[str, ...] = ()
    max_contacts = 2
    if family == NativeFamily.FALLING_CATCH:
        # A failed catch can legitimately contain tool contact followed by a
        # ground impact and one settling rebound.  The one/two-contact design
        # restriction applies to the projectile/rebound production family;
        # falling episodes declare this explicit three-event ceiling instead
        # of dropping post-failure physics evidence.
        max_contacts = 3
        x, y = 0.42, 0.0
        drift = (0.0, 0.0, 0.0)
        if scenario == RigidScenario.OFF_CENTER_DROP:
            x, y = 0.48, rng.choice((-1.0, 1.0)) * 0.09
        elif scenario == RigidScenario.DRIFTED_DROP:
            x, y = 0.34, -0.10
            drift = (rng.uniform(0.12, 0.22), rng.uniform(0.03, 0.08), 0.0)
        height = rng.uniform(1.05, 1.28)
        initial = InitialStateSpec(
            position_m=(x, y, height),
            linear_velocity_m_s=drift,
            angular_velocity_rad_s=tuple(rng.uniform(-2.0, 2.0) for _ in range(3)),
        )
        tool_start = (0.420, -0.320, 0.425)
        tool = ToolSpec(
            ToolKind.DEEP_TRAY if scenario in {RigidScenario.CATCH_BRAKE, RigidScenario.CATCH_TILT} else ToolKind.SHALLOW_TRAY,
            tool_start,
            half_extents_m=(0.145, 0.145, 0.012),
            wall_height_m=0.075,
        )
        landing_time = math.sqrt(max(0.0, 2.0 * (height - 0.48) / 9.81))
        landing = (x + drift[0] * landing_time, y + drift[1] * landing_time, 0.425)
        extra_values.setdefault("nominal_intercept_position_m", landing)
        extra_values.setdefault("robot_base_position_m", (-0.135, -0.320, -0.10))
        phases = _phases_for_falling(scenario, branch, landing, tool_start)
        expected = ("native_tool",)
        duration = 2.5
    elif family == NativeFamily.ROLLING_INTERCEPTION:
        direction_value = direction if direction in {-1, 1} else (-1 if seed % 2 else 1)
        x = -0.45 * direction_value
        speed = rng.uniform(0.65, 0.92)
        support_half_height = (
            radius if shape == RigidShape.SPHERE else half_extents[2]
        )
        # Begin in native support contact so a genuine sliding->rolling
        # transition is present in saved frame zero instead of disappearing
        # between 30 Hz samples during a short settling drop.
        z = support_half_height - 0.0002
        if scenario == RigidScenario.RAMP_TO_TABLE:
            x, z = -0.78, 0.17
            direction_value = 1
        initial = InitialStateSpec(
            position_m=(x, rng.uniform(-0.035, 0.035), z),
            linear_velocity_m_s=(direction_value * speed, 0.0, 0.0),
            angular_velocity_rad_s=(
                0.0,
                -direction_value * speed / radius
                if scenario not in {RigidScenario.STRAIGHT_SLIDE, RigidScenario.ROLLING_SLIDING_TRANSITION}
                else 0.0,
                0.0,
            ),
        )
        tool_start = (0.205, -0.220, 0.425)
        if scenario == RigidScenario.CONTAINER_RECEIVE:
            tool = ToolSpec(
                ToolKind.SMALL_BIN,
                tool_start,
                half_extents_m=(0.15, 0.15, 0.012),
                wall_height_m=0.09,
            )
        else:
            tool = ToolSpec(
                ToolKind.FLAT_PADDLE,
                tool_start,
                half_extents_m=(0.018, 0.14, 0.13),
            )
        goal_center = (0.16 * direction_value, 0.0, 0.0)
        if scenario == RigidScenario.CONTAINER_RECEIVE:
            target = (goal_center[0], initial.position_m[1], 0.13)
        else:
            target = (-0.08 * direction_value, initial.position_m[1], 0.08)
        extra_values.setdefault("nominal_intercept_position_m", target)
        phases = _phases_for_paddle(
            branch,
            target,
            duration_s=3.0,
            start=tool_start,
            approach_duration_s=0.40,
            contact_failure_offset_m=0.04,
        )
        if scenario in {
            RigidScenario.STRAIGHT_ROLL,
            RigidScenario.STRAIGHT_SLIDE,
            RigidScenario.ROLLING_SLIDING_TRANSITION,
            RigidScenario.SMALL_SLOPE,
            RigidScenario.RAMP_TO_TABLE,
            RigidScenario.ROLL_OFF_EDGE,
        }:
            # Passive calibration/transition scenes keep the robot outside the
            # interaction corridor.  The Panda remains actuator-held at its
            # keyframe; it does not move through the object's path merely to
            # satisfy an otherwise artificial phase target.
            phases = ()
            extra_values.setdefault("robot_base_position_m", (-0.35, -0.90, -0.10))
        else:
            # Keep the actuator-held home tool outside the object corridor.
            # Contact must be caused by a commanded branch, not by a passive
            # paddle fortuitously occupying the path in no-op episodes.
            extra_values.setdefault("robot_base_position_m", (-0.35, -0.220, -0.10))
        if scenario == RigidScenario.RAMP_TO_TABLE:
            expected = ("ramp_surface", "table_surface")
        elif scenario == RigidScenario.ROLL_OFF_EDGE:
            expected = ("table_surface", "free_flight")
        elif scenario in {
            RigidScenario.PADDLE_BLOCK,
            RigidScenario.REDIRECT_TO_TARGET,
            RigidScenario.OCCLUDED_INTERCEPTION,
        }:
            expected = ("table_surface", "native_tool")
        else:
            expected = ("table_surface",)
        extra_values.setdefault("direction", direction_value)
        extra_values.setdefault("goal_center_m", goal_center)
        duration = 3.0
    else:
        z = rng.uniform(0.62, 0.92)
        vx = rng.uniform(1.25, 1.75)
        vz = rng.uniform(-0.25, 0.20)
        if scenario in {
            RigidScenario.WALL_REBOUND,
            RigidScenario.ANGLED_BARRIER_REBOUND,
        }:
            # Strike the finite barrier while clearly airborne.  The previous
            # geometry let the projectile bounce on the table before reaching
            # the nominal wall, silently turning a one-contact wall scenario
            # into a ground-multibounce sequence.
            z, vz = 1.05, 0.0
        if scenario == RigidScenario.RAMP_LAUNCH:
            z, vx, vz = 0.17, 1.0, 0.35
        elif scenario == RigidScenario.PROJECTILE_ROLL_OFF_EDGE:
            z, vx, vz = radius + 0.002, rng.uniform(0.70, 0.90), 0.0
        initial = InitialStateSpec(
            position_m=(
                -0.78 if scenario == RigidScenario.RAMP_LAUNCH else -0.30,
                rng.uniform(-0.04, 0.04),
                z,
            ),
            linear_velocity_m_s=(vx, rng.uniform(-0.025, 0.025), vz),
            angular_velocity_rad_s=tuple(rng.uniform(-1.5, 1.5) for _ in range(3)),
        )
        if scenario == RigidScenario.PROJECTILE_ROLL_OFF_EDGE:
            initial = replace(
                initial,
                # The finite table spans x=[-0.20, 0.90].  Start one radius
                # inside its left edge so this is roll->edge->flight rather
                # than an object initialized in free fall.
                position_m=(-0.20 + radius, initial.position_m[1], z),
                angular_velocity_rad_s=(0.0, -vx / radius, 0.0),
            )
        if scenario == RigidScenario.BOUNCE_TO_INTERCEPTION:
            # Use the validated moderate-response regime for a single table
            # rebound followed by robot contact.  Higher proxy values produced
            # velocity-dependent solver energy gain at this impact speed.
            object_spec = replace(
                object_spec, effective_restitution_target=0.50
            )
        tool_start = (
            (0.50, -0.32, 0.20)
            if scenario == RigidScenario.BOUNCE_TO_INTERCEPTION
            else (0.205, -0.220, 0.425)
        )
        tool = ToolSpec(
            ToolKind.ANGLED_PADDLE if scenario == RigidScenario.PADDLE_DEFLECTION else ToolKind.FLAT_PADDLE,
            tool_start,
            half_extents_m=(0.018, 0.16, 0.16),
        )
        nominal_intercept = (
            (0.50, initial.position_m[1], 0.20)
            if scenario == RigidScenario.BOUNCE_TO_INTERCEPTION
            else (0.62, initial.position_m[1], 0.35)
        )
        extra_values.setdefault(
            "nominal_intercept_position_m", nominal_intercept
        )
        projectile_duration = (
            1.15
            if scenario == RigidScenario.BOUNCE_TO_INTERCEPTION
            else 3.2
        )
        phases = _phases_for_paddle(
            branch,
            nominal_intercept,
            duration_s=projectile_duration,
            start=tool_start,
            approach_start_s=0.0,
            approach_duration_s=(
                0.80 if scenario == RigidScenario.BOUNCE_TO_INTERCEPTION else 0.55
            ),
            near_miss_offset_m=0.30,
            contact_failure_offset_m=0.12,
        )
        if scenario in {
            RigidScenario.TABLE_BOUNCE,
            RigidScenario.WALL_REBOUND,
            RigidScenario.ANGLED_BARRIER_REBOUND,
            RigidScenario.RAMP_LAUNCH,
            RigidScenario.PROJECTILE_ROLL_OFF_EDGE,
            RigidScenario.FLOOR_TO_WALL,
            RigidScenario.FLIGHT_TO_TABLE_BOUNCE,
        }:
            phases = ()
            extra_values.setdefault("robot_base_position_m", (-0.35, -0.90, -0.10))
        else:
            extra_values.setdefault(
                "robot_base_position_m",
                (
                    (-0.05, -0.32, -0.25)
                    if scenario == RigidScenario.BOUNCE_TO_INTERCEPTION
                    else (-0.35, -0.220, -0.10)
                ),
            )
        if scenario in {RigidScenario.TABLE_BOUNCE, RigidScenario.FLIGHT_TO_TABLE_BOUNCE}:
            expected = ("table_surface",)
        elif scenario == RigidScenario.WALL_REBOUND:
            expected = ("wall_surface",)
        elif scenario == RigidScenario.ANGLED_BARRIER_REBOUND:
            expected = ("angled_barrier_surface",)
        elif scenario == RigidScenario.FLOOR_TO_WALL:
            expected = ("table_surface", "wall_surface")
        elif scenario == RigidScenario.BOUNCE_TO_INTERCEPTION:
            expected = ("table_surface", "native_tool")
        elif scenario == RigidScenario.RAMP_LAUNCH:
            expected = ("ramp_surface", "free_flight")
        elif scenario == RigidScenario.PROJECTILE_ROLL_OFF_EDGE:
            expected = ("table_surface", "free_flight")
        else:
            expected = ("native_tool",)
        duration = projectile_duration
    spec = ScenarioSpec(
        scenario_id=scenario_id or deterministic_uuid("native-scenario", scenario.value, seed),
        family=family,
        scenario=scenario,
        branch=branch,
        seed=seed,
        object=object_spec,
        initial_state=initial,
        tool=tool,
        phases=phases,
        cameras=_camera_pair(family),
        scene_style=scene_style,
        maximum_duration_s=duration,
        minimum_terminal_context_s=(
            0.20 if family == NativeFamily.PROJECTILE_REBOUND else 0.50
        ),
        controller_latency_s=controller_latency_s,
        camera_latency_s=camera_latency_s,
        robot_start_joint_offsets_rad=robot_start_joint_offsets_rad,
        physics_provenance=PhysicsRangeProvenance(partition=physics_partition),
        expected_contact_sequence=expected,
        max_task_contacts=max_contacts,
        extras=extra_values,
    )
    spec.validate()
    return spec


def _physics_fields(spec: ScenarioSpec) -> dict[str, Any]:
    return {
        "gravity": physics_field(list(spec.gravity_m_s2), "m/s^2"),
        "mass": physics_field(spec.object.mass_kg, "kg"),
        "density": physics_field(spec.object.density_kg_m3, "kg/m^3"),
        "radius": physics_field(
            spec.object.radius_m,
            "m",
            valid=spec.object.shape == RigidShape.SPHERE,
            interpretation="physical" if spec.object.shape == RigidShape.SPHERE else "unknown",
        ),
        "shape_half_extents": physics_field(list(spec.object.half_extents_m), "m"),
        "surface_dynamic_friction": physics_field(
            spec.surface_friction[0], "1", interpretation="simulator_proxy"
        ),
        "surface_static_friction": physics_field(None, "1", valid=False, interpretation="unknown"),
        "effective_restitution_target": physics_field(
            spec.object.effective_restitution_target,
            "1",
            interpretation=(
                "calibrated_effective"
                if spec.physics_provenance.calibrated
                else "simulator_proxy"
            ),
        ),
        "mujoco_torsional_friction": physics_field(
            spec.object.friction[1], "m", interpretation="simulator_proxy"
        ),
        "mujoco_rolling_friction": physics_field(
            spec.object.friction[2], "m", interpretation="simulator_proxy"
        ),
        "mujoco_surface_friction": physics_field(
            list(spec.surface_friction), "mixed", interpretation="simulator_proxy"
        ),
        "mujoco_object_friction": physics_field(
            list(spec.object.friction), "mixed", interpretation="simulator_proxy"
        ),
        "mujoco_surface_solref": physics_field(
            [0.006, 0.85], "solver", interpretation="simulator_proxy"
        ),
        "mujoco_surface_solimp": physics_field(
            [0.92, 0.99, 0.001], "solver", interpretation="simulator_proxy"
        ),
        "mujoco_object_solref": physics_field(
            [
                0.006,
                contact_damping_for_target(
                    spec.object.effective_restitution_target
                ),
            ],
            "solver",
            interpretation="simulator_proxy",
        ),
        "mujoco_object_contact_priority": physics_field(
            OBJECT_CONTACT_PRIORITY, "solver", interpretation="simulator_proxy"
        ),
        "mujoco_restitution_solver_profile": physics_field(
            RESTITUTION_SOLVER_PROFILE_VERSION,
            "identifier",
            interpretation="simulator_proxy",
        ),
        "mujoco_object_solimp": physics_field(
            [0.92, 0.99, 0.001], "solver", interpretation="simulator_proxy"
        ),
        "simulation_timestep": physics_field(1.0 / spec.sim_hz, "s"),
        "substeps": physics_field(1, "count"),
    }


def scenario_to_episode_plan(spec: ScenarioSpec) -> EpisodePlan:
    """Bind a typed native spec to the repository's canonical episode identity."""

    spec.validate()
    group_basis = str(spec.extras.get("scene_group_id") or spec.scenario_id)
    split_group_id = spec.split_group_id or deterministic_uuid("native-split", group_basis)
    physics_hash = stable_hash(_physics_fields(spec))
    action_payload = [
        {
            "name": phase.name,
            "start_s": phase.start_s,
            "end_s": phase.end_s,
            "target_position_m": phase.target_position_m,
            "target_rpy_rad": phase.target_rpy_rad,
            "interpolation": phase.interpolation,
        }
        for phase in spec.phases
    ]
    action_contract = {
        "controller_latency_s": spec.controller_latency_s,
        "phases": action_payload,
    }
    action_hash = stable_hash(action_contract)
    bundle_id = spec.counterfactual_bundle_id or deterministic_uuid(
        "native-action-bundle", group_basis, physics_hash
    )
    # A physics-family identity is meaningful only when the caller has
    # declared sibling interventions. Nominal singleton rollouts must not look
    # like incomplete counterfactual families.
    physics_cf_id = spec.physics_counterfactual_family_id
    scene = {
        "scenario_id": spec.scenario_id,
        "initial_position_m": list(spec.initial_state.position_m),
        "initial_velocity_mps": list(spec.initial_state.linear_velocity_m_s),
        "initial_angular_velocity_rad_s": list(spec.initial_state.angular_velocity_rad_s),
        "object_shape": spec.object.shape.value,
        "visual_seed": spec.seed,
        "background_style": spec.scene_style,
        "camera_roles": {camera.name: camera.role for camera in spec.cameras},
        "robot_start_joint_offsets_rad": list(spec.robot_start_joint_offsets_rad),
        "controller_latency_s": spec.controller_latency_s,
        "camera_latency_s": spec.camera_latency_s,
        "randomization": dict(spec.extras.get("randomization") or {}),
    }
    fixed_fields = {
        "scene": scene,
        "appearance": spec.object.rgba,
        "cameras": [camera.__dict__ for camera in spec.cameras],
        "robot_model": spec.robot_model,
        "tool": spec.tool.__dict__,
    }
    invariant_hash = stable_hash({**fixed_fields, "action_hash": action_hash})
    plan = EpisodePlan(
        schema_version=FAMILY_SCHEMA_VERSION,
        family=spec.family.value,
        subfamily=spec.scenario.value,
        variant="native_mujoco_v1",
        robot_model=spec.robot_model,
        tool_type=spec.tool.kind.value,
        episode_uuid=deterministic_uuid(
            "native-episode", bundle_id, physics_cf_id, physics_hash, spec.seed
        ),
        counterfactual_bundle_id=bundle_id,
        physics_counterfactual_family_id=physics_cf_id,
        split_group_id=split_group_id,
        scene_seed=spec.seed,
        branch_seed=deterministic_seed(spec.seed, spec.branch.value),
        intended_branch=spec.branch.value,
        branch_parameters={
            "controller_latency_s": spec.controller_latency_s,
            "controller_phases": action_payload,
        },
        scene_parameters=scene,
        physics=_physics_fields(spec),
        physics_variant=str(spec.extras.get("physics_variant", "nominal")),
        action_hash=action_hash,
        invariant_hash=invariant_hash,
        physics_hash=physics_hash,
        views=("main", "secondary"),
        rates_hz={
            "simulation": spec.sim_hz,
            "control": spec.control_hz,
            "video": spec.video_hz,
        },
        duration_s=spec.maximum_duration_s,
        randomization_level=str(spec.extras.get("randomization_level", "R1")),
        scene_style=spec.scene_style,
        source_generator="dynamic_robot_dataset.backends.mujoco_native",
        source_generator_version="1.0.0",
        config_hash=spec.spec_hash,
        options={
            "backend": "native_mujoco",
            "action_mode": "franka_joint_position_actuator",
            "native_scenario_spec": spec.to_dict(),
            "fixed_field_hash": stable_hash(fixed_fields),
            "counterfactual_expected_members": list(spec.extras.get("counterfactual_expected_members", ())),
            "counterfactual_intervention_fields": list(spec.extras.get("counterfactual_intervention_fields", ())),
        },
    )
    return plan


def _native_scenario_from_legacy_plan(plan: EpisodePlan) -> ScenarioSpec:
    try:
        alias = resolve_scenario_alias(plan.family, plan.subfamily)
    except ValueError as error:
        raise ValueError(
            f"EpisodePlan {plan.family}/{plan.subfamily} has no native scenario mapping; "
            "build it with scenario_to_episode_plan"
        ) from error
    branch_alias = {
        "success_seeking": IntendedBranch.SUCCESS,
        "near_miss": IntendedBranch.NEAR_MISS,
        "contact_failure": IntendedBranch.CONTACT_FAILURE,
        "bad_action": IntendedBranch.WRONG_ACTION,
        "no_op": IntendedBranch.NO_OP,
        "wrong_action": IntendedBranch.WRONG_ACTION,
    }
    spec = make_scenario_spec(
        alias.scenario,
        seed=plan.scene_seed,
        branch=branch_alias.get(plan.intended_branch, IntendedBranch.WRONG_ACTION),
        scene_style=plan.scene_style,
        scenario_id=str(plan.scene_parameters.get("scenario_id") or plan.split_group_id),
        extras={
            "scene_group_id": plan.split_group_id,
            "physics_variant": plan.physics_variant,
        },
        shape=alias.shape,
        direction=alias.direction,
    )
    if alias.tool_kind is not None:
        spec = replace(spec, tool=replace(spec.tool, kind=alias.tool_kind))
    # Do not transplant analytical-surrogate state targets into the native
    # scene.  Those coordinates were authored for a different solver/tool
    # geometry and can silently turn a native catch into an unrelated miss.
    # The legacy plan contributes stable identity and requested scenario; the
    # typed native factory owns mutually consistent initial state and actions.
    return replace(
        spec,
        counterfactual_bundle_id=plan.counterfactual_bundle_id,
        physics_counterfactual_family_id=plan.physics_counterfactual_family_id,
        split_group_id=plan.split_group_id,
    )


def scenario_spec_from_episode_plan(plan: EpisodePlan) -> ScenarioSpec:
    raw = plan.options.get("native_scenario_spec")
    return ScenarioSpec.from_dict(raw) if isinstance(raw, Mapping) else _native_scenario_from_legacy_plan(plan)


def compile_native_plan(plan: EpisodePlan) -> NativeEpisodePlan:
    raw_spec = plan.options.get("native_scenario_spec")
    spec = scenario_spec_from_episode_plan(plan)
    if not isinstance(raw_spec, Mapping):
        # The CLI may route an older family-adapter plan to the native backend.
        # Rebind its stable episode/counterfactual identity to the complete
        # native action/physics contract so downstream records cannot retain
        # analytical-surrogate metadata for a MuJoCo rollout.
        native = scenario_to_episode_plan(spec)
        plan = replace(
            native,
            episode_uuid=plan.episode_uuid,
            counterfactual_bundle_id=plan.counterfactual_bundle_id,
            physics_counterfactual_family_id=plan.physics_counterfactual_family_id,
            split_group_id=plan.split_group_id,
            branch_seed=plan.branch_seed,
            intended_branch=plan.intended_branch,
            subfamily=plan.subfamily,
            physics_variant=plan.physics_variant,
            options={
                **native.options,
                "legacy_plan_source_generator": plan.source_generator,
                "legacy_plan_source_generator_version": plan.source_generator_version,
            },
        )
    fixed_hash = str(plan.options.get("fixed_field_hash") or stable_hash({
        "scene": plan.scene_parameters,
        "views": plan.views,
        "robot_model": plan.robot_model,
        "tool_type": plan.tool_type,
    }))
    return NativeEpisodePlan(
        episode_plan=plan,
        scenario=spec,
        compiled_scenario_hash=spec.spec_hash,
        fixed_field_hash=fixed_hash,
        action_replay_hash=plan.action_hash,
    )
