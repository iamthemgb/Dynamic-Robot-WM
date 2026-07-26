"""Shared constructors used by canonical rigid scenario modules.

The low-level MuJoCo compiler remains the compatibility implementation during
the extraction-only commit.  All callers resolve a leaf module first, and the
module then invokes this adapter.  This preserves fixed-case physics bytes
while giving every subfamily one canonical, discoverable owner.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..common.embodiments import FRANKA_HAND, ROBOTIQ_2F85_THICK_PAD

from .types import ControllerPlan, ScenarioBuildContext, ScenarioModuleSpec


PROJECTILE_INITIAL_STATE_SAMPLER_VERSION = (
    "dynamic-robot-projectile-initial-state/v1"
)

# These are deliberately bounded review/pilot ranges, not an unconstrained
# production distribution.  F1d stays a mild, longer-flight projectile; F2a
# starts farther away and requires the arm to cover a broader interception
# workspace.  Position/velocity/spin are sampled from the initial-state RNG,
# while object size remains governed by the existing rigid physics profile.
_PROJECTILE_INITIAL_STATE_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "F1d": {
        # Keep the preview target inside the strict palm-up IK envelope; the
        # projectile diversity comes from launch position/velocity, not from
        # accepting 5--9 mm IK residuals at workspace-edge targets.
        "target_x_m": (0.465, 0.475),
        "target_y_m": (-0.015, 0.015),
        "incoming_azimuth_deg": (-18.0, 18.0),
        "horizontal_distance_m": (0.35, 0.48),
        "flight_time_s": (0.58, 0.68),
        "initial_vertical_velocity_m_s": (1.50, 2.30),
        "initial_spin_z_rad_s": (-4.0, 4.0),
    },
    "F2a": {
        "target_x_m": (0.41, 0.53),
        "target_y_m": (-0.08, 0.08),
        "incoming_azimuth_deg": (-22.0, 22.0),
        "horizontal_distance_m": (0.45, 0.65),
        "flight_time_s": (0.60, 0.72),
        "initial_vertical_velocity_m_s": (1.70, 2.60),
        "initial_spin_z_rad_s": (-6.0, 6.0),
    },
}


def projectile_randomization_contract(leaf_id: str) -> dict[str, Any]:
    """Return the public, versioned sampling envelope for one projectile leaf."""

    ranges = _PROJECTILE_INITIAL_STATE_RANGES.get(leaf_id)
    if ranges is None:
        raise ValueError(f"{leaf_id} has no projectile initial-state policy")
    return {
        "schema_version": PROJECTILE_INITIAL_STATE_SAMPLER_VERSION,
        "rng_stream": "initial_state",
        "sampling": "continuous_uniform_with_ballistic_solve",
        "derive_velocity_from_sampled_launch_target_and_flight_time": True,
        "outcome_conditioned_resampling": False,
        "ranges": {name: list(bounds) for name, bounds in ranges.items()},
    }


SCALE_INITIAL_STATE_SAMPLER_VERSION = "dynamic-robot-scale-initial-state/v3"

# Per-leaf sampler revisions.  Shard execution re-prepares every declaration
# with current code and requires the fresh spec to match the immutable plan
# bit for bit, so a version change may only reach leaves whose envelopes
# actually changed; every other leaf must keep minting exactly what its
# planned blocks contain.  v4 (2026-07-22, from the block-9000 calibration
# round): F1c narrows drift arrivals to the measured passive-finger-safe
# envelope, and F1d/F2a replace their preview-envelope targets with the
# IK-feasible sub-boxes measured by tools/probe_scale_ik_feasibility.py.
_SCALE_SAMPLER_LEAF_VERSIONS: dict[str, str] = {
    "F1c": "dynamic-robot-scale-initial-state/v4",
    "F1d": "dynamic-robot-scale-initial-state/v4",
    "F2a": "dynamic-robot-scale-initial-state/v4",
}

# Every sampler version whose minted contracts remain executable.  Planned
# blocks are immutable: mid-campaign sampler revisions must keep validating
# declarations minted under earlier versions, so validators check membership
# here instead of pinning one minting version.
SCALE_ACCEPTED_SAMPLER_VERSIONS: tuple[str, ...] = (
    "dynamic-robot-scale-initial-state/v3",
    "dynamic-robot-scale-initial-state/v4",
)


def scale_sampler_version(leaf_id: str) -> str:
    """Return the sampler version this leaf currently mints under."""

    return _SCALE_SAMPLER_LEAF_VERSIONS.get(
        leaf_id, SCALE_INITIAL_STATE_SAMPLER_VERSION
    )

# Bounded diagnostic scale envelopes.  Every range is anchored to a proven
# fixed-review or passive-variation operating point; scale sampling widens the
# initial-state distribution without leaving the measured physics envelope.
# All ``sampled_scale`` cases compile at the calibrated 1200 Hz reference rate.
_SCALE_BALLISTIC_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    # Near-vertical drops around the strict palm-up IK envelope proven by the
    # fixed F1 reviews (target 0.47/±y, event 0.44 s, ballistic construction).
    # v3 envelope, measured on F1a calibration blocks 9000/9001 (200 episodes):
    # the calibrated grasp contact is proven at exactly one descending impact
    # speed (4.32 m/s; the fixed-review 0.44 s free drop).  Sampled arrivals
    # both above (v1: 4.0-4.97) and below (v2: 3.4-4.0) that point produced
    # measured penetration/replay failures, dominantly on the Robotiq, so v3
    # pins the impact speed at the proven value.  By ballistics this also
    # pins the launch height at the proven ~1.50 m, inside every jittered
    # camera framing; diversity comes from target position, approach azimuth,
    # lateral drift, event timing, and spin.
    "F1a": {
        "target_x_m": (0.465, 0.475),
        "target_y_m": (-0.012, 0.012),
        "incoming_azimuth_deg": (-180.0, 180.0),
        "horizontal_distance_m": (0.0, 0.03),
        "flight_time_s": (0.40, 0.45),
        "impact_speed_m_s": (4.32, 4.32),
        "initial_spin_z_rad_s": (-2.0, 2.0),
    },
    # Off-center drop keeps the fixed +Y offset character (nominal 0.035 m).
    "F1b": {
        "target_x_m": (0.465, 0.475),
        "target_y_m": (0.025, 0.045),
        "incoming_azimuth_deg": (-180.0, 180.0),
        "horizontal_distance_m": (0.0, 0.03),
        "flight_time_s": (0.40, 0.45),
        "impact_speed_m_s": (4.32, 4.32),
        "initial_spin_z_rad_s": (-2.0, 2.0),
    },
    # Drifted drop samples around the fixed (-0.09, +0.05) start offset
    # (offset direction atan2(0.05, -0.09) is about 151 degrees).  v4 trims
    # the drift tail: calibration block 9000 measured every passive-finger
    # acceleration violation (97-119 rad/s^2 vs the 80 limit) and the one
    # 2.1 mm gripper penetration at horizontal drift >= 0.09 m arriving at
    # 0.21-0.29 m/s, so the sampled drift keeps the proven lower band and
    # the spin envelope returns to the F1a/F1b-proven +/-2 rad/s.
    "F1c": {
        "target_x_m": (0.465, 0.475),
        "target_y_m": (-0.012, 0.012),
        "incoming_azimuth_deg": (125.0, 175.0),
        "horizontal_distance_m": (0.08, 0.115),
        "flight_time_s": (0.40, 0.45),
        "impact_speed_m_s": (4.32, 4.32),
        "initial_spin_z_rad_s": (-2.0, 2.0),
    },
    # F1d/F2a keep their preview projectile diversity (azimuth, distance,
    # flight time, launch, spin) but bound the intercept target to the
    # IK-feasible sub-box measured on CPU with
    # tools/probe_scale_ik_feasibility.py: the lateral-catch hand pose
    # leaves 3.5-6.2 mm grasp-center residuals (tolerance 3 mm) over much
    # of the preview target boxes, and the residual depends only on the
    # sampled target position (pinning any other dimension leaves the
    # error bit-identical).
    "F1d": {
        **_PROJECTILE_INITIAL_STATE_RANGES["F1d"],
        # Feasible across the full +/-15 mm y band at x <= 0.4675; residuals
        # of 3.8-6.2 mm appear from x = 0.470 outward.
        "target_x_m": (0.465, 0.4675),
        "target_y_m": (-0.012, 0.012),
    },
    "F2a": {
        **_PROJECTILE_INITIAL_STATE_RANGES["F2a"],
        # The preview box (x 0.41-0.53, y +/-0.08) is a patchwork of 3-12 mm
        # residual ridges; this corner probed clean on a 5x5 fine grid.
        "target_x_m": (0.41, 0.44),
        "target_y_m": (0.03, 0.08),
    },
}

# Per-leaf/per-variant parameter envelopes for the non-ballistic scale
# samplers.  P0c reuses the proven ±20% passive-variation speed envelope.
_SCALE_PARAMETER_RANGES: dict[tuple[str, str | None], dict[str, tuple[float, float]]] = {
    ("P0a", "nominal_freefall"): {
        "drop_height_offset_m": (-0.05, 0.05),
        "lateral_speed_m_s": (0.0, 0.06),
    },
    ("P0a", "lateral_freefall"): {
        "drop_height_offset_m": (-0.05, 0.05),
        "lateral_speed_m_s": (0.22, 0.32),
    },
    ("P0b", "ballistic_projectile"): {
        "launch_speed_x_m_s": (1.00, 1.30),
        "launch_speed_y_m_s": (-0.06, 0.06),
        "launch_speed_z_m_s": (3.20, 3.80),
    },
    ("P0b", "angled_projectile"): {
        "launch_speed_x_m_s": (1.00, 1.30),
        "launch_speed_y_m_s": (0.26, 0.40),
        "launch_speed_z_m_s": (3.20, 3.80),
    },
    ("P0c", None): {"speed_factor": (0.80, 1.20)},
    ("P0d", None): {"speed_factor": (0.85, 1.15)},
    ("F2c", None): {
        "lane_speed_factor": (0.85, 1.15),
        "lane_y_m": (-0.02, 0.02),
    },
    ("F3b", None): {
        "speed_factor": (0.85, 1.15),
        "lane_y_m": (-0.03, 0.03),
    },
}

SCALE_SAMPLED_LEAVES = (
    "P0a",
    "P0b",
    "P0c",
    "P0d",
    "F1a",
    "F1b",
    "F1c",
    "F1d",
    "F2a",
    "F2c",
    "F3b",
)


def scale_parameter_ranges(
    leaf_id: str, task_variant: str | None = None
) -> dict[str, tuple[float, float]]:
    """Return the public scale envelope for one non-ballistic leaf/variant."""

    for key in ((leaf_id, task_variant), (leaf_id, None)):
        ranges = _SCALE_PARAMETER_RANGES.get(key)
        if ranges is not None:
            return dict(ranges)
    raise ValueError(f"{leaf_id}/{task_variant} has no scale parameter policy")


def scale_randomization_contract(
    leaf_id: str, task_variant: str | None = None
) -> dict[str, Any]:
    """Return the versioned scale sampling envelope for one leaf."""

    if leaf_id in _SCALE_BALLISTIC_RANGES:
        ranges = _SCALE_BALLISTIC_RANGES[leaf_id]
        sampling = "continuous_uniform_with_ballistic_solve"
    else:
        ranges = scale_parameter_ranges(leaf_id, task_variant)
        sampling = "continuous_uniform_around_fixed_review_recipe"
    return {
        "schema_version": scale_sampler_version(leaf_id),
        "rng_stream": "initial_state",
        "sampling": sampling,
        "outcome_conditioned_resampling": False,
        "ranges": {name: list(bounds) for name, bounds in ranges.items()},
    }


def _projectile_rng(seed: int) -> np.random.Generator:
    """Namespace the projectile sampler without depending on draw order elsewhere."""

    if seed < 0 or seed >= 2**64:
        raise ValueError("projectile initial-state seed must be an unsigned 64-bit value")
    sequence = np.random.SeedSequence(
        [seed & 0xFFFFFFFF, seed >> 32, 0x50524F4A, 1]
    )
    return np.random.default_rng(sequence)


def _scale_rng(seed: int) -> np.random.Generator:
    """Namespace the non-ballistic scale sampler independently of other draws."""

    if seed < 0 or seed >= 2**64:
        raise ValueError("scale initial-state seed must be an unsigned 64-bit value")
    sequence = np.random.SeedSequence(
        [seed & 0xFFFFFFFF, seed >> 32, 0x5343414C, 1]
    )
    return np.random.default_rng(sequence)


def _scale_contract(
    leaf_id: str,
    task_variant: str,
    *,
    seed: int,
    sampled: dict[str, Any],
) -> dict[str, Any]:
    contract = scale_randomization_contract(leaf_id, task_variant)
    contract.update({"source_seed": int(seed), **sampled})
    return contract


def _sample_projectile_initial_state(
    leaf_id: str,
    *,
    seed: int,
    target_z_m: float,
    gravity_z_m_s2: float,
    ranges: dict[str, tuple[float, float]] | None = None,
    contract_base: dict[str, Any] | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    tuple[float, float, float],
    dict[str, Any],
]:
    """Sample a reachable ballistic arc without looking at branch/outcome labels."""

    ranges = _PROJECTILE_INITIAL_STATE_RANGES[leaf_id] if ranges is None else ranges
    rng = _projectile_rng(seed)

    def sample(name: str) -> float:
        lower, upper = ranges[name]
        return float(rng.uniform(lower, upper))

    target = np.asarray(
        (sample("target_x_m"), sample("target_y_m"), target_z_m),
        dtype=np.float64,
    )
    azimuth_deg = sample("incoming_azimuth_deg")
    azimuth_rad = math.radians(azimuth_deg)
    direction_xy = np.asarray(
        (math.cos(azimuth_rad), math.sin(azimuth_rad)), dtype=np.float64
    )
    horizontal_distance = sample("horizontal_distance_m")
    flight_time = sample("flight_time_s")
    if "impact_speed_m_s" in ranges:
        # The calibrated grasp contact admits one proven descending impact
        # speed; derive the launch vertical velocity from it instead of
        # sampling the launch velocity directly.
        impact_speed = sample("impact_speed_m_s")
        initial_vertical_velocity = (
            abs(gravity_z_m_s2) * flight_time - impact_speed
        )
    else:
        impact_speed = None
        initial_vertical_velocity = sample("initial_vertical_velocity_m_s")
    launch_height = (
        target[2]
        - initial_vertical_velocity * flight_time
        - 0.5 * gravity_z_m_s2 * flight_time**2
    )
    # Spawn beyond the interception point and travel back toward the robot.
    # The opposite construction (spawn between the robot base and target)
    # crossed the lower arm during its reach and created an unintended early
    # contact event before the gripper could perform the interception.
    start = np.asarray(
        (
            target[0] + horizontal_distance * direction_xy[0],
            target[1] + horizontal_distance * direction_xy[1],
            launch_height,
        ),
        dtype=np.float64,
    )
    velocity = np.empty(3, dtype=np.float64)
    velocity[:2] = -horizontal_distance * direction_xy / flight_time
    velocity[2] = initial_vertical_velocity
    spin = (0.0, 0.0, sample("initial_spin_z_rad_s"))
    arrival = start + velocity * flight_time
    arrival[2] += 0.5 * gravity_z_m_s2 * flight_time**2
    if not np.allclose(arrival, target, atol=1e-12, rtol=0.0):
        raise RuntimeError("projectile sampler failed its ballistic construction")
    final_vertical_velocity = velocity[2] + gravity_z_m_s2 * flight_time
    if final_vertical_velocity >= 0.0:
        raise RuntimeError("projectile must reach the gripper on its descending arc")
    contract = (
        projectile_randomization_contract(leaf_id)
        if contract_base is None
        else dict(contract_base)
    )
    contract.update(
        {
            "source_seed": int(seed),
            "sampled_target_position_m": target.tolist(),
            "sampled_base_initial_position_m": start.tolist(),
            "sampled_base_initial_linear_velocity_m_s": velocity.tolist(),
            "sampled_base_initial_angular_velocity_rad_s": list(spin),
            "sampled_incoming_azimuth_deg": azimuth_deg,
            "sampled_horizontal_distance_m": horizontal_distance,
            "sampled_flight_time_s": flight_time,
            "sampled_launch_height_m": launch_height,
            "sampled_initial_vertical_velocity_m_s": initial_vertical_velocity,
            "sampled_pinned_impact_speed_m_s": impact_speed,
            "sampled_arrival_vertical_velocity_m_s": float(
                final_vertical_velocity
            ),
        }
    )
    return target, start, velocity, flight_time, spin, contract



def _recipe_payload(
    leaf_id: str,
    task_variant: str,
    embodiment: str,
    branch_role: str,
    *,
    seed: int,
    physics_seed: int,
    tabletop_height_m: float,
    rolling_island_scene: RollingIslandScenePlan | None = None,
    initial_state_mode: str = "fixed_review",
) -> dict[str, Any]:
    from ..backends.source_mujoco.compiler import (
        PhysicalSurface,
        SourceMujocoUnsupported,
        _P0_SUPPORT_HALF_XY_M,
        _grounded_supports_for_surface,
        _sphere_mass,
        _surface,
    )
    from ..backends.source_mujoco.profiles import RIGID_REVIEW_PROFILE
    from .f2b_ramp_launch import SurfaceTransitionContract
    from .f2e_multi_surface_rebound import (
        FLOOR_TO_WALL_REPAIR,
        OrderedContactContract,
    )
    from .f2f_arbitrary_surface_bounce import (
        F2F_BARRIER_CATCH_Z_M,
        F2F_BARRIER_HIT_Z_M,
        F2F_FRANKA_BARRIER_CANDIDATE_BIAS_M,
        F2F_FRANKA_BARRIER_INTERCEPT_BIAS_M,
        F2F_ROBOTIQ_PLANE_INTERCEPT_BIAS_X_M,
        sample_surface_candidate,
        sampled_surface_contract,
    )
    from ..backends.source_mujoco.rolling_island import (
        ROLLING_ISLAND_SURFACE_NAME,
    )

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
        "surface_transition_contract": None,
        "ordered_contact_contract": None,
        "sampled_surface_contract": None,
    }

    if leaf_id == "P0a":
        support_xy = (
            _P0_SUPPORT_HALF_XY_M[leaf_id]
            if tabletop_height_m > 0.0
            else (1.4, 1.4)
        )
        lateral = 0.28 if task_variant == "lateral_freefall" else 0.0
        drop_height = 0.47
        key_event = 0.3
        sampling_contract = None
        if initial_state_mode == "sampled_scale":
            ranges = scale_parameter_ranges(leaf_id, task_variant)
            scale_rng = _scale_rng(seed)
            height_offset = float(scale_rng.uniform(*ranges["drop_height_offset_m"]))
            lateral = float(scale_rng.uniform(*ranges["lateral_speed_m_s"]))
            drop_height += height_offset
            key_event = math.sqrt(
                2.0 * (drop_height - radius) / abs(gravity[2])
            )
            sampling_contract = _scale_contract(
                leaf_id,
                task_variant,
                seed=seed,
                sampled={
                    "sampled_drop_height_offset_m": height_offset,
                    "sampled_drop_height_m": drop_height,
                    "sampled_lateral_speed_m_s": lateral,
                    "recomputed_key_event_time_s": key_event,
                },
            )
        base.update(
            duration_s=0.8,
            key_event_time_s=key_event,
            motion_kind="passive_freefall",
            object_initial_position_m=(-0.15, -0.10, tabletop_height_m + drop_height),
            object_initial_linear_velocity_m_s=(lateral, 0.0, 0.0),
            surfaces=(_surface("supported_floor", "floor", (0, 0, r1_support_center_z), (*support_xy, r1_support_half_height)),),
            initial_state_sampling_contract=sampling_contract,
        )
    elif leaf_id == "P0b":
        support_xy = (
            _P0_SUPPORT_HALF_XY_M[leaf_id]
            if tabletop_height_m > 0.0
            else (1.4, 1.4)
        )
        vx = 1.15
        vy = 0.34 if task_variant == "angled_projectile" else 0.0
        vz = 3.50
        key_event = 0.4
        sampling_contract = None
        if initial_state_mode == "sampled_scale":
            ranges = scale_parameter_ranges(leaf_id, task_variant)
            scale_rng = _scale_rng(seed)
            vx = float(scale_rng.uniform(*ranges["launch_speed_x_m_s"]))
            vy = float(scale_rng.uniform(*ranges["launch_speed_y_m_s"]))
            vz = float(scale_rng.uniform(*ranges["launch_speed_z_m_s"]))
            # The key event is the ballistic apex of the sampled launch.
            key_event = vz / abs(gravity[2])
            sampling_contract = _scale_contract(
                leaf_id,
                task_variant,
                seed=seed,
                sampled={
                    "sampled_launch_velocity_m_s": [vx, vy, vz],
                    "recomputed_key_event_time_s": key_event,
                },
            )
        base.update(
            duration_s=0.8,
            key_event_time_s=key_event,
            motion_kind="passive_projectile",
            object_initial_position_m=(-0.70, -0.15, tabletop_height_m + 0.50),
            object_initial_linear_velocity_m_s=(vx, vy, vz),
            surfaces=(_surface("supported_floor", "floor", (0, 0, r1_support_center_z), (*support_xy, r1_support_half_height)),),
            initial_state_sampling_contract=sampling_contract,
        )
    elif leaf_id == "P0c" and task_variant == "table_bounce":
        support_xy = (
            _P0_SUPPORT_HALF_XY_M[leaf_id]
            if tabletop_height_m > 0.0
            else (1.2, 0.8)
        )
        bounce_vx = 0.55
        bounce_vz = -0.85
        key_event = 0.31
        sampling_contract = None
        if initial_state_mode == "sampled_scale":
            ranges = scale_parameter_ranges(leaf_id, task_variant)
            scale_rng = _scale_rng(seed)
            factor = float(scale_rng.uniform(*ranges["speed_factor"]))
            bounce_vx *= factor
            bounce_vz *= factor
            # Same closed form as the fixed passive speed variation: first
            # table contact of the scaled launch from the fixed drop height.
            height = 0.75 - radius
            gravity_mag = abs(gravity[2])
            key_event = (
                bounce_vz + math.sqrt(bounce_vz**2 + 2.0 * gravity_mag * height)
            ) / gravity_mag
            sampling_contract = _scale_contract(
                leaf_id,
                task_variant,
                seed=seed,
                sampled={
                    "sampled_speed_factor": factor,
                    "sampled_launch_velocity_m_s": [bounce_vx, 0.0, bounce_vz],
                    "recomputed_key_event_time_s": key_event,
                },
            )
        base.update(
            duration_s=1.3,
            key_event_time_s=key_event,
            motion_kind="passive_table_bounce",
            object_initial_position_m=(-0.30, 0.0, tabletop_height_m + 0.75),
            object_initial_linear_velocity_m_s=(bounce_vx, 0.0, bounce_vz),
            surfaces=(
                _surface(
                    "supported_bounce_table",
                    "table",
                    (0, 0, r1_support_center_z),
                    (*support_xy, r1_support_half_height),
                    table_rebound=True,
                ),
            ),
            initial_state_sampling_contract=sampling_contract,
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
        wall_speed = 1.10
        sampling_contract = None
        if initial_state_mode == "sampled_scale":
            ranges = scale_parameter_ranges(leaf_id, task_variant)
            scale_rng = _scale_rng(seed)
            factor = float(scale_rng.uniform(*ranges["speed_factor"]))
            wall_speed *= factor
        wall_event_time = (0.35 - 0.02 - radius + 0.45) / wall_speed
        wall_launch_vz = 0.5 * abs(gravity[2]) * wall_event_time
        if initial_state_mode == "sampled_scale":
            sampling_contract = _scale_contract(
                leaf_id,
                task_variant,
                seed=seed,
                sampled={
                    "sampled_speed_factor": factor,
                    "sampled_launch_velocity_m_s": [
                        wall_speed,
                        0.0,
                        wall_launch_vz,
                    ],
                    "recomputed_key_event_time_s": wall_event_time,
                },
            )
        base.update(
            duration_s=1.3,
            key_event_time_s=wall_event_time,
            motion_kind="passive_wall_rebound",
            object_initial_position_m=(-0.45, 0.0, tabletop_height_m + 0.62),
            object_initial_linear_velocity_m_s=(
                wall_speed,
                0.0,
                wall_launch_vz,
            ),
            surfaces=(
                _surface("supported_floor", "floor", (0, 0, r1_support_center_z), (*support_xy, r1_support_half_height)),
                _surface("supported_wall", "wall", (0.35, 0.0, tabletop_height_m + 0.62), (0.02, wall_half_y, 0.62)),
            ),
            initial_state_sampling_contract=sampling_contract,
        )
    elif leaf_id == "P0d":
        slope = 0.10 if task_variant == "slope_roll" else 0.0
        speed = 0.72
        sampling_contract = None
        if initial_state_mode == "sampled_scale":
            ranges = scale_parameter_ranges(leaf_id, task_variant)
            scale_rng = _scale_rng(seed)
            factor = float(scale_rng.uniform(*ranges["speed_factor"]))
            speed *= factor
            sampling_contract = _scale_contract(
                leaf_id,
                task_variant,
                seed=seed,
                sampled={
                    "sampled_speed_factor": factor,
                    "sampled_rolling_speed_m_s": speed,
                },
            )
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
            initial_state_sampling_contract=sampling_contract,
        )
    elif leaf_id.startswith("F1") or leaf_id == "F2a":
        event_time = 0.44
        # The ballistic state is constructed so the ball center reaches
        # ``close_z`` at the closure event.  The physical/controller target
        # must be that same point; the previous catch_z target placed the palm
        # 55 mm below the actual event and let the ball sink into the hand.
        target = np.array((0.47, 0.0, close_z), dtype=np.float64)
        start_xy = target[:2].copy()
        initial_spin = (0.0, 0.0, 0.0)
        sampling_contract = None
        if initial_state_mode == "sampled_scale":
            if leaf_id not in _SCALE_BALLISTIC_RANGES:
                raise ValueError(
                    f"{leaf_id} has no scale ballistic initial-state policy"
                )
            (
                target,
                sampled_start,
                sampled_velocity,
                event_time,
                initial_spin,
                sampling_contract,
            ) = _sample_projectile_initial_state(
                leaf_id,
                seed=seed,
                target_z_m=close_z,
                gravity_z_m_s2=gravity[2],
                ranges=_SCALE_BALLISTIC_RANGES[leaf_id],
                contract_base=scale_randomization_contract(leaf_id, task_variant),
            )
            start_xy = sampled_start[:2].copy()
        elif leaf_id == "F1b":
            target[1] = 0.035
            start_xy = target[:2].copy()
        elif leaf_id == "F1c":
            start_xy += np.array((-0.09, 0.05))
        elif leaf_id in {"F1d", "F2a"} and initial_state_mode == "sampled_preview":
            (
                target,
                sampled_start,
                sampled_velocity,
                event_time,
                initial_spin,
                sampling_contract,
            ) = _sample_projectile_initial_state(
                leaf_id,
                seed=seed,
                target_z_m=close_z,
                gravity_z_m_s2=gravity[2],
            )
            start_xy = sampled_start[:2].copy()
        elif leaf_id in {"F1d", "F2a"}:
            if initial_state_mode != "fixed_review":
                raise ValueError(
                    f"unsupported projectile initial-state mode {initial_state_mode!r}"
                )
            start_xy += np.array((-0.12, -0.055))
        # Freeze the nominal physical velocity before applying a negative
        # initial-state intervention.  Recomputing it after shifting the start
        # position silently steered the object back to the grasp target and
        # turned every intended initial-state failure into a success.
        velocity_xy = (
            sampled_velocity[:2].copy()
            if sampling_contract is not None
            else (target[:2] - start_xy) / event_time
        )
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
        start_z = (
            float(sampled_start[2])
            if sampling_contract is not None
            else float(target[2]) + 0.5 * 9.81 * event_time**2
        )
        velocity_z = (
            float(sampled_velocity[2])
            if sampling_contract is not None
            else 0.0
        )
        transport = (
            (float(controller_target[0] + 0.12), float(controller_target[1]), catch_z + 0.06)
            if task_variant == "catch_transport"
            else None
        )
        base.update(
            key_event_time_s=event_time,
            motion_kind="direct_free_contact_interception",
            object_initial_position_m=(float(start_xy[0]), float(start_xy[1]), float(start_z)),
            object_initial_linear_velocity_m_s=(
                float(velocity_xy[0]),
                float(velocity_xy[1]),
                velocity_z,
            ),
            object_initial_angular_velocity_rad_s=initial_spin,
            ballistic_event_time_s=event_time,
            physical_target_position_m=tuple(float(value) for value in target),
            controller_target_position_m=tuple(float(value) for value in controller_target),
            controller_transport_position_m=transport,
            surfaces=(),
            initial_state_sampling_contract=sampling_contract,
        )
        if sampling_contract is not None:
            intervention = (
                "initial_state_lateral_shift"
                if negative and "initial_state" in branch_role
                else (
                    "controller_target_offset"
                    if negative and "controller" in branch_role
                    else "none"
                )
            )
            sampling_contract.update(
                {
                    "applied_initial_position_m": list(
                        base["object_initial_position_m"]
                    ),
                    "applied_initial_linear_velocity_m_s": list(
                        base["object_initial_linear_velocity_m_s"]
                    ),
                    "applied_initial_angular_velocity_rad_s": list(
                        base["object_initial_angular_velocity_rad_s"]
                    ),
                    "applied_physical_target_position_m": list(
                        base["physical_target_position_m"]
                    ),
                    "applied_controller_target_position_m": list(
                        base["controller_target_position_m"]
                    ),
                    "declared_intervention": intervention,
                }
            )
    elif leaf_id == "F2b":
        # A grounded incline redirects an admitted rolling initial state into
        # a projectile.  The ball traverses the physical ramp for ~0.49 s,
        # leaves its upper edge, and reaches the catch near the free-flight
        # apex.  Initial velocity/spin are part of the immutable initial state;
        # after initialization only contact and gravity move the object.
        ramp_pitch = math.radians(60.0)
        ramp_half = (0.35, 0.32, 0.02)
        start_local_x = 0.15
        edge_local_x = -ramp_half[0]
        travel = start_local_x - edge_local_x
        desired_exit_vertical_speed = 0.90
        rolling_deceleration = (
            5.0 / 7.0 * abs(gravity[2]) * math.sin(ramp_pitch)
        )
        exit_speed = desired_exit_vertical_speed / math.sin(ramp_pitch)
        initial_speed = math.sqrt(
            exit_speed**2 + 2.0 * rolling_deceleration * travel
        )
        support_time = (initial_speed - exit_speed) / rolling_deceleration
        launch_tangent = np.asarray(
            (-math.cos(ramp_pitch), 0.0, math.sin(ramp_pitch)),
            dtype=np.float64,
        )
        exit_velocity = launch_tangent * exit_speed
        free_flight_time = float(exit_velocity[2]) / abs(gravity[2])
        target_x = 0.47
        edge_top_z = 0.60
        normal = np.asarray(
            (math.sin(ramp_pitch), 0.0, math.cos(ramp_pitch)),
            dtype=np.float64,
        )
        ramp_center_x = target_x - (
            math.cos(ramp_pitch) * edge_local_x
            + math.sin(ramp_pitch) * ramp_half[2]
            + normal[0] * radius
            + exit_velocity[0] * free_flight_time
        )
        ramp_center_z = edge_top_z + (
            math.sin(ramp_pitch) * edge_local_x
            - math.cos(ramp_pitch) * ramp_half[2]
        )
        ramp_position = (ramp_center_x, 0.0, ramp_center_z)
        ramp = _surface(
            "owned_ramp_launch_surface",
            "ramp",
            ramp_position,
            ramp_half,
            euler=(0.0, ramp_pitch, 0.0),
        )

        def top_point(local_x: float) -> np.ndarray:
            return np.asarray(
                (
                    ramp_position[0]
                    + math.cos(ramp_pitch) * local_x
                    + math.sin(ramp_pitch) * ramp_half[2],
                    0.0,
                    ramp_position[2]
                    - math.sin(ramp_pitch) * local_x
                    + math.cos(ramp_pitch) * ramp_half[2],
                ),
                dtype=np.float64,
            )

        start = top_point(start_local_x) + normal * radius
        edge = top_point(edge_local_x) + normal * radius
        target = edge + exit_velocity * free_flight_time + np.asarray(
            (0.0, 0.0, 0.5 * gravity[2] * free_flight_time**2),
            dtype=np.float64,
        )
        target[0] = target_x
        event_time = support_time + free_flight_time
        controller_target = target.copy()
        if embodiment == ROBOTIQ_2F85_THICK_PAD:
            from .f2b_ramp_launch import SCENARIO as F2B_SCENARIO

            robotiq_bias = (
                F2B_SCENARIO.controller_plan.robotiq_controller_target_bias_m
            )
            if robotiq_bias is None:
                raise RuntimeError(
                    "F2b Robotiq controller lacks its calibrated target bias"
                )
            controller_target += np.asarray(robotiq_bias, dtype=np.float64)
        # Unlike a surface pickup, the ramp-launched ball is airborne at the
        # apex.  The Robotiq knuckles therefore need no table-clearance
        # standoff; adding the F3b 22 mm offset put the complete ball below the
        # physical pad corridor and converted the nominal seed into an arm hit.
        if negative:
            if "initial_state" in branch_role:
                start[1] += 0.135
            elif "controller" in branch_role:
                controller_target[1] += 0.10
            else:
                controller_target[0] -= 0.10
        supports = _grounded_supports_for_surface(
            ramp,
            prefix="owned_ramp_launch_surface",
            local_x_positions_m=(-0.27, 0.12),
            world_y_positions_m=(-0.27, 0.27),
        )
        transition = SurfaceTransitionContract(
            support_surface_id=ramp.name,
            support_normal_world_xyz=tuple(float(value) for value in normal),
            minimum_support_contact_s=0.08,
            minimum_free_flight_s=0.08,
        )
        base.update(
            duration_s=2.2,
            key_event_time_s=event_time,
            motion_kind="ramp_launch_pickup_interception",
            object_initial_position_m=tuple(float(value) for value in start),
            object_initial_linear_velocity_m_s=tuple(
                float(value) for value in launch_tangent * initial_speed
            ),
            object_initial_angular_velocity_rad_s=(
                0.0,
                float(-initial_speed / radius),
                0.0,
            ),
            ballistic_event_time_s=event_time,
            physical_target_position_m=tuple(float(value) for value in target),
            controller_target_position_m=tuple(
                float(value) for value in controller_target
            ),
            surfaces=(ramp, *supports),
            surface_transition_contract=transition,
        )
    elif leaf_id == "F3b":
        base["duration_s"] = 2.5
        event_time = RIGID_REVIEW_PROFILE.rolling_pickup_event_time_s
        speed = RIGID_REVIEW_PROFILE.rolling_pickup_speed_m_s
        lane_y = 0.0
        sampling_contract = None
        if initial_state_mode == "sampled_scale":
            ranges = scale_parameter_ranges(leaf_id, task_variant)
            scale_rng = _scale_rng(seed)
            speed_factor = float(scale_rng.uniform(*ranges["speed_factor"]))
            lane_y = float(scale_rng.uniform(*ranges["lane_y_m"]))
            speed *= speed_factor
            sampling_contract = _scale_contract(
                leaf_id,
                task_variant,
                seed=seed,
                sampled={
                    "sampled_speed_factor": speed_factor,
                    "sampled_rolling_speed_m_s": speed,
                    "sampled_lane_y_m": lane_y,
                },
            )
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
        target = np.array((0.47, lane_y, table_top_z + radius), dtype=np.float64)
        start_xy = np.array((float(target[0]) + speed * event_time, lane_y))
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
            initial_state_sampling_contract=sampling_contract,
        )
    elif leaf_id == "F2c":
        # Both variants share one measured launch: the ball strikes the
        # calibrated priority-2 bounce pad at exactly the profiled impact
        # speed and the post-bounce state follows the measured 1200 Hz pad
        # restitution.  The old plate design demanded a ~16 m/s impact
        # (guaranteed tunneling) to reach a 0.5 m catch through the dead
        # 0.20 wall constant; the repaired construction instead intercepts
        # near the measured bounce apex.
        gravity_mag = abs(gravity[2])
        impact = RIGID_REVIEW_PROFILE.bounce_impact_speed_m_s
        launch_vz = RIGID_REVIEW_PROFILE.bounce_launch_vz_m_s
        lane_vx = RIGID_REVIEW_PROFILE.bounce_lane_speed_m_s
        lane_y = 0.0
        sampling_contract = None
        if initial_state_mode == "sampled_scale":
            # The calibrated pad impact speed is never varied; only the
            # horizontal lane speed and its lateral offset are sampled.
            ranges = scale_parameter_ranges(leaf_id, task_variant)
            scale_rng = _scale_rng(seed)
            lane_factor = float(scale_rng.uniform(*ranges["lane_speed_factor"]))
            lane_y = float(scale_rng.uniform(*ranges["lane_y_m"]))
            lane_vx *= lane_factor
            sampling_contract = _scale_contract(
                leaf_id,
                task_variant,
                seed=seed,
                sampled={
                    "sampled_lane_speed_factor": lane_factor,
                    "sampled_lane_speed_m_s": lane_vx,
                    "sampled_lane_y_m": lane_y,
                },
            )
        fall_time = (impact + launch_vz) / gravity_mag
        drop = (impact**2 - launch_vz**2) / (2.0 * gravity_mag)
        vout = impact * RIGID_REVIEW_PROFILE.bounce_pad_effective_restitution
        rise_time = vout / gravity_mag
        rise = vout**2 / (2.0 * gravity_mag)
        # The saturated friction cone sticks the contact tangentially: the
        # measured post-bounce lane speed is the retained fraction, and the
        # ball exits rolling.
        lane_out = lane_vx * RIGID_REVIEW_PROFILE.bounce_pad_lane_retention
        pad_half_z = 0.016
        # Both variants are top-down apex pickups on the F3b lane (the ball
        # travels toward the robot along -X, entering the open finger
        # corridor), differing only in the calibrated pad's elevation.
        # Every other arrangement measurably failed: a palm-up catch of the
        # rising arc struck the gripper's underside 50 ms after the bounce,
        # a +Y apex lane drove the ball broadside into a hanging finger
        # post before the grasp point, and the low thrown-down lane clipped
        # the reaching arm.  The tossed arc crosses the workspace more than
        # a metre up, bounces, and hangs nearly stationary between the
        # descending fingers at the apex — the proven F3b grasp regime.
        pad_top = (
            RIGID_REVIEW_PROFILE.floor_bounce_pad_top_z_m
            if task_variant == "floor_bounce"
            else RIGID_REVIEW_PROFILE.table_bounce_pad_top_z_m
        )
        contact_z = pad_top + radius
        target = np.array((0.47, lane_y, contact_z + rise), dtype=np.float64)
        event_time = fall_time + rise_time
        bounce_x = float(target[0]) + lane_out * rise_time
        pad_half = (0.10, 0.18, pad_half_z)
        pad_center_x = bounce_x + 0.05
        motion_kind = (
            "bounce_apex_pickup_interception"
            if task_variant == "floor_bounce"
            else "table_bounce_apex_pickup_interception"
        )
        start_xy = np.array(
            (bounce_x + lane_vx * fall_time, lane_y),
            dtype=np.float64,
        )
        controller_target = target.copy()
        if embodiment == ROBOTIQ_2F85_THICK_PAD:
            controller_target[2] += RIGID_REVIEW_PROFILE.robotiq_pickup_standoff_m
            # The pads finish closing after the apex event while the ball
            # keeps moving along the lane at the measured retained speed;
            # the pinch center sits where the ball will be at full closure
            # (measured 0.063 s of pad contact before the centered pinch
            # slid off the still-moving ball).
            controller_target[0] -= lane_out * (
                RIGID_REVIEW_PROFILE.robotiq_pickup_closure_duration_s
                - RIGID_REVIEW_PROFILE.robotiq_pickup_closure_start_before_event_s
            )
            # The spinning rebound needs a slightly downstream jaw center to
            # remain nested as the tendon closes.  Without this measured bias
            # the nominal fixed seed is bilateral for 0.71 s but is squeezed
            # out before the final frame.
            controller_target[0] += (
                RIGID_REVIEW_PROFILE.f2c_robotiq_intercept_bias_x_m
            )
        if negative:
            # Keep the physical bounce and the intended difficult seed; a
            # negative changes only its declared intervention stream along
            # the lane's lateral axis, exactly like the F3b negatives.  The
            # pad spans the shifted lane so the declared rebound still
            # occurs.
            if "initial_state" in branch_role:
                start_xy[1] += 0.135
            elif "controller" in branch_role:
                controller_target[1] += 0.10
            else:
                controller_target[0] -= 0.10
        pad = _surface(
            "owned_bounce_pad",
            "table",
            (pad_center_x, 0.0, pad_top - pad_half_z),
            pad_half,
            rebound_pad=True,
        )
        # Both named variants place the calibrated contact plate above the
        # room floor (170 mm for floor_bounce, 400 mm for table_bounce).
        # Give both the same explicit four-leg support contract; a world-fixed
        # plate suspended in mid-air is not a physically supported fixture.
        leg_half = (0.015, 0.015, (pad_top - 2.0 * pad_half_z) / 2.0)
        supports = tuple(
            PhysicalSurface(
                name=f"owned_bounce_pad_leg_{index}",
                role="structural_support",
                position_m=(
                    pad_center_x + sign_x * (pad_half[0] - 0.015),
                    sign_y * (pad_half[1] - 0.015),
                    leg_half[2],
                ),
                half_size_m=leg_half,
                solref=(0.003, 1.0),
                expected_task_contact=False,
                supports_fixture_id="owned_bounce_pad",
                grounded_plane_z_m=0.0,
                support_interface_maximum_mismatch_m=0.0,
            )
            for index, (sign_x, sign_y) in enumerate(
                ((-1, -1), (-1, 1), (1, -1), (1, 1))
            )
        )
        base.update(
            key_event_time_s=event_time,
            motion_kind=motion_kind,
            object_initial_position_m=(
                float(start_xy[0]),
                float(start_xy[1]),
                float(contact_z + drop),
            ),
            object_initial_linear_velocity_m_s=(
                float(-lane_vx),
                0.0,
                float(launch_vz),
            ),
            ballistic_event_time_s=event_time,
            physical_target_position_m=tuple(float(value) for value in target),
            controller_target_position_m=tuple(
                float(value) for value in controller_target
            ),
            surfaces=(pad, *supports),
            initial_state_sampling_contract=sampling_contract,
        )
    elif leaf_id == "F2d":
        # The carom reuses the P0c-calibrated wall contact pair at its
        # proven 1.10 m/s impact speed.  The old construction dropped the
        # ball from a free-fall column that crossed the wall plane roughly
        # one metre above the wall top (measured: no wall contact at all,
        # a 4.2 mm floor penetration, and every key event out of frame).
        gravity_mag = abs(gravity[2])
        launch = RIGID_REVIEW_PROFILE.wall_rebound_launch_speed_m_s
        wall_time = RIGID_REVIEW_PROFILE.wall_rebound_wall_time_s
        post_time = RIGID_REVIEW_PROFILE.wall_rebound_post_time_s
        z_hit = RIGID_REVIEW_PROFILE.wall_rebound_hit_z_m
        e_wall = RIGID_REVIEW_PROFILE.wall_rebound_effective_restitution
        event_time = wall_time + post_time
        # The wall friction brakes the rising tangential velocity by the
        # measured retention while the ball is in contact.
        retention = RIGID_REVIEW_PROFILE.barrier_tangential_retention
        vz_wall = (
            close_z - z_hit + 0.5 * gravity_mag * post_time**2
        ) / (retention * post_time)
        vz0 = vz_wall + gravity_mag * wall_time
        start_z = z_hit - vz0 * wall_time + 0.5 * gravity_mag * wall_time**2
        # The ball rises along +Y past the robot's flank and caroms off a
        # laterally yawed wall so the retained tangential drift runs
        # x-dominant AWAY from the base — along the jaw slot toward the
        # fingertip taper, the regime the sealed F1c drift catches hold.
        # Every radial wall placement measurably failed: a robot-facing
        # wall's carom always drifts palm-ward and sank through the
        # Robotiq jaw, an across-slot drift knifed 4.6-6.6 mm into the
        # unmargined pads, and a wall between the base and the catch
        # overlapped the initialized arm.  The launch starts above
        # shoulder height so the steep rising leg clears the reaching arm.
        in_xy = np.array((0.0, launch), dtype=np.float64)
        wall_half = (0.18, 0.02, 0.78)
        if task_variant == "wall_rebound":
            # The Panda's open cage holds this straight wall's across-slot
            # drift (measured 1.07 s retention).  The Robotiq jaw does not:
            # its carom always drifts palm-ward for any radial wall and
            # measurably sank through the jaw, and every lateral-yaw wall
            # that turned the drift fingertip-ward either crossed the
            # pedestal envelope or left the settled jaw unverifiable.  F2d
            # therefore stays execution-blocked on the named Robotiq
            # retention defect while this construction remains the honest
            # shared recipe.
            yaw = 0.0
            aim_xy = np.array((0.47, 0.053), dtype=np.float64)
        else:
            yaw = -RIGID_REVIEW_PROFILE.angled_barrier_yaw_rad
            aim_xy = np.array((0.42, 0.0), dtype=np.float64)
        normal = np.array((math.sin(yaw), -math.cos(yaw)), dtype=np.float64)
        normal_speed_in = float(in_xy @ normal)
        tangent_xy = in_xy - normal_speed_in * normal
        out_xy = -e_wall * normal_speed_in * normal + retention * tangent_xy
        contact_xy = aim_xy - out_xy * post_time
        wall_center_xy = contact_xy - (wall_half[1] + radius) * normal
        wall_yaw = yaw
        catch_xy = contact_xy + out_xy * post_time
        target = np.array(
            (catch_xy[0], catch_xy[1], close_z), dtype=np.float64
        )
        start_xy = contact_xy - in_xy * wall_time
        controller_target = target.copy()
        if embodiment == FRANKA_HAND:
            # The open cage closes around the drifting carom and pins it at
            # closure completion; the measured successful catch centered the
            # cage on the ball's position at that instant, while centering
            # on the event-time position measurably let the ball drift past
            # the closing fingers.
            wrap_lead_s = (
                RIGID_REVIEW_PROFILE.closure_duration_s
                - RIGID_REVIEW_PROFILE.bounce_closure_start_before_event_s
            )
            controller_target[0] += float(out_xy[0]) * wrap_lead_s
            controller_target[1] += float(out_xy[1]) * wrap_lead_s
        if negative:
            # Same declared intervention streams as the F1/F2a negatives,
            # applied along the lane's lateral axis (X for the +Y lane);
            # the wall spans the shifted lane so the rebound still occurs.
            if "initial_state" in branch_role:
                start_xy[0] += 0.135
            elif "controller" in branch_role:
                from .f2d_wall_barrier_rebound import SCENARIO as F2D_SCENARIO

                negative_offset = (
                    F2D_SCENARIO.controller_plan.negative_controller_offset_m
                )
                if negative_offset is None:
                    raise RuntimeError(
                        "F2d controller-negative branch lacks a declared offset"
                    )
                controller_target += np.asarray(
                    negative_offset,
                    dtype=np.float64,
                )
            else:
                controller_target[1] -= 0.10
        base.update(
            key_event_time_s=event_time,
            motion_kind="wall_rebound_interception",
            object_initial_position_m=(
                float(start_xy[0]),
                float(start_xy[1]),
                float(start_z),
            ),
            object_initial_linear_velocity_m_s=(0.0, float(launch), float(vz0)),
            ballistic_event_time_s=event_time,
            physical_target_position_m=tuple(float(value) for value in target),
            controller_target_position_m=tuple(
                float(value) for value in controller_target
            ),
            surfaces=(
                _surface(
                    "supported_wall_rebound_barrier",
                    "wall",
                    (float(wall_center_xy[0]), float(wall_center_xy[1]), wall_half[2]),
                    wall_half,
                    euler=(0.0, 0.0, float(wall_yaw)),
                    rebound_wall=True,
                ),
            ),
        )
    elif leaf_id == "F2e":
        # Both variants persist the exact ordered fixture identities and
        # measured face normals.  The construction is analytical only at
        # initialization: MuJoCo contact determines every realized rebound
        # and a mismatched sequence remains a failed fixed-seed attempt.
        gravity_mag = abs(gravity[2])
        wall_restitution = RIGID_REVIEW_PROFILE.wall_rebound_effective_restitution
        wall_retention = RIGID_REVIEW_PROFILE.barrier_tangential_retention
        pad_restitution = RIGID_REVIEW_PROFILE.bounce_pad_effective_restitution
        pad_lane_retention = RIGID_REVIEW_PROFILE.bounce_pad_lane_retention
        wall_half = (0.32, 0.02, 0.78)
        pad_half = (0.22, 0.22, 0.016)
        pad_top = (
            0.40
            if task_variant == "floor_to_wall"
            else RIGID_REVIEW_PROFILE.table_bounce_pad_top_z_m
        )
        contact_z = pad_top + radius
        in_xy = np.asarray((0.0, 1.10), dtype=np.float64)

        if task_variant == "floor_to_wall":
            FLOOR_TO_WALL_REPAIR.validate()
            # The original 640 mm-wide wall extended far beyond the ball's
            # centered impact corridor and overlapped the initialized Panda
            # hand by 4.07 mm.  Retain the exact contact face and normal while
            # removing only that unused tangential extent from the robot
            # workspace.  The alternate F2e variant keeps its original wall.
            wall_half = (
                FLOOR_TO_WALL_REPAIR.wall_tangent_half_extent_m,
                wall_half[1],
                wall_half[2],
            )
            # A faster but still timestep-resolved floor lane creates enough
            # post-wall normal separation for the real finger geometry.  At
            # the earlier 1.1 m/s lane the calibrated low-restitution wall
            # left the grasp center only 17 mm from the face, so the arm
            # physically rested against the barrier during servo settling.
            in_xy = np.asarray((0.0, 3.0), dtype=np.float64)
            pad_time = 0.45
            wall_delay = 0.05
            post_time = 0.15
            impact_speed = RIGID_REVIEW_PROFILE.bounce_impact_speed_m_s
            yaw = math.radians(-40.0)
            normal = np.asarray(
                (math.sin(yaw), -math.cos(yaw)), dtype=np.float64
            )
            floor_out_xy = in_xy * pad_lane_retention
            normal_speed = float(floor_out_xy @ normal)
            tangent_xy = floor_out_xy - normal_speed * normal
            wall_out_xy = (
                -wall_restitution * normal_speed * normal
                + wall_retention * tangent_xy
            )
            pad_vz_out = impact_speed * pad_restitution
            wall_hit_z = (
                contact_z
                + pad_vz_out * wall_delay
                - 0.5 * gravity_mag * wall_delay**2
            )
            wall_vz_out = wall_retention * (
                pad_vz_out - gravity_mag * wall_delay
            )
            target_z = (
                wall_hit_z
                + wall_vz_out * post_time
                - 0.5 * gravity_mag * post_time**2
            )
            construction_target = np.asarray(
                (0.47, 0.0, target_z), dtype=np.float64
            )
            wall_contact_xy = construction_target[:2] - wall_out_xy * post_time
            pad_contact_xy = wall_contact_xy - floor_out_xy * wall_delay
            start_xy = pad_contact_xy - in_xy * pad_time
            initial_vz = -impact_speed + gravity_mag * pad_time
            start_z = (
                contact_z
                - initial_vz * pad_time
                + 0.5 * gravity_mag * pad_time**2
            )
            wall_center_xy = wall_contact_xy - (
                wall_half[1] + radius
            ) * normal
            ordered_ids = ("owned_multi_floor", "owned_multi_wall")
            ordered_normals = (
                (0.0, 0.0, 1.0),
                (float(normal[0]), float(normal[1]), 0.0),
            )
            # At the required 1200 Hz reference rate, the realized two-contact
            # trajectory reaches its catchable post-wall apex before the
            # analytical mixed-contact estimate.  Bind the physical/controller
            # target to that same-seed measured trajectory; do not move the
            # fixtures or resample an easier initial state.
            target = construction_target + np.asarray(
                FLOOR_TO_WALL_REPAIR.measured_target_bias_world_xyz_m,
                dtype=np.float64,
            )
            event_time = FLOOR_TO_WALL_REPAIR.catch_event_time_s
            pad = _surface(
                ordered_ids[0],
                "floor",
                (
                    float(pad_contact_xy[0]),
                    float(pad_contact_xy[1]),
                    pad_top - pad_half[2],
                ),
                pad_half,
                rebound_pad=True,
            )
            wall = _surface(
                ordered_ids[1],
                "wall",
                (
                    float(wall_center_xy[0]),
                    float(wall_center_xy[1]),
                    wall_half[2],
                ),
                wall_half,
                euler=(0.0, 0.0, yaw),
                rebound_wall=True,
            )
            surfaces = (
                pad,
                wall,
                *_grounded_supports_for_surface(
                    pad, prefix="owned_multi_floor"
                ),
            )
        else:
            wall_time = 0.30
            pad_time = 0.75
            between_time = pad_time - wall_time
            impact_speed = RIGID_REVIEW_PROFILE.bounce_impact_speed_m_s
            yaw = math.radians(-25.0)
            normal = np.asarray(
                (math.sin(yaw), -math.cos(yaw)), dtype=np.float64
            )
            normal_speed = float(in_xy @ normal)
            tangent_xy = in_xy - normal_speed * normal
            wall_out_xy = (
                -wall_restitution * normal_speed * normal
                + wall_retention * tangent_xy
            )
            rise_speed = impact_speed * pad_restitution
            rise_time = rise_speed / gravity_mag
            rise = rise_speed**2 / (2.0 * gravity_mag)
            table_out_xy = wall_out_xy * pad_lane_retention
            target = np.asarray(
                (0.47, 0.0, contact_z + rise), dtype=np.float64
            )
            pad_contact_xy = target[:2] - table_out_xy * rise_time
            wall_contact_xy = pad_contact_xy - wall_out_xy * between_time
            start_xy = wall_contact_xy - in_xy * wall_time
            pre_wall_vz = (
                -impact_speed + gravity_mag * between_time
            ) / wall_retention
            initial_vz = pre_wall_vz + gravity_mag * wall_time
            displacement_to_wall = (
                initial_vz * wall_time
                - 0.5 * gravity_mag * wall_time**2
            )
            displacement_to_pad = (
                wall_retention * pre_wall_vz * between_time
                - 0.5 * gravity_mag * between_time**2
            )
            start_z = contact_z - displacement_to_wall - displacement_to_pad
            wall_center_xy = wall_contact_xy - (
                wall_half[1] + radius
            ) * normal
            ordered_ids = ("owned_multi_wall", "owned_multi_table")
            ordered_normals = (
                (float(normal[0]), float(normal[1]), 0.0),
                (0.0, 0.0, 1.0),
            )
            event_time = pad_time + rise_time
            wall = _surface(
                ordered_ids[0],
                "wall",
                (
                    float(wall_center_xy[0]),
                    float(wall_center_xy[1]),
                    wall_half[2],
                ),
                wall_half,
                euler=(0.0, 0.0, yaw),
                rebound_wall=True,
            )
            pad = _surface(
                ordered_ids[1],
                "table",
                (
                    float(pad_contact_xy[0]),
                    float(pad_contact_xy[1]),
                    pad_top - pad_half[2],
                ),
                pad_half,
                rebound_pad=True,
            )
            surfaces = (
                wall,
                pad,
                *_grounded_supports_for_surface(
                    pad, prefix="owned_multi_table"
                ),
            )

        controller_target = target.copy()
        if embodiment == ROBOTIQ_2F85_THICK_PAD:
            if task_variant == "floor_to_wall":
                controller_target += np.asarray(
                    FLOOR_TO_WALL_REPAIR.robotiq_aim_bias_world_xyz_m,
                    dtype=np.float64,
                )
            else:
                controller_target[2] += (
                    RIGID_REVIEW_PROFILE.robotiq_pickup_standoff_m
                )
        if negative:
            if "initial_state" in branch_role:
                # Shift across the fixture width without changing the ordered
                # surface construction or selecting a replacement seed.
                start_xy[0] += 0.135
            elif "controller" in branch_role:
                if task_variant == "floor_to_wall":
                    # Move away from the angled wall and the incoming lane.
                    # The former +Y branch embedded both real grippers in the
                    # wall; this fixed -Y miss preserves the seed and label.
                    negative_aim_delta = (
                        FLOOR_TO_WALL_REPAIR.negative_controller_aim_delta_world_xyz_m
                    )
                    controller_target += np.asarray(
                        negative_aim_delta,
                        dtype=np.float64,
                    )
                else:
                    controller_target[1] += 0.10
            else:
                controller_target[0] -= 0.10
        ordered = OrderedContactContract(
            ordered_surface_ids=ordered_ids,
            ordered_surface_normals_world_xyz=ordered_normals,
            minimum_separated_pre_post_samples=2,
            minimum_inter_contact_free_flight_s=0.04,
        )
        base.update(
            duration_s=2.2,
            key_event_time_s=event_time,
            motion_kind="ordered_multi_rebound_pickup_interception",
            object_initial_position_m=(
                float(start_xy[0]),
                float(start_xy[1]),
                float(start_z),
            ),
            object_initial_linear_velocity_m_s=(
                float(in_xy[0]),
                float(in_xy[1]),
                float(initial_vz),
            ),
            ballistic_event_time_s=event_time,
            physical_target_position_m=tuple(float(value) for value in target),
            controller_target_position_m=tuple(
                float(value) for value in controller_target
            ),
            surfaces=surfaces,
            ordered_contact_contract=ordered,
        )
    elif leaf_id == "F2f":
        # Construction may select a deterministic review candidate before the
        # candidate is admitted.  The persisted admission flags remain false,
        # so source-spec validation and pilot/production execution fail closed
        # until the recorded 600/1200 Hz and clearance checks are complete.
        candidate = sample_surface_candidate(
            task_variant, source_seed=physics_seed
        )
        stable_id = f"owned_arbitrary_surface__{candidate.candidate_id}"
        gravity_mag = abs(gravity[2])
        if task_variant == "random_plane_bounce":
            impact = RIGID_REVIEW_PROFILE.bounce_impact_speed_m_s
            launch_vz = RIGID_REVIEW_PROFILE.bounce_launch_vz_m_s
            lane_vx = RIGID_REVIEW_PROFILE.bounce_lane_speed_m_s
            fall_time = (impact + launch_vz) / gravity_mag
            drop = (impact**2 - launch_vz**2) / (2.0 * gravity_mag)
            vout = impact * RIGID_REVIEW_PROFILE.bounce_pad_effective_restitution
            rise_time = vout / gravity_mag
            rise = vout**2 / (2.0 * gravity_mag)
            lane_out = lane_vx * RIGID_REVIEW_PROFILE.bounce_pad_lane_retention
            pad_top = candidate.position_m[2] + candidate.half_size_m[2]
            contact_z = pad_top + radius
            surface_offset_x = min(0.05, 0.4 * candidate.half_size_m[0])
            bounce_x = candidate.position_m[0] - surface_offset_x
            target = np.asarray(
                (
                    bounce_x - lane_out * rise_time,
                    candidate.position_m[1],
                    contact_z + rise,
                ),
                dtype=np.float64,
            )
            event_time = fall_time + rise_time
            position = candidate.position_m
            start_xy = np.asarray(
                (
                    bounce_x + lane_vx * fall_time,
                    candidate.position_m[1],
                ),
                dtype=np.float64,
            )
            controller_target = target.copy()
            if embodiment == ROBOTIQ_2F85_THICK_PAD:
                controller_target[2] += RIGID_REVIEW_PROFILE.robotiq_pickup_standoff_m
                controller_target[0] -= lane_out * (
                    RIGID_REVIEW_PROFILE.robotiq_pickup_closure_duration_s
                    - RIGID_REVIEW_PROFILE.robotiq_pickup_closure_start_before_event_s
                )
                controller_target[0] += F2F_ROBOTIQ_PLANE_INTERCEPT_BIAS_X_M
            if negative:
                if "initial_state" in branch_role:
                    start_xy[1] += 0.135
                elif "controller" in branch_role:
                    controller_target[1] += 0.10
                else:
                    controller_target[0] -= 0.10
            surface = _surface(
                stable_id,
                "table",
                position,
                candidate.half_size_m,
                euler=candidate.euler_rad,
                rebound_pad=True,
            )
            supports = _grounded_supports_for_surface(
                surface, prefix=stable_id
            )
            initial_position = (
                float(start_xy[0]),
                float(start_xy[1]),
                float(contact_z + drop),
            )
            initial_velocity = (-lane_vx, 0.0, launch_vz)
            motion_kind = "random_plane_bounce_pickup_interception"
        else:
            launch = RIGID_REVIEW_PROFILE.wall_rebound_launch_speed_m_s
            wall_time = RIGID_REVIEW_PROFILE.wall_rebound_wall_time_s
            post_time = RIGID_REVIEW_PROFILE.wall_rebound_post_time_s
            z_hit = F2F_BARRIER_HIT_Z_M
            restitution = RIGID_REVIEW_PROFILE.wall_rebound_effective_restitution
            retention = RIGID_REVIEW_PROFILE.barrier_tangential_retention
            yaw = candidate.euler_rad[2]
            in_xy = np.asarray((0.0, launch), dtype=np.float64)
            normal = np.asarray(
                (math.sin(yaw), -math.cos(yaw)), dtype=np.float64
            )
            normal_speed = float(in_xy @ normal)
            tangent_xy = in_xy - normal_speed * normal
            out_xy = -restitution * normal_speed * normal + retention * tangent_xy
            position = candidate.position_m
            contact_xy = np.asarray(position[:2], dtype=np.float64) + (
                candidate.half_size_m[1] + radius
            ) * normal
            target_xy = contact_xy + out_xy * post_time
            target = np.asarray(
                (target_xy[0], target_xy[1], F2F_BARRIER_CATCH_Z_M),
                dtype=np.float64,
            )
            start_xy = contact_xy - in_xy * wall_time
            vz_wall = (
                F2F_BARRIER_CATCH_Z_M
                - z_hit
                + 0.5 * gravity_mag * post_time**2
            ) / (retention * post_time)
            initial_vz = vz_wall + gravity_mag * wall_time
            start_z = z_hit - initial_vz * wall_time + 0.5 * gravity_mag * wall_time**2
            event_time = wall_time + post_time
            controller_target = target.copy()
            if embodiment == FRANKA_HAND:
                wrap_lead_s = (
                    RIGID_REVIEW_PROFILE.closure_duration_s
                    - RIGID_REVIEW_PROFILE.bounce_closure_start_before_event_s
                )
                controller_target[:2] += out_xy * wrap_lead_s
                controller_target += np.asarray(
                    F2F_FRANKA_BARRIER_INTERCEPT_BIAS_M,
                    dtype=np.float64,
                )
                controller_target += np.asarray(
                    F2F_FRANKA_BARRIER_CANDIDATE_BIAS_M.get(
                        candidate.candidate_id,
                        (0.0, 0.0, 0.0),
                    ),
                    dtype=np.float64,
                )
            if negative:
                if "initial_state" in branch_role:
                    start_xy[0] += 0.135
                elif "controller" in branch_role:
                    controller_target[0] += 0.10
                else:
                    controller_target[1] -= 0.10
            surface = _surface(
                stable_id,
                "wall",
                position,
                candidate.half_size_m,
                euler=candidate.euler_rad,
                rebound_wall=True,
            )
            supports = ()
            initial_position = (
                float(start_xy[0]),
                float(start_xy[1]),
                float(start_z),
            )
            initial_velocity = (0.0, float(launch), float(initial_vz))
            motion_kind = "arbitrary_surface_rebound_interception"
        sampled = sampled_surface_contract(
            candidate,
            source_seed=physics_seed,
        )
        base.update(
            key_event_time_s=event_time,
            motion_kind=motion_kind,
            object_initial_position_m=initial_position,
            object_initial_linear_velocity_m_s=initial_velocity,
            ballistic_event_time_s=event_time,
            physical_target_position_m=tuple(float(value) for value in target),
            controller_target_position_m=tuple(
                float(value) for value in controller_target
            ),
            surfaces=(surface, *supports),
            sampled_surface_contract=sampled,
        )
    else:
        raise SourceMujocoUnsupported(
            f"no physical source_mujoco recipe for {leaf_id}/{task_variant}"
        )
    contract = base.get("initial_state_sampling_contract")
    if contract is not None and "applied_initial_position_m" not in contract:
        # Bind the sampled contract to the exact compiled state, recorded
        # after any declared negative-branch intervention.  The projectile
        # block fills these fields itself; every other sampler shares this
        # single applied-state binding.
        intervention = "none"
        if negative:
            intervention = (
                "initial_state_lateral_shift"
                if "initial_state" in branch_role
                else "controller_target_offset"
                if "controller" in branch_role
                else "task_geometry_target_offset"
            )
        contract.update(
            {
                "applied_initial_position_m": list(
                    base["object_initial_position_m"]
                ),
                "applied_initial_linear_velocity_m_s": list(
                    base["object_initial_linear_velocity_m_s"]
                ),
                "applied_initial_angular_velocity_rad_s": list(
                    base["object_initial_angular_velocity_rad_s"]
                ),
                "applied_physical_target_position_m": list(
                    base["physical_target_position_m"] or ()
                ),
                "applied_controller_target_position_m": list(
                    base["controller_target_position_m"] or ()
                ),
                "declared_intervention": intervention,
            }
        )
    return base



def build_source_mujoco_recipe(context: ScenarioBuildContext):
    return _recipe_payload(
        context.leaf_id,
        context.task_variant,
        context.embodiment,
        context.branch_role,
        seed=context.seed,
        physics_seed=context.physics_seed,
        tabletop_height_m=context.tabletop_height_m,
        rolling_island_scene=context.rolling_island_scene,
        initial_state_mode=context.initial_state_mode,
    )


def rigid_module(
    leaf_id: str,
    family: str,
    subfamily: str,
    *,
    fixture_policy: str,
    controller_kind: str,
    trajectory: str,
    retention_required: bool,
    hand_orientation: str = "auto",
    settled_aim_correction: bool = False,
    compact_pickup_ready: bool = False,
    robotiq_reach_arrival_lead_s: float | None = None,
    robotiq_tendon_profile: str = "default",
    robotiq_tendon_target: float | None = None,
    robotiq_actuator_force_limit_n: float | None = None,
    robotiq_pad_half_depth_m: float | None = None,
    robotiq_pad_contact_margin_m: float | None = None,
    robotiq_controller_target_bias_m: tuple[float, float, float] | None = None,
    negative_controller_offset_m: tuple[float, float, float] | None = None,
    interior_joint_margin_rad: float = 0.0,
    randomization_contract: dict[str, Any] | None = None,
    sampled_projectile_ready_offset_m: tuple[float, float, float] = (
        0.0,
        0.0,
        0.0,
    ),
) -> ScenarioModuleSpec:
    return ScenarioModuleSpec(
        leaf_id=leaf_id,
        family=family,
        subfamily=subfamily,
        backend="source_mujoco",
        fixture_policy=fixture_policy,
        controller_plan=ControllerPlan(
            kind=controller_kind,
            trajectory=trajectory,
            retention_required=retention_required,
            hand_orientation=hand_orientation,
            settled_aim_correction=settled_aim_correction,
            compact_pickup_ready=compact_pickup_ready,
            robotiq_reach_arrival_lead_s=robotiq_reach_arrival_lead_s,
            robotiq_tendon_profile=robotiq_tendon_profile,
            robotiq_tendon_target=robotiq_tendon_target,
            robotiq_actuator_force_limit_n=robotiq_actuator_force_limit_n,
            robotiq_pad_half_depth_m=robotiq_pad_half_depth_m,
            robotiq_pad_contact_margin_m=robotiq_pad_contact_margin_m,
            robotiq_controller_target_bias_m=(
                robotiq_controller_target_bias_m
            ),
            negative_controller_offset_m=negative_controller_offset_m,
            interior_joint_margin_rad=interior_joint_margin_rad,
            sampled_projectile_ready_offset_m=(
                sampled_projectile_ready_offset_m
            ),
        ),
        build_recipe=build_source_mujoco_recipe,
        implementation_note="canonical source_mujoco fixed-review recipe",
        randomization_contract=randomization_contract,
    )


def blocked_module(
    leaf_id: str,
    family: str,
    subfamily: str,
    backend: str,
    *,
    fixture_policy: str,
    controller_kind: str,
    implementation_note: str,
) -> ScenarioModuleSpec:
    return ScenarioModuleSpec(
        leaf_id=leaf_id,
        family=family,
        subfamily=subfamily,
        backend=backend,
        fixture_policy=fixture_policy,
        controller_plan=ControllerPlan(
            kind=controller_kind,
            trajectory="not_implemented_fail_closed",
            hand_orientation="none",
        ),
        build_recipe=None,
        implementation_note=implementation_note,
    )


__all__ = [
    "PROJECTILE_INITIAL_STATE_SAMPLER_VERSION",
    "SCALE_ACCEPTED_SAMPLER_VERSIONS",
    "SCALE_INITIAL_STATE_SAMPLER_VERSION",
    "SCALE_SAMPLED_LEAVES",
    "scale_sampler_version",
    "blocked_module",
    "build_source_mujoco_recipe",
    "projectile_randomization_contract",
    "rigid_module",
    "scale_parameter_ranges",
    "scale_randomization_contract",
]
