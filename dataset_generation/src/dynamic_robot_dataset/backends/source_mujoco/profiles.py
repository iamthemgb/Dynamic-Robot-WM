"""Calibrated, review-only profiles for the owned ``source_mujoco`` backend.

The constants in this module are evidence, not release claims.  They preserve
the July 2026 timestep/contact experiments which motivated this backend while
the fixed six-case review suite is still outstanding.  Runtime code validates
the constants before compiling a model and persists the complete profile in
every result.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

from ...common.grasp_retention import (
    DEFAULT_GRASP_RETENTION,
    GraspRetentionThresholds,
)
from ...common.rebound import (
    DEFAULT_REBOUND_ACCEPTANCE,
    ReboundAcceptanceThresholds,
)


SOURCE_MUJOCO_PROFILE_SCHEMA = "dynamic-robot-source-mujoco-profile/v1"
SOURCE_MUJOCO_PROFILE_VERSION = "source-mujoco-rigid-review-2026-07-v15"


@dataclass(frozen=True, slots=True)
class RigidReviewProfile:
    """One fail-closed rigid solver/controller calibration profile."""

    profile_id: str = SOURCE_MUJOCO_PROFILE_VERSION
    simulation_hz: int = 600
    comparison_simulation_hz: int = 1200
    control_hz: int = 60
    video_hz: int = 30
    width: int = 832
    height: int = 480
    wall_solref: tuple[float, float] = (0.012, 0.7)
    table_rebound_solref: tuple[float, float] = (0.0045, 0.42)
    wall_effective_restitution: float = 0.20
    minimum_rebound_effective_restitution: float = (
        DEFAULT_REBOUND_ACCEPTANCE.minimum_effective_restitution
    )
    minimum_rebound_outgoing_normal_speed_m_s: float = (
        DEFAULT_REBOUND_ACCEPTANCE.minimum_outgoing_normal_speed_m_s
    )
    minimum_rebound_normal_separation_m: float = (
        DEFAULT_REBOUND_ACCEPTANCE.minimum_normal_separation_m
    )
    minimum_rebound_normal_separation_radius_fraction: float = (
        DEFAULT_REBOUND_ACCEPTANCE.minimum_normal_separation_radius_fraction
    )
    minimum_rebound_separation_duration_s: float = (
        DEFAULT_REBOUND_ACCEPTANCE.minimum_separation_duration_s
    )
    robotiq_pad_condim: int = 3
    robotiq_pad_friction: tuple[float, float, float] = (1.5, 0.005, 0.0001)
    robotiq_pad_solref: tuple[float, float] = (0.012, 0.7)
    # The external prototype's 44 mm-deep invisible boxes stopped a rolling
    # ball on their leading corners before it entered the physical jaw gap.
    # v13 retains the measured 20 mm full pad depth and the visual mesh.
    robotiq_pad_half_depth_m: float = 0.010
    robotiq_passive_finger_acceleration_limit_rad_s2: float = 200.0
    # v2: the physical thick-pad midpoint replaces the historical base site.
    # 115 is the smallest tested bounded command whose fixed-review contact
    # event agrees at 600 and 1200 Hz within the 1 cm / one-frame gate.
    robotiq_tendon_target: float = 115.0
    robotiq_tendon_force_range_n: tuple[float, float] = (-0.16, 0.16)
    closure_start_before_ballistic_s: float = 0.16
    closure_duration_s: float = 0.10
    # v9 reaching-catch controller: the arm initializes at a ready waypoint
    # hovering above the intercept and descends onto it with a minimum-jerk
    # ctrl-only trajectory that ends before the ballistic event.  Feasibility
    # uses the published Franka per-joint limits at a safety fraction so the
    # recorded commands remain replayable on real hardware.
    ready_hover_above_intercept_m: float = 0.045
    reach_arrival_before_ballistic_s: float = 0.055
    minimum_reach_duration_s: float = 0.18
    reach_limit_safety_fraction: float = 0.8
    franka_joint_velocity_limit_rad_s: tuple[float, ...] = (
        2.175,
        2.175,
        2.175,
        2.175,
        2.610,
        2.610,
        2.610,
    )
    franka_joint_acceleration_limit_rad_s2: tuple[float, ...] = (
        15.0,
        7.5,
        10.0,
        12.5,
        15.0,
        20.0,
        20.0,
    )
    minimum_arm_command_travel_rad: float = 0.01
    maximum_reach_arrival_distance_m: float = 0.025
    # v13 rolling-island pickup.  The 0.8 s horizon matches the collaborator's
    # visible reach-and-intercept staging while leaving enough time for a
    # limit-checked actuator trajectory.  The ball rolls without slip on the
    # real counter top; there is no miniature runway or artificial backstop.
    rolling_pickup_speed_m_s: float = 0.38
    rolling_pickup_event_time_s: float = 0.95
    # Start clearly retracted and raised, then reach through the arm actuators.
    # These are the largest tested offsets that fit the embodiment-specific
    # 0.80 s Robotiq / 0.895 s Panda arrival deadlines and strict limits.
    pickup_ready_retract_x_m: float = 0.08
    pickup_ready_raise_z_m: float = 0.12
    robotiq_pickup_ready_retract_x_m: float = 0.05
    robotiq_pickup_ready_raise_z_m: float = 0.09
    robotiq_pickup_reach_arrival_before_event_s: float = 0.15
    robotiq_pickup_closure_start_before_event_s: float = 0.08
    robotiq_pickup_closure_duration_s: float = 0.15
    robotiq_pickup_tendon_target: float = 140.0
    robotiq_pickup_capture_followthrough_x_m: float = 0.035
    robotiq_pickup_capture_start_s: float = 1.02
    robotiq_pickup_capture_end_s: float = 1.30
    # A catch is not successful merely because it pinches the ball briefly.
    # The last 100 ms must remain bilateral and transform-stable so the fixed
    # final review frame cannot show a nominal object falling to the floor.
    final_retention_window_s: float = DEFAULT_GRASP_RETENTION.final_window_s
    minimum_final_bilateral_fraction: float = (
        DEFAULT_GRASP_RETENTION.minimum_bilateral_fraction
    )
    # A pickup success must exhibit contact-supported displacement, so both
    # F3b variants lift well beyond the 0.06 m transport-evidence threshold
    # (the slow transport quintic places 0.702 of the lift inside the fixed
    # [1.0, 1.6] s evidence window and the servo tracks 0.78 of the command,
    # so 0.12 m yields a measured ~0.066 m).  0.15 m measurably excited
    # 108 rad/s^2 wrist servo transients from the intrinsic palm-down wrist
    # rotation of a vertical lift.
    pickup_lift_height_m: float = 0.12
    # The transport variant carries the pickup laterally as well as up.  The
    # offset is bounded by the same wrist servo-transient physics as the
    # lift: 0.12 m measurably excited 117 rad/s^2 on the empty-hand
    # deterministic negatives.
    pickup_transport_lateral_m: float = 0.08
    # The Robotiq 2f85 finger structure extends below the thick-pad midpoint:
    # commanding the pad midpoint to ball-center height above a support
    # measurably bottomed the knuckles out on the runway (a 148 rad/s^2
    # shoulder slam) 21 mm above the target.  The commanded pickup intercept
    # therefore stands off by the measured geometric clearance and pinches
    # the ball's upper hemisphere, which the retention evidence validates.
    robotiq_pickup_standoff_m: float = 0.022
    # The pickup transport uses the slowest feasible quintic over the whole
    # post-grasp window after the 0.95 s rolling interception.  The default
    # 0.6 s falling-catch transport window measurably excited 127 rad/s^2
    # wrist servo transients on the deep-fold lift.
    pickup_transport_start_s: float = 1.08
    pickup_transport_end_s: float = 2.43
    robotiq_pickup_transport_start_s: float = 1.30
    robotiq_pickup_transport_end_s: float = 2.45
    # v15 bounce/rebound interception (F2c/F2d).  The bounce pad is a
    # priority-2 owned fixture emitted like the room floor and hand shells.
    # The margin-sweep calibration measured that mixed-pair plates either
    # exceed the 3 mm geometric-penetration gate above ~3.5 m/s or, when
    # underdamped below ~0.5 damping ratio, inject energy through the 4 mm
    # predictive margin (measured effective restitution up to 45.0 at
    # 0.5 m/s: the margin acts as an undamped catapult).  (0.005, 0.5)
    # keeps every measured construction-speed restitution inside
    # [0.15, 1.05] with zero recorded geometric penetration at 1200 Hz.
    bounce_pad_solref: tuple[float, float] = (0.005, 0.5)
    bounce_pad_contact_margin_m: float = 0.004
    # Measured pad restitution at the fixed 4.5 m/s construction impact,
    # pinned from the compiled F2c scene at the 1200 Hz reference rate
    # (isolated drop rigs measured 0.301; the full compiled contact stack
    # measures 0.2466 and the recipe designs against the compiled value).
    # The 600 Hz candidate under-resolves the stiff pad contact with
    # step-phase jitter, so every F2c fixed class executes at the
    # reference rate.
    bounce_pad_effective_restitution: float = 0.2466
    # Measured post-bounce lane-speed retention: the large normal impulse
    # saturates the friction cone, the contact sticks tangentially, and the
    # ball exits rolling at the measured fraction of its incoming lane
    # speed (clean off-lane measurement: 0.6329 m/s out of 0.85 m/s in).
    bounce_pad_lane_retention: float = 0.7446
    bounce_impact_speed_m_s: float = 4.5
    # The 140-unit rolling-pickup tendon command over-closes on the spinning
    # F2c rebound and squeezes the fixed Robotiq nominal ball out at 1.58 s.
    # The measured 125-unit command plus a 15 mm downstream aim correction
    # retains the same fixed seed through 2.0 s with 1.835 mm peak gripper
    # penetration.  Keep these F2c-specific so accepted F3b behavior is not
    # changed by the rebound repair.
    f2c_robotiq_tendon_target: float = 125.0
    f2c_robotiq_intercept_bias_x_m: float = 0.015
    # Both variants toss the ball upward: the top-down apex pickup needs the
    # ~0.70 s Franka-feasible reach, which measurably did not fit before a
    # thrown-down ball's 0.47 s apex event.  A palm-up catch of the rising
    # post-bounce arc is not physically available at this pad's measured
    # restitution: the riser struck the hovering gripper's underside 50 ms
    # after every bounce.
    bounce_launch_vz_m_s: float = 3.0
    # Rebound/bounce interceptions keep the finger cage open at arrival and
    # close it around the event: the standard pre-closed fingertip wedge
    # measurably deflected both the slow apex arrivals and the drifting
    # wall carom instead of capturing them.
    bounce_closure_start_before_event_s: float = 0.05
    # The lane is fast enough that the post-bounce travel clears the pad
    # laterally: at 0.45 m/s the palm bottom sits ~1 cm above the plate top
    # with overlapping x-extents at every pad height, and the settled arm
    # measurably rested against the plate edge instead of its commanded
    # aim point.
    bounce_lane_speed_m_s: float = 0.85
    # A 0.032 m floor pad put the Robotiq apex pinch at a 0.15 m fold the
    # arm measurably cannot reach inside any legal toss window (1.02 s
    # Franka-feasible reach vs at most a 0.88 s arrival deadline); the low
    # riser keeps the floor-level bounce semantics inside the proven
    # pickup-reach envelope.
    floor_bounce_pad_top_z_m: float = 0.17
    # The taller pad places the post-bounce catch near z=0.48, inside the
    # proven catch-extension band; at a 0.30 m pad the settled low folded
    # posture measurably rested against the plate edge or could not verify
    # its servo-settled aim.
    table_bounce_pad_top_z_m: float = 0.40
    # F2d reuses the P0c-calibrated wall contact pair at its proven
    # 1.10 m/s impact speed; 0.20 effective restitution is the existing
    # calibrated wall constant.  The arc meets the wall at 1.30 m while
    # still rising so the carom drops into the fixed catch point.
    wall_rebound_launch_speed_m_s: float = 1.10
    wall_rebound_wall_time_s: float = 0.35
    wall_rebound_post_time_s: float = 0.45
    wall_rebound_hit_z_m: float = 1.50
    # Measured F2d barrier restitution at the fixed 1.10 m/s impact under
    # the 6 mm predictive wall margin.  The P0c 2 mm margin recorded a
    # 3.51 mm geometric wall penetration on the F2d arc and 4 mm still
    # recorded 3.09 mm, both past the 3 mm gate; the catch is placed from
    # the measured compiled-scene restitution at the final margin (an
    # earlier 0.1795 pin from the 4 mm measurement left the carom 1.4 cm
    # off-aim and rammed a single pad edge 6.6 mm deep).
    wall_rebound_contact_margin_m: float = 0.006
    wall_rebound_effective_restitution: float = 0.1869
    angled_barrier_yaw_rad: float = 0.4363323129985824
    # Measured tangential speed retention through the barrier contact,
    # pinned from the compiled-scene probe: the wall friction braked the
    # rising tangential velocity from 0.584 to 0.472 m/s.  The same
    # retention applies to the yawed barrier's in-plane tangent.
    barrier_tangential_retention: float = 0.808
    maximum_gripper_penetration_m: float = 0.002
    maximum_task_surface_penetration_m: float = 0.003
    maximum_effective_restitution: float = 1.05
    maximum_free_flight_energy_drift_fraction: float = 0.05
    panda_ball_radius_range_m: tuple[float, float] = (0.024, 0.0245)
    robotiq_calibration_seed: int = 7401
    robotiq_calibration_penetration_m: float = 0.001547687056695994
    robotiq_first_bilateral_contact_600_s: float = 0.43666666666753823
    robotiq_first_bilateral_contact_1200_s: float = 0.4366666666664955
    robotiq_final_position_delta_m: float = 0.004186892649805686
    wall_600_1200_outcome_match: bool = True
    # Fixed review classes for which the 600 Hz candidate failed strict QC or
    # the one-frame/1 cm timestep-halving agreement gate.  They execute at the
    # calibrated 1200 Hz reference rather than receiving a relaxed
    # penetration or event-evidence threshold.  The two v9 Robotiq entries
    # measured first-bilateral-contact sample shifts of 12.3 mm and 21.8 mm
    # under the reaching controller while outcome/QC/replay still agreed at
    # both rates.
    reference_rate_required_case_classes: tuple[str, ...] = (
        "P0c/lower_initial_speed",
        "F1a/franka_hand/deterministic_negative_controller_timing",
        "F1b/franka_hand/deterministic_negative_controller_timing",
        "F2a/franka_hand/deterministic_negative_controller_timing",
        "F1a/robotiq_2f85_thick_pad/nominal_success",
        "F1d/robotiq_2f85_thick_pad/nominal_success",
        "F2a/robotiq_2f85_thick_pad/nominal_success",
        # The 600 Hz candidate under-resolves the stiff F2c bounce-pad
        # contact: measured pad restitution 0.216 vs 0.301 at the reference
        # rate flips the designed apex interception into a measured miss, a
        # semantic 600 Hz defect like the listed P0c low-speed wall class.
        "F2c/franka_hand/nominal_success",
        "F2c/robotiq_2f85_thick_pad/nominal_success",
        "F2c/franka_hand/deterministic_negative_initial_state",
        "F2c/robotiq_2f85_thick_pad/deterministic_negative_initial_state",
        "F2c/franka_hand/deterministic_negative_controller_timing",
        "F2c/robotiq_2f85_thick_pad/deterministic_negative_controller_timing",
        # F2d's repaired Robotiq controller-negative branch is an honest miss
        # at both rates, but its subsequent floor impact measures 4.458 mm
        # penetration at 600 Hz versus 1.845 mm at the 1200 Hz reference.
        "F2d/robotiq_2f85_thick_pad/deterministic_negative_controller_timing",
        # Every F2e fixed class: the two-contact trajectory is phase-sensitive
        # at 600 Hz.  Same-seed planned-event positions differ from the 1200 Hz
        # reference by 39--59 mm, and both nominal outcomes flip to misses.
        "F2e/franka_hand/nominal_success",
        "F2e/robotiq_2f85_thick_pad/nominal_success",
        "F2e/franka_hand/deterministic_negative_initial_state",
        "F2e/robotiq_2f85_thick_pad/deterministic_negative_initial_state",
        "F2e/franka_hand/deterministic_negative_controller_timing",
        "F2e/robotiq_2f85_thick_pad/deterministic_negative_controller_timing",
    )
    release_state: str = "blocked"
    schema_version: str = SOURCE_MUJOCO_PROFILE_SCHEMA

    def validate(self) -> None:
        if self.schema_version != SOURCE_MUJOCO_PROFILE_SCHEMA:
            raise ValueError("unsupported source_mujoco profile schema")
        if self.release_state != "blocked":
            raise ValueError("the review profile must not activate a release state")
        if (self.simulation_hz, self.comparison_simulation_hz) != (600, 1200):
            raise ValueError("the calibrated rigid profile is fixed at 600/1200 Hz")
        if self.simulation_hz % self.control_hz or self.simulation_hz % self.video_hz:
            raise ValueError("simulation rate must divide control and video schedules exactly")
        if (self.width, self.height, self.video_hz) != (832, 480, 30):
            raise ValueError("canonical source media must remain 832x480 at 30 Hz")
        if self.wall_solref != (0.012, 0.7):
            raise ValueError("wall restitution calibration changed without a profile version")
        if self.table_rebound_solref != (0.0045, 0.42):
            raise ValueError(
                "table rebound calibration changed without a profile version"
            )
        if self.robotiq_pad_condim != 3:
            raise ValueError("Robotiq thick pads must use condim=3")
        if not (
            0.9 <= self.robotiq_pad_friction[0] <= 2.0
            and self.robotiq_pad_friction[1:] == (0.005, 0.0001)
        ):
            raise ValueError("Robotiq thick-pad friction is outside v15 calibration")
        if not 0.010 <= self.robotiq_pad_half_depth_m <= 0.022:
            raise ValueError("Robotiq collision-pad half depth is outside calibration")
        if not 100.0 <= self.robotiq_passive_finger_acceleration_limit_rad_s2 <= 250.0:
            raise ValueError("Robotiq passive-finger acceleration limit is invalid")
        if self.robotiq_tendon_force_range_n != (-0.16, 0.16):
            raise ValueError("Robotiq calibrated tendon force range must remain +/-0.16 N")
        if not 0.0 < self.closure_duration_s <= self.closure_start_before_ballistic_s:
            raise ValueError("closure timing is not a bounded pre-intercept trajectory")
        if not 0.0 < self.ready_hover_above_intercept_m <= 0.10:
            raise ValueError("ready hover must place the hand slightly above the intercept")
        if not (
            0.0
            < self.reach_arrival_before_ballistic_s
            < self.closure_start_before_ballistic_s
            + self.minimum_reach_duration_s
        ):
            raise ValueError(
                "the reach must still be moving when the bounded closure starts "
                "and must arrive before the ballistic event"
            )
        if not 0.0 < self.reach_limit_safety_fraction <= 1.0:
            raise ValueError("reach limit safety fraction must lie in (0, 1]")
        if len(self.franka_joint_velocity_limit_rad_s) != 7 or len(
            self.franka_joint_acceleration_limit_rad_s2
        ) != 7:
            raise ValueError("Franka feasibility limits must cover all seven arm joints")
        if any(
            value <= 0.0
            for value in (
                *self.franka_joint_velocity_limit_rad_s,
                *self.franka_joint_acceleration_limit_rad_s2,
                self.minimum_arm_command_travel_rad,
                self.maximum_reach_arrival_distance_m,
            )
        ):
            raise ValueError("reaching-controller limits must be positive")
        if self.robotiq_calibration_penetration_m > self.maximum_gripper_penetration_m:
            raise ValueError("persisted Robotiq calibration exceeds the 2 mm admission limit")
        if not 0.0 < self.rolling_pickup_speed_m_s <= 1.0:
            raise ValueError("rolling pickup speed must be a bounded positive roll")
        if not 0.0 < self.rolling_pickup_event_time_s <= 1.0:
            raise ValueError("rolling pickup event horizon must be positive and bounded")
        if not 0.0 <= self.pickup_ready_retract_x_m <= 0.20:
            raise ValueError("pickup ready-pose retraction must be bounded")
        if not 0.05 <= self.pickup_ready_raise_z_m <= 0.20:
            raise ValueError("pickup ready pose must visibly rise above the intercept")
        if not 0.0 <= self.robotiq_pickup_ready_retract_x_m <= 0.20:
            raise ValueError("Robotiq pickup ready-pose retraction must be bounded")
        if not 0.05 <= self.robotiq_pickup_ready_raise_z_m <= 0.20:
            raise ValueError("Robotiq pickup ready pose must visibly rise")
        if not 0.05 <= self.robotiq_pickup_reach_arrival_before_event_s <= 0.25:
            raise ValueError("Robotiq pickup arrival lead must be bounded")
        if not (
            0.0
            < self.robotiq_pickup_closure_start_before_event_s
            < self.robotiq_pickup_closure_duration_s
        ):
            raise ValueError("Robotiq pickup closure must straddle the intercept")
        if not self.closure_duration_s < self.robotiq_pickup_closure_duration_s <= 0.20:
            raise ValueError("Robotiq pickup closure must remain smooth and bounded")
        if not self.robotiq_tendon_target < self.robotiq_pickup_tendon_target <= 200.0:
            raise ValueError("Robotiq pickup tendon target must add bounded retention")
        if self.grasp_retention() != DEFAULT_GRASP_RETENTION:
            raise ValueError(
                "grasp-retention thresholds changed without a profile/evaluator version"
            )
        if self.f2c_robotiq_tendon_target != 125.0:
            raise ValueError("F2c Robotiq tendon target differs from v15 calibration")
        if self.f2c_robotiq_intercept_bias_x_m != 0.015:
            raise ValueError("F2c Robotiq intercept bias differs from v15 calibration")
        if not 0.0 < self.robotiq_pickup_capture_followthrough_x_m <= 0.06:
            raise ValueError("Robotiq capture follow-through must be small and positive")
        if not (
            self.rolling_pickup_event_time_s
            < self.robotiq_pickup_capture_start_s
            < self.robotiq_pickup_capture_end_s
            <= self.robotiq_pickup_transport_start_s
            < self.robotiq_pickup_transport_end_s
            <= 2.5
        ):
            raise ValueError("Robotiq capture/lift phases are not ordered")
        if not 0.10 <= self.pickup_lift_height_m <= 0.30:
            raise ValueError(
                "pickup lift must exceed the transport-displacement evidence floor"
            )
        if not 0.0 < self.robotiq_pickup_standoff_m <= 0.05:
            raise ValueError(
                "Robotiq pickup standoff must be a small positive clearance"
            )
        if not 0.0 < self.pickup_transport_lateral_m <= 0.20:
            raise ValueError(
                "pickup transport lateral carry must be positive and bounded"
            )
        if not (
            self.rolling_pickup_event_time_s
            < self.pickup_transport_start_s
            < 1.10
            < 1.6
            < self.pickup_transport_end_s
            <= 3.0
        ):
            raise ValueError(
                "pickup transport must start after the grasp event and span "
                "the fixed displacement-evidence window"
            )
        if not self.wall_600_1200_outcome_match:
            raise ValueError("600 Hz cannot be admitted when the 1200 Hz outcome differs")
        if self.reference_rate_required_case_classes != (
            "P0c/lower_initial_speed",
            "F1a/franka_hand/deterministic_negative_controller_timing",
            "F1b/franka_hand/deterministic_negative_controller_timing",
            "F2a/franka_hand/deterministic_negative_controller_timing",
            "F1a/robotiq_2f85_thick_pad/nominal_success",
            "F1d/robotiq_2f85_thick_pad/nominal_success",
            "F2a/robotiq_2f85_thick_pad/nominal_success",
            "F2c/franka_hand/nominal_success",
            "F2c/robotiq_2f85_thick_pad/nominal_success",
            "F2c/franka_hand/deterministic_negative_initial_state",
            "F2c/robotiq_2f85_thick_pad/deterministic_negative_initial_state",
            "F2c/franka_hand/deterministic_negative_controller_timing",
            "F2c/robotiq_2f85_thick_pad/deterministic_negative_controller_timing",
            "F2d/robotiq_2f85_thick_pad/deterministic_negative_controller_timing",
            "F2e/franka_hand/nominal_success",
            "F2e/robotiq_2f85_thick_pad/nominal_success",
            "F2e/franka_hand/deterministic_negative_initial_state",
            "F2e/robotiq_2f85_thick_pad/deterministic_negative_initial_state",
            "F2e/franka_hand/deterministic_negative_controller_timing",
            "F2e/robotiq_2f85_thick_pad/deterministic_negative_controller_timing",
        ):
            raise ValueError("reference-rate exception classes changed without calibration")
        if self.bounce_pad_solref != (0.005, 0.5):
            raise ValueError("bounce pad calibration changed without a profile version")
        if not (
            0.0
            < self.bounce_pad_effective_restitution
            <= self.maximum_effective_restitution
        ):
            raise ValueError("bounce pad restitution must stay within measured limits")
        if self.bounce_pad_contact_margin_m <= 0.0:
            raise ValueError("bounce pad predictive margin must be positive")
        if (
            self.bounce_impact_speed_m_s <= 0.0
            or self.bounce_lane_speed_m_s <= 0.0
            or self.bounce_launch_vz_m_s <= 0.0
        ):
            raise ValueError("bounce construction speeds lost their measured signs")
        if self.bounce_launch_vz_m_s >= self.bounce_impact_speed_m_s:
            raise ValueError("the tossed bounce must still fall onto its pad")
        if not 0.0 < self.floor_bounce_pad_top_z_m < self.table_bounce_pad_top_z_m:
            raise ValueError("bounce pad heights must stay ordered above the floor")
        if not 0.0 < self.bounce_closure_start_before_event_s < self.closure_duration_s:
            raise ValueError(
                "bounce closure must start before its event and finish after it"
            )
        if (
            self.wall_rebound_launch_speed_m_s <= 0.0
            or self.wall_rebound_wall_time_s <= 0.0
            or self.wall_rebound_post_time_s <= 0.0
            or self.wall_rebound_hit_z_m <= 0.0
        ):
            raise ValueError("wall rebound construction lost its measured geometry")
        if self.wall_rebound_contact_margin_m <= 0.0:
            raise ValueError("wall rebound predictive margin must be positive")
        if not (
            self.minimum_rebound_effective_restitution
            <= self.wall_rebound_effective_restitution
            <= self.maximum_effective_restitution
        ):
            raise ValueError("wall rebound restitution must stay within measured limits")
        if not 0.0 < self.angled_barrier_yaw_rad < math.pi / 2:
            raise ValueError("angled barrier yaw must stay in (0, pi/2)")
        if not 0.0 < self.barrier_tangential_retention <= 1.0:
            raise ValueError("barrier tangential retention must be measured in (0, 1]")
        if not 0.0 < self.bounce_pad_lane_retention <= 1.0:
            raise ValueError("bounce pad lane retention must be measured in (0, 1]")
        if self.wall_effective_restitution > self.maximum_effective_restitution:
            raise ValueError("wall calibration injects contact energy")
        if self.rebound_acceptance() != DEFAULT_REBOUND_ACCEPTANCE:
            raise ValueError(
                "rebound acceptance changed without an evaluator/profile version"
            )
        numeric = (
            *self.wall_solref,
            *self.table_rebound_solref,
            *self.bounce_pad_solref,
            self.bounce_pad_contact_margin_m,
            self.bounce_pad_effective_restitution,
            self.bounce_impact_speed_m_s,
            self.bounce_launch_vz_m_s,
            self.bounce_lane_speed_m_s,
            self.floor_bounce_pad_top_z_m,
            self.table_bounce_pad_top_z_m,
            self.bounce_closure_start_before_event_s,
            self.wall_rebound_launch_speed_m_s,
            self.wall_rebound_wall_time_s,
            self.wall_rebound_post_time_s,
            self.wall_rebound_hit_z_m,
            self.wall_rebound_contact_margin_m,
            self.wall_rebound_effective_restitution,
            self.angled_barrier_yaw_rad,
            self.barrier_tangential_retention,
            self.bounce_pad_lane_retention,
            self.wall_effective_restitution,
            self.minimum_rebound_effective_restitution,
            self.minimum_rebound_outgoing_normal_speed_m_s,
            self.minimum_rebound_normal_separation_m,
            self.minimum_rebound_normal_separation_radius_fraction,
            self.minimum_rebound_separation_duration_s,
            *self.robotiq_pad_friction,
            *self.robotiq_pad_solref,
            self.robotiq_pad_half_depth_m,
            self.robotiq_passive_finger_acceleration_limit_rad_s2,
            self.robotiq_tendon_target,
            *self.robotiq_tendon_force_range_n,
            self.closure_start_before_ballistic_s,
            self.closure_duration_s,
            self.ready_hover_above_intercept_m,
            self.reach_arrival_before_ballistic_s,
            self.minimum_reach_duration_s,
            self.reach_limit_safety_fraction,
            *self.franka_joint_velocity_limit_rad_s,
            *self.franka_joint_acceleration_limit_rad_s2,
            self.minimum_arm_command_travel_rad,
            self.maximum_reach_arrival_distance_m,
            self.maximum_gripper_penetration_m,
            self.maximum_task_surface_penetration_m,
            self.maximum_effective_restitution,
            self.maximum_free_flight_energy_drift_fraction,
            *self.panda_ball_radius_range_m,
            self.robotiq_calibration_penetration_m,
            self.robotiq_first_bilateral_contact_600_s,
            self.robotiq_first_bilateral_contact_1200_s,
            self.robotiq_final_position_delta_m,
            self.rolling_pickup_speed_m_s,
            self.rolling_pickup_event_time_s,
            self.pickup_ready_retract_x_m,
            self.pickup_ready_raise_z_m,
            self.robotiq_pickup_ready_retract_x_m,
            self.robotiq_pickup_ready_raise_z_m,
            self.robotiq_pickup_reach_arrival_before_event_s,
            self.robotiq_pickup_closure_start_before_event_s,
            self.robotiq_pickup_closure_duration_s,
            self.robotiq_pickup_tendon_target,
            self.robotiq_pickup_capture_followthrough_x_m,
            self.robotiq_pickup_capture_start_s,
            self.robotiq_pickup_capture_end_s,
            self.final_retention_window_s,
            self.minimum_final_bilateral_fraction,
            self.pickup_lift_height_m,
            self.pickup_transport_start_s,
            self.pickup_transport_end_s,
            self.robotiq_pickup_transport_start_s,
            self.robotiq_pickup_transport_end_s,
            self.robotiq_pickup_standoff_m,
            self.pickup_transport_lateral_m,
            self.f2c_robotiq_tendon_target,
            self.f2c_robotiq_intercept_bias_x_m,
        )
        if any(not math.isfinite(float(value)) for value in numeric):
            raise ValueError("calibration profile contains a non-finite value")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    def rebound_acceptance(self) -> ReboundAcceptanceThresholds:
        result = ReboundAcceptanceThresholds(
            minimum_effective_restitution=(
                self.minimum_rebound_effective_restitution
            ),
            maximum_effective_restitution=self.maximum_effective_restitution,
            minimum_outgoing_normal_speed_m_s=(
                self.minimum_rebound_outgoing_normal_speed_m_s
            ),
            minimum_normal_separation_m=(
                self.minimum_rebound_normal_separation_m
            ),
            minimum_normal_separation_radius_fraction=(
                self.minimum_rebound_normal_separation_radius_fraction
            ),
            minimum_separation_duration_s=(
                self.minimum_rebound_separation_duration_s
            ),
        )
        result.validate()
        return result

    def grasp_retention(self) -> GraspRetentionThresholds:
        result = GraspRetentionThresholds(
            final_window_s=self.final_retention_window_s,
            minimum_bilateral_fraction=self.minimum_final_bilateral_fraction,
        )
        result.validate()
        return result


RIGID_REVIEW_PROFILE = RigidReviewProfile()
RIGID_REVIEW_PROFILE.validate()


def minimum_jerk_fraction(value: float) -> float:
    """Quintic minimum-jerk interpolation in ``[0, 1]``."""

    if not math.isfinite(value):
        raise ValueError("minimum-jerk phase must be finite")
    phase = min(1.0, max(0.0, float(value)))
    return phase**3 * (10.0 - 15.0 * phase + 6.0 * phase**2)


def exact_frame_schedule(
    duration_s: float,
    *,
    simulation_hz: int = 600,
    video_hz: int = 30,
) -> tuple[tuple[int, float], ...]:
    """Return exact simulation-step/frame timestamps without an endpoint frame.

    Canonical duration is represented by ``round(duration * fps)`` frames at
    timestamps ``k / fps``.  This deliberately avoids the historical extra
    frame at ``duration``.
    """

    if not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("duration_s must be finite and positive")
    if simulation_hz <= 0 or video_hz <= 0 or simulation_hz % video_hz:
        raise ValueError("simulation_hz must be a positive multiple of video_hz")
    frame_count = int(round(float(duration_s) * video_hz))
    if frame_count <= 0:
        raise ValueError("duration produces no canonical video frames")
    stride = simulation_hz // video_hz
    return tuple((index * stride, index / video_hz) for index in range(frame_count))


def timestep_comparison_failures(
    coarse: Mapping[str, Any],
    fine: Mapping[str, Any],
) -> tuple[str, ...]:
    """Apply the calibrated 600-vs-1200 admission thresholds."""

    failures: list[str] = []
    # Passive rows historically used the constant display label
    # ``passive_observation`` as ``outcome``.  That can hide a strict-QC or
    # saved-artifact replay failure at one of the two rates.  Require semantic
    # booleans and make missing evidence fail closed; retain the label check as
    # an additional actuated-outcome guard.
    if coarse.get("outcome") != fine.get("outcome"):
        failures.append("outcome_changed_at_1200_hz")
    for field, failure in (
        ("task_success", "task_success_changed_at_1200_hz"),
        ("physics_qc_pass", "physics_qc_changed_at_1200_hz"),
        (
            "saved_artifact_objective_replay_matches",
            "saved_artifact_replay_changed_at_1200_hz",
        ),
    ):
        if field not in coarse or field not in fine:
            failures.append(f"{field}_missing")
        elif not isinstance(coarse[field], bool) or not isinstance(fine[field], bool):
            failures.append(f"{field}_invalid")
        elif coarse[field] != fine[field]:
            failures.append(failure)
    if all(isinstance(row.get("physics_qc_pass"), bool) for row in (coarse, fine)):
        if not bool(coarse["physics_qc_pass"] and fine["physics_qc_pass"]):
            failures.append("physics_qc_failed_at_comparison_rate")
    if all(
        isinstance(row.get("saved_artifact_objective_replay_matches"), bool)
        for row in (coarse, fine)
    ):
        if not bool(
            coarse["saved_artifact_objective_replay_matches"]
            and fine["saved_artifact_objective_replay_matches"]
        ):
            failures.append("saved_artifact_replay_failed_at_comparison_rate")
    try:
        event_delta = abs(
            float(coarse["key_event_time_s"]) - float(fine["key_event_time_s"])
        )
    except (KeyError, TypeError, ValueError):
        failures.append("key_event_time_missing")
    else:
        if not math.isfinite(event_delta) or event_delta > 1.0 / 30.0:
            failures.append("key_event_shift_exceeds_one_frame")
    try:
        left = tuple(float(value) for value in coarse["key_event_position_m"])
        right = tuple(float(value) for value in fine["key_event_position_m"])
        position_delta = math.dist(left, right)
    except (KeyError, TypeError, ValueError):
        failures.append("key_event_position_missing")
    else:
        if len(left) != 3 or len(right) != 3 or position_delta > 0.01:
            failures.append("key_event_position_shift_exceeds_1cm")
    return tuple(failures)


__all__ = [
    "RIGID_REVIEW_PROFILE",
    "RigidReviewProfile",
    "SOURCE_MUJOCO_PROFILE_SCHEMA",
    "SOURCE_MUJOCO_PROFILE_VERSION",
    "exact_frame_schedule",
    "minimum_jerk_fraction",
    "timestep_comparison_failures",
]
