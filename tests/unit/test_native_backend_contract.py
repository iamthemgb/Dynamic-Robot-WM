from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest

from dynamic_robot_dataset.backends import IntendedBranch, ScenarioSpec
from dynamic_robot_dataset.backends.mujoco_native import (
    EVALUATOR_ID,
    EVALUATOR_VERSION,
    evaluate_saved_native_episode,
    make_scenario_spec,
    scenario_to_episode_plan,
)
from dynamic_robot_dataset.backends.mujoco_native.backend import (
    _verify_content_bound_artifact,
)
from dynamic_robot_dataset.backends.mujoco_native.model import _add_tool
from dynamic_robot_dataset.common.hashing import sha256_file
from dynamic_robot_dataset.common.episode_writer import (
    read_parquet_rows,
    write_parquet_atomic,
)
from dynamic_robot_dataset.common.contract_v2 import (
    DEFAULT_OBJECTIVE_EVALUATORS,
    ObjectiveRecomputeInput,
)


def _miss_rows() -> list[dict[str, object]]:
    return [
        {
            "timestamp": 0.0,
            "object.position": [2.0, 2.0, 2.0],
            "object.linear_velocity": [0.0, 0.0, -0.1],
            "robot.tool_position": [0.0, 0.0, 0.4],
            "robot.tool_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            "action.command.enabled": True,
            "action.command.joint_position": [0.0] * 7,
            "task_phase": "approach",
            "motion_mode": "free_flight",
        },
        {
            "timestamp": 0.1,
            "object.position": [2.0, 2.0, 1.98],
            "object.linear_velocity": [0.0, 0.0, -0.2],
            "robot.tool_position": [0.0, 0.0, 0.4],
            "robot.tool_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            "action.command.enabled": True,
            "action.command.joint_position": [0.01] * 7,
            "task_phase": "intercept",
            "motion_mode": "free_flight",
        },
    ]


def _projectile_action_rows(
    terminal_tool_position: list[float],
    *,
    moving_command: bool = True,
    terminal_object_velocity: list[float] | None = None,
) -> list[dict[str, object]]:
    """Minimal persisted evidence for branch-independent action labeling."""

    start = [0.205, -0.220, 0.425]
    terminal_command = [0.12] * 7 if moving_command else [0.0] * 7
    terminal_joint = [0.10] * 7 if moving_command else [0.0] * 7
    return [
        {
            "timestamp": 0.0,
            "object.position": [0.62, 0.0, 0.35],
            "object.linear_velocity": [0.8, 0.0, -0.2],
            "robot.tool_position": start,
            "robot.tool_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            "robot.joint_position": [0.0] * 7,
            "action.command.enabled": moving_command,
            "action.command.joint_position": [0.0] * 7,
            "task_phase": "approach",
            "motion_mode": "free_flight",
        },
        {
            "timestamp": 0.5,
            "object.position": [0.62, 0.0, 0.35],
            "object.linear_velocity": terminal_object_velocity
            or [0.8, 0.0, -0.6],
            "robot.tool_position": terminal_tool_position,
            "robot.tool_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            "robot.joint_position": terminal_joint,
            "action.command.enabled": moving_command,
            "action.command.joint_position": terminal_command,
            "task_phase": "intercept",
            "motion_mode": "free_flight",
        },
    ]


def test_scenario_round_trip_preserves_density_and_identity() -> None:
    spec = make_scenario_spec("centered_vertical_drop", seed=712)
    restored = ScenarioSpec.from_dict(spec.to_dict())
    assert restored.to_dict() == spec.to_dict()
    assert restored.object.density_kg_m3 is not None
    assert restored.spec_hash == spec.spec_hash
    assert scenario_to_episode_plan(restored).options["native_scenario_spec"] == spec.to_dict()


def test_native_action_counterfactual_episode_ids_are_unique() -> None:
    success = scenario_to_episode_plan(
        make_scenario_spec(
            "centered_vertical_drop", seed=713, branch=IntendedBranch.SUCCESS
        )
    )
    no_op = scenario_to_episode_plan(
        make_scenario_spec(
            "centered_vertical_drop", seed=713, branch=IntendedBranch.NO_OP
        )
    )

    assert success.counterfactual_bundle_id == no_op.counterfactual_bundle_id
    assert success.split_group_id == no_op.split_group_id
    assert success.physics_hash == no_op.physics_hash
    assert success.action_hash != no_op.action_hash
    assert success.episode_uuid != no_op.episode_uuid


def test_ramp_launch_rejects_initial_or_contact_chatter_free_flight() -> None:
    spec = make_scenario_spec("ramp_launch", seed=714)
    rows = _projectile_action_rows([0.205, -0.220, 0.425], moving_command=False)
    rows[0].update(
        {
            "object.position": [-0.78, 0.0, 0.11],
            "motion_mode": "free_flight",
            "object.active_surface": "none",
        }
    )
    rows[1].update(
        {
            # Still above the ramp, before the declared positive-X release
            # edge; unordered mode membership must not count as a launch.
            "object.position": [-0.40, 0.0, 0.16],
            "motion_mode": "free_flight",
            "object.active_surface": "none",
        }
    )
    evaluation = evaluate_saved_native_episode(
        spec,
        rows,
        [{"timestamp": 0.1, "object_b": "ramp_surface"}],
    )

    assert evaluation.outcome.task_success is False
    assert evaluation.outcome.failure_mode == "contact_without_completion"
    assert evaluation.actual_outcome_class == "contact_failure"
    assert evaluation.outcome.metrics["surface_to_free_flight"]["passed"] is False


def test_ramp_to_table_requires_contact_order() -> None:
    spec = make_scenario_spec("ramp_to_table", seed=719)
    rows = _projectile_action_rows(
        [0.205, -0.220, 0.425], moving_command=False
    )
    rows[0].update(
        {
            "object.position": [-0.78, 0.0, 0.12],
            "motion_mode": "rolling",
            "object.active_surface": "ramp_surface",
        }
    )
    rows[1].update(
        {
            "object.position": [0.10, 0.0, 0.03],
            "motion_mode": "rolling",
            "object.active_surface": "table_surface",
        }
    )
    reversed_order = evaluate_saved_native_episode(
        spec,
        rows,
        [
            {"timestamp": 0.1, "object_b": "table_surface"},
            {"timestamp": 0.2, "object_b": "ramp_surface"},
        ],
    )
    correct_order = evaluate_saved_native_episode(
        spec,
        rows,
        [
            {"timestamp": 0.1, "object_b": "ramp_surface"},
            {"timestamp": 0.2, "object_b": "table_surface"},
        ],
    )

    assert reversed_order.outcome.task_success is False
    assert reversed_order.outcome.metrics["expected_sequence_observed"] is False
    assert correct_order.outcome.task_success is True
    assert correct_order.outcome.metrics["expected_sequence_observed"] is True


def test_ramp_launch_does_not_credit_a_later_bounce_after_short_release() -> None:
    spec = make_scenario_spec("ramp_launch", seed=715)
    release_x = float(spec.extras["transition_release_x_m"])
    template = _projectile_action_rows(
        [0.205, -0.220, 0.425], moving_command=False
    )[0]

    def row(
        timestamp: float,
        position_x: float,
        mode: str,
        surface: str,
    ) -> dict[str, object]:
        value = dict(template)
        value.update(
            {
                "timestamp": timestamp,
                "object.position": [position_x, 0.0, 0.15],
                "motion_mode": mode,
                "object.active_surface": surface,
            }
        )
        return value

    rows = [
        row(0.00, release_x - 0.10, "rolling", "ramp_surface"),
        row(0.10, release_x + 0.01, "free_flight", "none"),
        row(0.12, release_x + 0.03, "free_flight", "none"),
        row(0.15, release_x + 0.04, "sliding", "table_surface"),
        row(0.30, release_x + 0.08, "free_flight", "none"),
        row(0.40, release_x + 0.12, "free_flight", "none"),
        row(0.50, release_x + 0.16, "free_flight", "none"),
    ]
    evaluation = evaluate_saved_native_episode(
        spec,
        rows,
        [
            {"timestamp": 0.0, "object_b": "ramp_surface"},
            {"timestamp": 0.15, "object_b": "table_surface"},
        ],
    )

    transition = evaluation.outcome.metrics["surface_to_free_flight"]
    assert transition["free_flight_start_time_s"] == pytest.approx(0.10)
    assert transition["free_flight_dwell_s"] == pytest.approx(0.02)
    assert transition["passed"] is False
    assert evaluation.actual_outcome_class == "contact_failure"


def test_ramp_launch_does_not_credit_first_sampled_flight_after_fixture_contact() -> None:
    spec = make_scenario_spec("ramp_launch", seed=716)
    release_x = float(spec.extras["transition_release_x_m"])
    template = _projectile_action_rows(
        [0.205, -0.220, 0.425], moving_command=False
    )[0]

    def row(
        timestamp: float,
        position_x: float,
        mode: str,
        surface: str,
    ) -> dict[str, object]:
        value = dict(template)
        value.update(
            {
                "timestamp": timestamp,
                "object.position": [position_x, 0.0, 0.15],
                "motion_mode": mode,
                "object.active_surface": surface,
            }
        )
        return value

    rows = [
        row(0.00, release_x - 0.10, "rolling", "ramp_surface"),
        # The direct departure was shorter than the persisted frame period;
        # the first saved post-edge row is already on another fixture.
        row(0.15, release_x + 0.04, "sliding", "table_surface"),
        row(0.30, release_x + 0.08, "free_flight", "none"),
        row(0.40, release_x + 0.12, "free_flight", "none"),
        row(0.50, release_x + 0.16, "free_flight", "none"),
    ]
    evaluation = evaluate_saved_native_episode(
        spec,
        rows,
        [
            {"timestamp": 0.0, "object_b": "ramp_surface"},
            {"timestamp": 0.15, "object_b": "table_surface"},
        ],
    )

    transition = evaluation.outcome.metrics["surface_to_free_flight"]
    assert transition["free_flight_start_time_s"] is None
    assert transition["intervening_contact_time_s"] == pytest.approx(0.15)
    assert transition["passed"] is False
    assert evaluation.actual_outcome_class == "contact_failure"

    # A rearmed contact with the origin surface is equally decisive: the
    # unsampled first departure has already ended, so a later bounce cannot
    # become the intended launch.
    same_surface_rows = [dict(value) for value in rows]
    same_surface_rows[1]["object.active_surface"] = "ramp_surface"
    same_surface = evaluate_saved_native_episode(
        spec,
        same_surface_rows,
        [
            {"timestamp": 0.0, "object_b": "ramp_surface"},
            {"timestamp": 0.15, "object_b": "ramp_surface"},
        ],
    )
    same_transition = same_surface.outcome.metrics["surface_to_free_flight"]
    assert same_transition["free_flight_start_time_s"] is None
    assert same_transition["intervening_contact_time_s"] == pytest.approx(0.15)
    assert same_transition["passed"] is False


def test_transition_rows_bind_first_subframe_boundary_departure() -> None:
    spec = make_scenario_spec("ramp_launch", seed=717)
    release_x = float(spec.extras["transition_release_x_m"])
    template = _projectile_action_rows(
        [0.205, -0.220, 0.425], moving_command=False
    )[0]

    def row(timestamp: float, position_x: float, mode: str) -> dict[str, object]:
        value = dict(template)
        value.update(
            {
                "timestamp": timestamp,
                "object.position": [position_x, 0.0, 0.15],
                "motion_mode": mode,
                "object.active_surface": (
                    "none" if mode == "free_flight" else "ramp_surface"
                ),
            }
        )
        return value

    rows = [
        row(0.0, release_x - 0.10, "rolling"),
        row(0.1, release_x + 0.01, "free_flight"),
        row(0.2, release_x + 0.10, "free_flight"),
    ]
    transitions = [
        {
            "timestamp": 0.04,
            "event_type": "motion_mode_transition",
            "from": "rolling",
            "to": "free_flight",
            "object_position_m": [release_x - 0.03, 0.0, 0.15],
        },
        {
            "timestamp": 0.07,
            "event_type": "surface_release_boundary_crossing",
            "from": "pre_release_region",
            "to": "post_release_region",
            "surface": "ramp_surface",
            "release_x_m": release_x,
            "direction": 1,
            "object_position_m": [release_x, 0.0, 0.15],
        },
    ]
    evaluation = evaluate_saved_native_episode(
        spec,
        rows,
        [{"timestamp": 0.0, "object_b": "ramp_surface"}],
        transitions,
    )
    transition = evaluation.outcome.metrics["surface_to_free_flight"]
    assert transition["evidence_source"] == "persisted_transition_rows"
    assert transition["free_flight_start_time_s"] == pytest.approx(0.07)
    assert transition["passed"] is True
    assert evaluation.actual_outcome_class == "success"

    fabricated = [dict(value) for value in transitions]
    fabricated[1]["release_x_m"] = release_x + 0.05
    rejected = evaluate_saved_native_episode(
        spec,
        rows,
        [{"timestamp": 0.0, "object_b": "ramp_surface"}],
        fabricated,
    )
    rejected_transition = rejected.outcome.metrics["surface_to_free_flight"]
    assert rejected_transition["invalid_boundary_row_count"] == 1
    assert rejected_transition["passed"] is False

    # Even without a rearmed contact event, a sub-frame mode change closes
    # the first boundary-crossing run. A later flight cannot replace it.
    failed = evaluate_saved_native_episode(
        spec,
        [
            rows[0],
            row(0.30, release_x + 0.08, "free_flight"),
            row(0.40, release_x + 0.12, "free_flight"),
        ],
        [{"timestamp": 0.0, "object_b": "ramp_surface"}],
        [
            *transitions,
            {
                "timestamp": 0.09,
                "event_type": "motion_mode_transition",
                "from": "free_flight",
                "to": "rolling",
            },
            {
                "timestamp": 0.25,
                "event_type": "surface_release_boundary_crossing",
                "from": "pre_release_region",
                "to": "post_release_region",
                "surface": "ramp_surface",
                "release_x_m": release_x,
                "direction": 1,
            },
        ],
    )
    failed_transition = failed.outcome.metrics["surface_to_free_flight"]
    assert failed_transition["free_flight_dwell_s"] == pytest.approx(0.02)
    assert failed_transition["passed"] is False
    assert failed.actual_outcome_class == "contact_failure"

    # If the first departure returns to support before crossing the boundary,
    # a later solver-rate crossing is still not allowed to replace it.
    preboundary_return = evaluate_saved_native_episode(
        spec,
        [
            rows[0],
            row(0.30, release_x + 0.08, "free_flight"),
            row(0.40, release_x + 0.12, "free_flight"),
        ],
        [{"timestamp": 0.0, "object_b": "ramp_surface"}],
        [
            transitions[0],
            {
                "timestamp": 0.08,
                "event_type": "motion_mode_transition",
                "from": "free_flight",
                "to": "rolling",
            },
            {
                "timestamp": 0.20,
                "event_type": "motion_mode_transition",
                "from": "rolling",
                "to": "free_flight",
            },
            {
                "timestamp": 0.25,
                "event_type": "surface_release_boundary_crossing",
                "from": "pre_release_region",
                "to": "post_release_region",
                "surface": "ramp_surface",
                "release_x_m": release_x,
                "direction": 1,
                "object_position_m": [release_x, 0.0, 0.15],
            },
        ],
    )
    preboundary_transition = preboundary_return.outcome.metrics[
        "surface_to_free_flight"
    ]
    assert preboundary_transition[
        "pre_boundary_free_flight_return_time_s"
    ] == pytest.approx(0.08)
    assert preboundary_transition["passed"] is False


def test_transition_evidence_survives_parquet_roundtrip(tmp_path) -> None:
    spec = make_scenario_spec("ramp_launch", seed=718)
    release_x = float(spec.extras["transition_release_x_m"])
    rows = _projectile_action_rows(
        [0.205, -0.220, 0.425], moving_command=False
    )
    rows[0].update(
        {
            "object.position": [release_x - 0.10, 0.0, 0.15],
            "motion_mode": "rolling",
            "object.active_surface": "ramp_surface",
        }
    )
    rows[1].update(
        {
            "object.position": [release_x + 0.10, 0.0, 0.18],
            "motion_mode": "free_flight",
            "object.active_surface": "none",
        }
    )
    transitions = [
        {
            "timestamp": 0.04,
            "event_type": "motion_mode_transition",
            "from": "rolling",
            "to": "free_flight",
            "active_surface": "none",
            "object_position_m": [release_x - 0.03, 0.0, 0.15],
        },
        {
            "timestamp": 0.07,
            "event_type": "surface_release_boundary_crossing",
            "from": "pre_release_region",
            "to": "post_release_region",
            "surface": "ramp_surface",
            "release_x_m": release_x,
            "direction": 1,
            "object_position_m": [release_x, 0.0, 0.16],
        },
    ]
    path = write_parquet_atomic(tmp_path / "transitions.parquet", transitions)
    persisted = read_parquet_rows(path)

    assert persisted[1]["surface"] == "ramp_surface"
    assert persisted[1]["direction"] == 1
    evaluation = evaluate_saved_native_episode(
        spec,
        rows,
        [{"timestamp": 0.0, "object_b": "ramp_surface"}],
        persisted,
    )
    assert evaluation.outcome.metrics["surface_to_free_flight"]["passed"] is True
    assert evaluation.actual_outcome_class == "success"


def test_paddle_geometry_maps_thickness_width_and_height_to_local_axes() -> None:
    spec = make_scenario_spec("paddle_block", seed=17)
    root = ET.fromstring("<mujoco><worldbody><body name='hand'/></worldbody></mujoco>")

    names = _add_tool(root, spec)

    assert names == ("native_tool_paddle",)
    geom = root.find(".//geom[@name='native_tool_paddle']")
    assert geom is not None
    size = tuple(float(value) for value in geom.attrib["size"].split())
    thickness, width, height = spec.tool.half_extents_m
    assert size == pytest.approx((width, thickness, height))
    assert size[1] < size[2]


def test_persisted_evaluator_cannot_read_intended_branch() -> None:
    spec = make_scenario_spec("centered_vertical_drop", seed=19)
    rows = _miss_rows()
    success_intent = evaluate_saved_native_episode(spec, rows, [])
    no_op_intent = evaluate_saved_native_episode(
        replace(spec, branch=IntendedBranch.NO_OP), rows, []
    )
    assert success_intent == no_op_intent
    assert success_intent.actual_outcome_class == "miss"


def test_wrong_action_is_measured_against_branch_independent_target() -> None:
    nominal_target = [0.62, 0.0, 0.35]
    spec = make_scenario_spec("bounce_to_robot_interception", seed=31)
    spec = replace(
        spec,
        extras={**spec.extras, "nominal_intercept_position_m": nominal_target},
    )
    # This terminal pose is 0.384 m from the nominal intercept target, above
    # the objective threshold, after substantial measured tool motion.
    rows = _projectile_action_rows([0.37, 0.25, 0.50])
    results = [
        evaluate_saved_native_episode(replace(spec, branch=branch), rows, [])
        for branch in (
            IntendedBranch.SUCCESS,
            IntendedBranch.NEAR_MISS,
            IntendedBranch.NO_OP,
            IntendedBranch.WRONG_ACTION,
        )
    ]

    assert {result.actual_outcome_class for result in results} == {"wrong_action"}
    assert {result.outcome.failure_mode for result in results} == {"wrong_action"}
    metrics = results[0].outcome.metrics
    assert metrics["misdirected_action_measured"] is True
    assert metrics["terminal_nominal_intercept_error_m"] > 0.30


def test_near_miss_and_no_op_are_not_mislabeled_as_wrong_actions() -> None:
    nominal_target = [0.62, 0.0, 0.35]
    spec = make_scenario_spec("bounce_to_robot_interception", seed=37)
    spec = replace(
        spec,
        extras={**spec.extras, "nominal_intercept_position_m": nominal_target},
    )
    near = evaluate_saved_native_episode(
        spec,
        _projectile_action_rows([0.62, 0.22, 0.35]),
        [],
    )
    no_op = evaluate_saved_native_episode(
        spec,
        _projectile_action_rows(
            [0.205, -0.220, 0.425], moving_command=False
        ),
        [],
    )

    assert near.actual_outcome_class == "near_miss"
    assert near.outcome.metrics["misdirected_action_measured"] is False
    assert no_op.actual_outcome_class == "no_op"
    assert no_op.outcome.metrics["misdirected_action_measured"] is False


def test_projectile_contact_score_uses_measured_post_contact_deflection() -> None:
    spec = make_scenario_spec("direct_projectile", seed=43)
    event = {"timestamp": 0.1, "object_b": "native_tool"}
    wrong_direction = evaluate_saved_native_episode(
        spec,
        _projectile_action_rows(
            [0.62, 0.0, 0.35],
            terminal_object_velocity=[0.20, 0.0, -0.1],
        ),
        [event],
    )
    weak_reversal = evaluate_saved_native_episode(
        spec,
        _projectile_action_rows(
            [0.62, 0.0, 0.35],
            terminal_object_velocity=[-0.06, 0.0, -0.1],
        ),
        [event],
    )
    decisive_reversal = evaluate_saved_native_episode(
        spec,
        _projectile_action_rows(
            [0.62, 0.0, 0.35],
            terminal_object_velocity=[-0.20, 0.0, -0.1],
        ),
        [event],
    )

    assert wrong_direction.actual_outcome_class == "contact_failure"
    assert wrong_direction.outcome.partial_success_score == 0.0
    assert weak_reversal.actual_outcome_class == "partial_success"
    assert weak_reversal.outcome.partial_success_score == pytest.approx(0.6)
    assert decisive_reversal.actual_outcome_class == "success"
    assert decisive_reversal.outcome.partial_success_score == 1.0


def test_native_evaluator_is_available_through_common_registry() -> None:
    spec = make_scenario_spec("centered_vertical_drop", seed=29)
    evaluator = DEFAULT_OBJECTIVE_EVALUATORS.get(EVALUATOR_ID, EVALUATOR_VERSION)
    assert evaluator is not None
    record = SimpleNamespace(extras={"native_scenario_spec": spec.to_dict()})
    result = evaluator(ObjectiveRecomputeInput(record, _miss_rows(), [], []))
    assert result.task_success is False
    assert result.actual_outcome_class.value == "miss"
    assert result.primary_failure_code == "receptacle_missed_object"
    assert result.evidence["stored_rows_recomputed"] is True


def test_backend_import_does_not_eagerly_require_mujoco() -> None:
    # The stable typed contract and planner are usable by inventory/dry-run
    # commands even if the optional native simulator is unavailable.
    from dynamic_robot_dataset.backends import get_backend

    assert callable(get_backend)


def test_admission_artifacts_are_resolved_and_byte_verified(tmp_path) -> None:
    artifact = tmp_path / "calibration.json"
    artifact.write_text('{"approved": true}\n', encoding="utf-8")
    digest = sha256_file(artifact)

    assert _verify_content_bound_artifact(artifact, digest) == (
        True,
        str(artifact.resolve()),
        digest,
    )
    assert _verify_content_bound_artifact(artifact, "0" * 64)[0] is False
    artifact.write_text('{"approved": false}\n', encoding="utf-8")
    assert _verify_content_bound_artifact(artifact, digest)[0] is False
