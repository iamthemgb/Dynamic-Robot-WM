from __future__ import annotations

from copy import deepcopy

import pytest

from dynamic_robot_dataset.backends.source_mujoco import SourceMujocoBackend
from dynamic_robot_dataset.backends.source_mujoco.backend import (
    _contact_body_proxy_visibility,
    _measured_visibility_key_event,
    _tool_render_geom_ids_by_side,
)
from dynamic_robot_dataset.common.hashing import sha256_json
from dynamic_robot_dataset.common.review import event_strip_frame_indices
from dynamic_robot_dataset.common.qc import (
    EpisodeQC,
    QCValidator,
    _validate_source_mujoco_visibility_qc,
)
from dynamic_robot_dataset.common.review_suite import build_review_suite_plan
from dynamic_robot_dataset.common.run_orchestration import RunPlanEpisode
from dynamic_robot_dataset.common.source_execution import (
    materialize_source_mujoco_result,
    prepare_source_review_declaration,
)
from dynamic_robot_dataset.common.visual_qc import (
    NATIVE_VISUAL_QC_SCHEMA,
    NATIVE_VISUAL_THRESHOLDS,
    SOURCE_MUJOCO_VISIBILITY_QC_SCHEMA,
    SOURCE_MUJOCO_VISUAL_THRESHOLDS,
    source_mujoco_visibility_media_binding,
)
from dynamic_robot_dataset.common.schema import EpisodeRecord
from dynamic_robot_dataset.common.synchronization import (
    fixed_duration_frame_timestamps,
)


def _case(case_id: str):
    return next(
        case
        for case in build_review_suite_plan().cases
        if case.case_id == case_id
    )


def _entry(declaration: dict) -> RunPlanEpisode:
    return RunPlanEpisode(
        episode_uuid=declaration["episode_uuid"],
        episode_index=declaration["episode_index"],
        shard_id=0,
        declaration=declaration,
        source_scenario_spec_sha256=declaration[
            "source_scenario_spec_sha256"
        ],
    )


def _tool_visibility_topology(*, actuated: bool) -> dict:
    return {
        "schema_version": "source-mujoco-tool-visibility-topology/v1",
        "tool_geom_body_ids": {"1": 11, "2": 12} if actuated else {},
        "left_tool_geom_ids": [1] if actuated else [],
        "right_tool_geom_ids": [2] if actuated else [],
        "left_tool_body_id": 11 if actuated else None,
        "right_tool_body_id": 12 if actuated else None,
    }


def _source_scenario(*, actuated: bool, topology: dict | None = None) -> dict:
    topology = topology or _tool_visibility_topology(actuated=actuated)
    return {
        "backend": "source_mujoco",
        "physics": {
            "tool_visibility_topology": topology,
            "tool_visibility_topology_sha256": sha256_json(topology),
        },
    }


def _passing_visibility(*, actuated: bool) -> dict:
    planned_key_event_time_s = 4 / 30.0
    actual_key_event_time_s = 5 / 30.0 if actuated else planned_key_event_time_s
    actual_key_event_frame_index = 5 if actuated else 4
    checkpoint_indices = {
        "initial": 0,
        "apex": 1,
        "key_event": actual_key_event_frame_index,
        "final": 9,
    }
    checkpoints = {
        name: {
            "frame_index": index,
            "timestamp_s": index / 30.0,
            "visible_in_any_view": True,
        }
        for name, index in checkpoint_indices.items()
    }
    frame_metrics = []
    for index in range(10):
        frame_metrics.append(
            {
                "frame_index": index,
                "timestamp_s": index / 30.0,
                "object_pixel_count": 64,
                "segmentation_bbox_margin_px": 8.0,
                "projected_sphere_margin_px": 8.0,
                "projected_center_visible": True,
                "target_present": True,
                "tool_pixel_count": 32 if actuated else 0,
                "left_tool_pixel_count": 16 if actuated else 0,
                "right_tool_pixel_count": 16 if actuated else 0,
                "fixture_pixel_count": 0 if actuated else 16,
                "geom_pixel_counts": (
                    {"1": 16, "2": 16} if actuated else {"3": 16}
                ),
                "underexposed_fraction": 0.0,
                "overexposed_fraction": 0.0,
            }
        )
    per_view_checkpoints = {
        name: {**frame_metrics[index], "visible": True}
        for name, index in checkpoint_indices.items()
    }
    view = {
        "target_visible_frame_fraction": 1.0,
        "minimum_object_area_px": 64,
        "minimum_segmentation_bbox_margin_px": 8.0,
        "minimum_projected_sphere_margin_px": 8.0,
        "key_event_object_pixel_count": 64,
        "key_event_bbox_margin_px": 8.0,
        "key_event_tool_pixel_count": 32 if actuated else 0,
        "key_event_left_tool_pixel_count": 16 if actuated else 0,
        "key_event_right_tool_pixel_count": 16 if actuated else 0,
        "key_event_fixture_pixel_count": 0 if actuated else 16,
        "frames": frame_metrics,
        "checkpoints": per_view_checkpoints,
    }
    return {
        "schema_version": SOURCE_MUJOCO_VISIBILITY_QC_SCHEMA,
        "evaluated": True,
        "rendered_streams_complete": True,
        "rendered_frame_count": 10,
        "expected_frame_count": 10,
        "planned_key_event_time_s": planned_key_event_time_s,
        "planned_key_event_frame_index": 4,
        "planned_key_event_frame_timestamp_s": 4 / 30.0,
        "actual_key_event_time_s": actual_key_event_time_s,
        "actual_key_event_frame_index": actual_key_event_frame_index,
        "actual_key_event_frame_timestamp_s": actual_key_event_frame_index / 30.0,
        "actual_key_event_name": (
            "bilateral_grasp_onset" if actuated else "task_interaction"
        ),
        "actual_key_event_source": (
            "persisted_bilateral_contact"
            if actuated
            else "planned_source_scenario_event"
        ),
        "planned_key_event_name": "task_interaction",
        "required_views": ["main", "secondary"],
        "required_checkpoint_names": ["initial", "apex", "key_event", "final"],
        "target_visible_frame_fraction": 1.0,
        "initial_state_visible_in_any_view": True,
        "apex_visible_in_any_view": True,
        "key_event_visible_in_any_view": True,
        "final_state_visible_in_any_view": True,
        "critically_cropped": False,
        "physical_contact_applicable": actuated,
        "contact_occluded_both_views": False if actuated else None,
        "actual_contact_counterpart_visible_at_key_event": (
            True if actuated else None
        ),
        "contact_body_proxy_resolution_complete": None,
        "contact_counterpart_body_ids": [],
        "contact_proxy_geom_ids": [],
        "contact_visible_proxy_geom_ids_by_view": {},
        "contact_proxy_pixel_counts_by_view": {},
        "unresolved_contact_counterpart_geom_ids": [],
        "contact_exact_fixture_geom_ids": [],
        "contact_visible_fixture_geom_ids_by_view": {},
        "contact_fixture_pixel_counts_by_view": {},
        "bilateral_tool_sides_visible_at_key_event": True if actuated else None,
        "planned_checkpoint_covisible_in_any_view": True,
        "contact_counterpart_geom_ids": [1, 2] if actuated else [],
        "left_tool_geom_ids": [1] if actuated else [],
        "right_tool_geom_ids": [2] if actuated else [],
        "tool_geom_body_ids": {"1": 11, "2": 12} if actuated else {},
        "left_tool_body_id": 11 if actuated else None,
        "right_tool_body_id": 12 if actuated else None,
        "tool_visibility_topology_sha256": sha256_json(
            _tool_visibility_topology(actuated=actuated)
        ),
        "minimum_bbox_margin_px": 8.0,
        "key_event_object_area_px": 64,
        "tool_visibility_applicable": actuated,
        "tool_visible_at_key_event": True if actuated else None,
        "fixture_visibility_applicable": not actuated,
        "fixture_visible_at_key_event": None if actuated else True,
        "counterpart_visible_at_key_event": True,
        "camera_roles_correct": True,
        "maximum_underexposed_fraction": 0.0,
        "maximum_overexposed_fraction": 0.0,
        "thresholds": dict(SOURCE_MUJOCO_VISUAL_THRESHOLDS),
        "checkpoints": checkpoints,
        "views": {"main": deepcopy(view), "secondary": deepcopy(view)},
    }


def _passing_frame_rows() -> list[dict]:
    return [
        {
            "timestamp": index / 30.0,
            "object.position": [0.0, 0.0, 1.0 if index == 1 else 0.5],
        }
        for index in range(10)
    ]


def _replay_visibility(
    visibility: dict,
    *,
    actuated: bool,
    event_rows: list[dict] | tuple[dict, ...] = (),
    source_scenario: dict | None = None,
) -> EpisodeQC:
    result = EpisodeQC("episode", 0, False)
    _validate_source_mujoco_visibility_qc(
        result,
        visibility,
        end_effector="franka_hand" if actuated else "no_robot",
        expected_frame_count=10,
        frame_rows=_passing_frame_rows(),
        event_rows=event_rows,
        record_key_event_name=visibility["actual_key_event_name"],
        record_key_event_time_s=visibility["actual_key_event_time_s"],
        objective_key_event_source=visibility["actual_key_event_source"],
        source_scenario=(
            source_scenario or _source_scenario(actuated=actuated)
        ),
    )
    return result


def _qc_record(
    visibility: dict | None,
    *,
    backend: str = "source_mujoco",
    visibility_hash: str | None = None,
    actuated: bool = False,
) -> EpisodeRecord:
    provenance = {"backend": backend}
    extras = {
        "end_effector": "franka_hand" if actuated else "no_robot",
        "backend_provenance": provenance,
        "source_scenario_spec": _source_scenario(actuated=actuated),
    }
    if visibility is not None:
        digest = visibility_hash or sha256_json(visibility)
        extras.update(visibility_qc=visibility, visibility_qc_sha256=digest)
        provenance["visibility_qc_sha256"] = digest
    return EpisodeRecord(
        episode_uuid="10000000-0000-4000-8000-000000000001",
        episode_index=0,
        counterfactual_bundle_id="visibility-test",
        scene_seed=1,
        branch_seed=2,
        family="passive_physics",
        subfamily="wall_rebound",
        intended_branch="passive_observation",
        actual_outcome="passive_observation",
        task_success=True,
        failure_mode="none",
        source_generator="visibility-test",
        source_generator_version="1",
        config_hash="a" * 64,
        simulator_name="mujoco",
        simulator_version="test",
        renderer="mujoco.Renderer",
        frame_count=10,
        duration_s=1.0,
        event_time_s=0.5,
        key_event_name="wall_contact",
        objective_evaluator_id="unregistered_visibility_test",
        objective_evaluator_version="1",
        objective_evidence={
            "independently_recomputed": True,
            "evidence_hash": "b" * 64,
            "source": "test",
        },
        objective_metrics={"objective_success": True},
        quality_flags=["review_only"],
        extras=extras,
    )


@pytest.mark.parametrize(
    ("end_effector", "actuated"),
    (("no_robot", False), ("franka_hand", True)),
)
def test_strict_source_visibility_is_counterpart_applicability_aware(
    end_effector: str,
    actuated: bool,
) -> None:
    result = EpisodeQC("episode", 0, False)
    _validate_source_mujoco_visibility_qc(
        result,
        _passing_visibility(actuated=actuated),
        end_effector=end_effector,
        expected_frame_count=10,
        frame_rows=_passing_frame_rows(),
        event_rows=(
            [
                {
                    "timestamp": 5 / 30.0,
                    "contact_category": "gripper",
                    "counterpart_geom_id": geom_id,
                }
                for geom_id in (1, 2)
            ]
            if actuated
            else []
        ),
        record_key_event_name=(
            "bilateral_grasp_onset" if actuated else "task_interaction"
        ),
        record_key_event_time_s=(5 / 30.0 if actuated else 4 / 30.0),
        objective_key_event_source=(
            "persisted_bilateral_contact"
            if actuated
            else "planned_source_scenario_event"
        ),
        source_scenario=_source_scenario(actuated=actuated),
    )
    assert result.passed


def test_strict_source_visibility_rejects_missing_final_and_apex() -> None:
    visibility = _passing_visibility(actuated=False)
    visibility["final_state_visible_in_any_view"] = False
    visibility["apex_visible_in_any_view"] = False
    visibility["checkpoints"]["final"]["visible_in_any_view"] = False
    visibility["checkpoints"]["apex"]["visible_in_any_view"] = False
    result = EpisodeQC("episode", 0, False)

    _validate_source_mujoco_visibility_qc(
        result,
        visibility,
        end_effector="no_robot",
        expected_frame_count=10,
        frame_rows=_passing_frame_rows(),
        source_scenario=_source_scenario(actuated=False),
    )

    assert any("final target state" in failure for failure in result.hard_failures)
    assert any("target apex" in failure for failure in result.hard_failures)
    assert any("final checkpoint" in failure for failure in result.hard_failures)
    assert any("apex checkpoint" in failure for failure in result.hard_failures)


def test_whole_trajectory_presence_does_not_reuse_key_event_64px_threshold() -> None:
    assert SOURCE_MUJOCO_VISUAL_THRESHOLDS["minimum_trajectory_object_area_px"] == 4
    assert SOURCE_MUJOCO_VISUAL_THRESHOLDS["minimum_key_event_object_area_px"] == 64


def test_visibility_media_binding_changes_with_camera_or_encoded_video() -> None:
    streams = (
        "observation.images.main",
        "observation.images.secondary",
    )
    camera_rows = [
        {"camera_id": f"camera-{index}", "camera_name": stream, "fovy": 45.0}
        for index, stream in enumerate(streams)
    ]
    calibration_ids = {
        stream: f"camera-{index}" for index, stream in enumerate(streams)
    }
    video_paths = {stream: f"videos/{index}.mp4" for index, stream in enumerate(streams)}
    content_hashes = {
        relative: str(index + 1) * 64
        for index, relative in enumerate(video_paths.values())
    }
    baseline = source_mujoco_visibility_media_binding(
        visibility_qc_sha256="a" * 64,
        camera_rows=camera_rows,
        camera_stream_calibration_ids=calibration_ids,
        video_paths=video_paths,
        content_hashes=content_hashes,
    )

    changed_video = dict(content_hashes)
    changed_video[video_paths[streams[0]]] = "f" * 64
    rebound = source_mujoco_visibility_media_binding(
        visibility_qc_sha256="a" * 64,
        camera_rows=camera_rows,
        camera_stream_calibration_ids=calibration_ids,
        video_paths=video_paths,
        content_hashes=changed_video,
    )

    assert sha256_json(baseline) != sha256_json(rebound)


def test_no_contact_planned_checkpoint_accepts_eight_pixel_tool_evidence() -> None:
    visibility = _passing_visibility(actuated=True)
    visibility.update(
        actual_key_event_time_s=4 / 30.0,
        actual_key_event_frame_index=4,
        actual_key_event_frame_timestamp_s=4 / 30.0,
        actual_key_event_name="task_interaction",
        actual_key_event_source="planned_interception_for_measured_miss",
        physical_contact_applicable=False,
        contact_occluded_both_views=None,
        actual_contact_counterpart_visible_at_key_event=None,
        bilateral_tool_sides_visible_at_key_event=None,
        contact_counterpart_geom_ids=[],
    )
    visibility["checkpoints"]["key_event"].update(
        frame_index=4, timestamp_s=4 / 30.0
    )
    for view in visibility["views"].values():
        for frame in view["frames"]:
                frame.update(
                    tool_pixel_count=8,
                    left_tool_pixel_count=4,
                    right_tool_pixel_count=4,
                    geom_pixel_counts={"1": 4, "2": 4},
                )
        view.update(
            key_event_tool_pixel_count=8,
            key_event_left_tool_pixel_count=4,
            key_event_right_tool_pixel_count=4,
        )
        for checkpoint_name, checkpoint in view["checkpoints"].items():
            checkpoint_index = (
                4 if checkpoint_name == "key_event" else checkpoint["frame_index"]
            )
            view["checkpoints"][checkpoint_name] = {
                **view["frames"][checkpoint_index],
                "visible": True,
            }
    result = EpisodeQC("episode", 0, False)

    _validate_source_mujoco_visibility_qc(
        result,
        visibility,
        end_effector="franka_hand",
        expected_frame_count=10,
        frame_rows=_passing_frame_rows(),
        record_key_event_name="task_interaction",
        record_key_event_time_s=4 / 30.0,
        objective_key_event_source="planned_interception_for_measured_miss",
        source_scenario=_source_scenario(actuated=True),
    )

    assert result.passed, result.hard_failures


@pytest.mark.parametrize(
    ("aggregate_field", "frame_field", "bad_fraction"),
    (
        ("maximum_underexposed_fraction", "underexposed_fraction", 0.9),
        ("maximum_overexposed_fraction", "overexposed_fraction", 0.8),
    ),
)
def test_visibility_replay_rejects_passing_exposure_summary_over_bad_frame(
    aggregate_field: str,
    frame_field: str,
    bad_fraction: float,
) -> None:
    visibility = _passing_visibility(actuated=False)
    # Frame 2 is not a checkpoint, so this isolates the aggregate replay from
    # the existing checkpoint-copy binding.
    visibility["views"]["main"]["frames"][2][frame_field] = bad_fraction

    result = _replay_visibility(visibility, actuated=False)

    exposure_label = aggregate_field.removeprefix("maximum_").replace("_", " ")
    assert any(exposure_label in value for value in result.hard_failures)
    assert any("recomputed" in value and "fraction exceeds" in value for value in result.hard_failures)


@pytest.mark.parametrize(
    ("actuated", "summary_field"),
    (
        (False, "key_event_fixture_pixel_count"),
        (True, "key_event_tool_pixel_count"),
        (True, "key_event_left_tool_pixel_count"),
        (True, "key_event_right_tool_pixel_count"),
    ),
)
def test_visibility_replay_rejects_key_event_pixel_summary_not_from_frame(
    actuated: bool,
    summary_field: str,
) -> None:
    visibility = _passing_visibility(actuated=actuated)
    visibility["views"]["main"][summary_field] += 1
    event_rows = (
        [
            {
                "timestamp": 5 / 30.0,
                "contact_category": "gripper",
                "counterpart_geom_id": geom_id,
            }
            for geom_id in (1, 2)
        ]
        if actuated
        else []
    )

    result = _replay_visibility(
        visibility,
        actuated=actuated,
        event_rows=event_rows,
    )

    assert any(summary_field in value for value in result.hard_failures)


def test_visibility_replay_derives_passive_physical_contact_from_events() -> None:
    visibility = _passing_visibility(actuated=False)
    # Keep every summary field and its hashable payload internally consistent
    # with the old no-contact claim. Saved task-surface evidence must override
    # that claim during strict replay.
    event_rows = [
        {
            "timestamp": visibility["actual_key_event_time_s"],
            "contact_category": "task_surface",
            "counterpart_geom_id": 3,
        }
    ]

    result = _replay_visibility(
        visibility,
        actuated=False,
        event_rows=event_rows,
    )

    assert any(
        "physical-contact applicability differs" in value
        for value in result.hard_failures
    )
    assert any(
        "actual-contact visibility differs" in value
        for value in result.hard_failures
    )


def test_visibility_replay_rejects_bilateral_source_with_false_applicability() -> None:
    visibility = _passing_visibility(actuated=True)
    visibility.update(
        physical_contact_applicable=False,
        actual_contact_counterpart_visible_at_key_event=None,
        contact_occluded_both_views=None,
    )
    event_rows = [
        {
            "timestamp": visibility["actual_key_event_time_s"],
            "contact_category": "gripper",
            "counterpart_geom_id": geom_id,
        }
        for geom_id in (1, 2)
    ]

    result = _replay_visibility(
        visibility,
        actuated=True,
        event_rows=event_rows,
    )

    assert any(
        "physical-contact applicability differs" in value
        for value in result.hard_failures
    )


def test_visibility_replay_rejects_measured_miss_with_saved_tool_contact() -> None:
    visibility = _passing_visibility(actuated=True)
    visibility.update(
        actual_key_event_time_s=4 / 30.0,
        actual_key_event_frame_index=4,
        actual_key_event_frame_timestamp_s=4 / 30.0,
        actual_key_event_name="task_interaction",
        actual_key_event_source="planned_interception_for_measured_miss",
        physical_contact_applicable=False,
        contact_occluded_both_views=None,
        actual_contact_counterpart_visible_at_key_event=None,
        bilateral_tool_sides_visible_at_key_event=None,
        contact_counterpart_geom_ids=[],
    )
    visibility["checkpoints"]["key_event"].update(
        frame_index=4,
        timestamp_s=4 / 30.0,
    )
    for view in visibility["views"].values():
        view["checkpoints"]["key_event"] = {
            **view["frames"][4],
            "visible": True,
        }

    result = _replay_visibility(
        visibility,
        actuated=True,
        event_rows=[
            {
                "timestamp": 0.05,
                "contact_category": "robot_arm",
                "counterpart_geom_id": 9,
            }
        ],
    )

    assert any(
        "measured-miss event source conflicts" in value
        for value in result.hard_failures
    )


def test_visibility_replay_rejects_claimed_bilateral_visibility_over_zero_side_masks() -> None:
    visibility = _passing_visibility(actuated=True)
    actual_index = visibility["actual_key_event_frame_index"]
    for view in visibility["views"].values():
        metric = view["frames"][actual_index]
        metric["left_tool_pixel_count"] = 0
        metric["right_tool_pixel_count"] = 0
        view["key_event_left_tool_pixel_count"] = 0
        view["key_event_right_tool_pixel_count"] = 0
        view["checkpoints"]["key_event"] = {**metric, "visible": True}
    event_rows = [
        {
            "timestamp": visibility["actual_key_event_time_s"],
            "contact_category": "gripper",
            "counterpart_geom_id": geom_id,
        }
        for geom_id in (1, 2)
    ]

    result = _replay_visibility(
        visibility,
        actuated=True,
        event_rows=event_rows,
    )

    assert any(
        "bilateral visibility claim differs" in value
        for value in result.hard_failures
    )
    assert any(
        "actual-contact visibility differs" in value
        for value in result.hard_failures
    )


def test_visibility_replay_rejects_claimed_generic_contact_visibility_over_zero_geom_pixels() -> None:
    visibility = _passing_visibility(actuated=True)
    topology = _tool_visibility_topology(actuated=True)
    topology["tool_geom_body_ids"]["10"] = 11
    visibility.update(
        actual_key_event_name="tool_contact_onset",
        actual_key_event_source="persisted_contact_event",
        contact_counterpart_geom_ids=[10],
        tool_geom_body_ids={"1": 11, "2": 12, "10": 11},
        contact_body_proxy_resolution_complete=True,
        contact_counterpart_body_ids=[11],
        contact_proxy_geom_ids=[1, 10],
        contact_visible_proxy_geom_ids_by_view={
            "main": [1],
            "secondary": [1],
        },
        contact_proxy_pixel_counts_by_view={"main": 16, "secondary": 16},
        bilateral_tool_sides_visible_at_key_event=None,
        tool_visibility_topology_sha256=sha256_json(topology),
    )
    actual_index = visibility["actual_key_event_frame_index"]
    for view in visibility["views"].values():
        metric = view["frames"][actual_index]
        metric["geom_pixel_counts"]["1"] = 0
        metric["geom_pixel_counts"]["10"] = 0
        view["checkpoints"]["key_event"] = {**metric, "visible": True}

    result = _replay_visibility(
        visibility,
        actuated=True,
        event_rows=[
            {
                "timestamp": visibility["actual_key_event_time_s"],
                "contact_category": "gripper",
                "counterpart_geom_id": 10,
            }
        ],
        source_scenario=_source_scenario(actuated=True, topology=topology),
    )

    assert any(
        "actual-contact visibility differs" in value
        for value in result.hard_failures
    )
    assert any(
        "contact-occlusion claim differs" in value
        for value in result.hard_failures
    )
    assert any(
        "contact_proxy_pixel_counts_by_view" in value
        for value in result.hard_failures
    )


def test_visibility_replay_accepts_visible_same_body_proxy_for_hidden_contact_geom() -> None:
    visibility = _passing_visibility(actuated=True)
    topology = _tool_visibility_topology(actuated=True)
    topology["tool_geom_body_ids"]["10"] = 11
    visibility.update(
        actual_key_event_name="tool_contact_onset",
        actual_key_event_source="persisted_contact_event",
        contact_counterpart_geom_ids=[10],
        tool_geom_body_ids={"1": 11, "2": 12, "10": 11},
        contact_body_proxy_resolution_complete=True,
        contact_counterpart_body_ids=[11],
        contact_proxy_geom_ids=[1, 10],
        contact_visible_proxy_geom_ids_by_view={
            "main": [1],
            "secondary": [1],
        },
        contact_proxy_pixel_counts_by_view={"main": 16, "secondary": 16},
        unresolved_contact_counterpart_geom_ids=[],
        bilateral_tool_sides_visible_at_key_event=None,
        tool_visibility_topology_sha256=sha256_json(topology),
    )
    actual_index = visibility["actual_key_event_frame_index"]
    for view in visibility["views"].values():
        metric = view["frames"][actual_index]
        metric["geom_pixel_counts"]["10"] = 0
        view["checkpoints"]["key_event"] = {**metric, "visible": True}

    result = _replay_visibility(
        visibility,
        actuated=True,
        event_rows=[
            {
                "timestamp": visibility["actual_key_event_time_s"],
                "contact_category": "gripper",
                "counterpart_geom_id": 10,
            }
        ],
        source_scenario=_source_scenario(actuated=True, topology=topology),
    )

    assert result.passed, result.hard_failures


def test_persisted_visibility_rejects_self_rehashed_tool_topology_tamper(
    tmp_path,
) -> None:
    visibility = _passing_visibility(actuated=True)
    forged_topology = _tool_visibility_topology(actuated=True)
    forged_topology.update(
        tool_geom_body_ids={"1": 91, "2": 92},
        left_tool_body_id=91,
        right_tool_body_id=92,
    )
    visibility.update(
        tool_geom_body_ids=forged_topology["tool_geom_body_ids"],
        left_tool_body_id=forged_topology["left_tool_body_id"],
        right_tool_body_id=forged_topology["right_tool_body_id"],
        tool_visibility_topology_sha256=sha256_json(forged_topology),
    )
    record = _qc_record(visibility, actuated=True)
    result, _ = QCValidator(tmp_path, deep_video_checks=False)._validate_episode(
        record
    )

    assert record.extras["visibility_qc_sha256"] == sha256_json(visibility)
    assert record.extras["backend_provenance"]["visibility_qc_sha256"] == sha256_json(
        visibility
    )
    assert not any(
        "visibility QC hash binding changed" in failure
        for failure in result.hard_failures
    )
    assert any(
        "tool topology hash differs from SourceScenarioSpec" in failure
        or "differs from compiled SourceScenarioSpec topology" in failure
        for failure in result.hard_failures
    )


def test_visibility_replay_rejects_passing_counterpart_claim_over_zero_pixels() -> None:
    visibility = _passing_visibility(actuated=False)
    actual_index = visibility["actual_key_event_frame_index"]
    for view in visibility["views"].values():
        metric = view["frames"][actual_index]
        metric["fixture_pixel_count"] = 0
        view["key_event_fixture_pixel_count"] = 0
        view["checkpoints"]["key_event"] = {**metric, "visible": True}

    result = _replay_visibility(visibility, actuated=False)

    assert any(
        "fixture visibility differs" in value for value in result.hard_failures
    )
    assert any(
        "counterpart visibility differs" in value
        for value in result.hard_failures
    )
    assert any(
        "planned checkpoint co-visibility differs" in value
        for value in result.hard_failures
    )


@pytest.mark.parametrize("malformed", (None, "not-a-number", float("nan")))
@pytest.mark.parametrize(
    "field",
    (
        "rendered_frame_count",
        "target_visible_frame_fraction",
        "minimum_bbox_margin_px",
        "key_event_object_area_px",
    ),
)
def test_visibility_numeric_malformations_are_hard_failures_not_exceptions(
    field: str, malformed: object
) -> None:
    visibility = _passing_visibility(actuated=False)
    visibility[field] = malformed
    result = EpisodeQC("episode", 0, False)

    _validate_source_mujoco_visibility_qc(
        result,
        visibility,
        end_effector="no_robot",
        expected_frame_count=10,
        frame_rows=_passing_frame_rows(),
        source_scenario=_source_scenario(actuated=False),
    )

    assert not result.passed
    assert any(field in failure for failure in result.hard_failures)


def test_success_visibility_uses_measured_bilateral_onset_not_planned_time() -> None:
    scenario = SourceMujocoBackend().compile_case(_case("F1a-review-00"))
    measured_time_s = scenario.key_event_time_s + 0.05
    event = _measured_visibility_key_event(
        scenario,
        (
            {"timestamp": scenario.key_event_time_s, "contact.bilateral": False},
            {"timestamp": measured_time_s, "contact.bilateral": True},
        ),
        (),
    )
    assert event["planned_key_event_time_s"] == scenario.key_event_time_s
    assert event["actual_key_event_time_s"] == measured_time_s
    assert event["actual_key_event_time_s"] != event["planned_key_event_time_s"]
    assert event["actual_key_event_name"] == "bilateral_grasp_onset"
    assert event["actual_key_event_source"] == "persisted_bilateral_contact"


def test_passive_and_no_contact_events_match_persisted_evaluator_semantics() -> None:
    passive_scenario = SourceMujocoBackend().compile_case(_case("P0c-review-05"))
    passive = _measured_visibility_key_event(passive_scenario, (), ())
    assert passive["actual_key_event_name"] == "task_interaction"
    assert passive["actual_key_event_source"] == "planned_source_scenario_event"

    miss_scenario = SourceMujocoBackend().compile_case(_case("F1a-review-01"))
    miss = _measured_visibility_key_event(miss_scenario, (), ())
    assert miss["actual_key_event_name"] == "task_interaction"
    assert miss["actual_key_event_source"] == (
        "planned_interception_for_measured_miss"
    )
    assert miss["physical_contact_applicable"] is False


def test_p0a_visibility_uses_first_persisted_surface_contact() -> None:
    scenario = SourceMujocoBackend().compile_case(_case("P0a-review-00"))
    event = _measured_visibility_key_event(
        scenario,
        (),
        (
            {
                "timestamp": 0.305,
                "contact_category": "task_surface",
                "counterpart_geom_id": 17,
            },
            {
                "timestamp": 0.315,
                "contact_category": "task_surface",
                "counterpart_geom_id": 23,
            },
        ),
    )

    assert event["actual_key_event_name"] == "task_surface_contact_onset"
    assert event["actual_key_event_time_s"] == 0.305
    assert event["actual_key_event_source"] == (
        "persisted_task_surface_contact"
    )
    assert event["physical_contact_applicable"] is True
    assert event["contact_counterpart_geom_ids"] == [17]


def test_p0c_visibility_uses_measured_rebound_contact() -> None:
    scenario = SourceMujocoBackend().compile_case(_case("P0c-review-00"))
    event = _measured_visibility_key_event(
        scenario,
        (),
        (
            {
                "timestamp": 0.308333333333,
                "contact_category": "task_surface",
                "counterpart_geom_id": 31,
            },
        ),
    )

    assert event["actual_key_event_name"] == "task_surface_contact_onset"
    assert event["actual_key_event_time_s"] == pytest.approx(0.308333333333)
    assert event["actual_key_event_source"] == "persisted_task_surface_contact"
    assert event["physical_contact_applicable"] is True
    assert event["contact_counterpart_geom_ids"] == [31]


@pytest.mark.parametrize("rollout_index", range(6))
def test_p0b_fixed_six_visibility_and_strips_use_persisted_apex(
    rollout_index: int,
) -> None:
    scenario = SourceMujocoBackend().compile_case(
        _case(f"P0b-review-{rollout_index:02d}")
    )
    apex_time_s = scenario.object_initial_linear_velocity_m_s[2] / abs(
        scenario.gravity_m_s2[2]
    )
    delta_s = 0.01
    state_rows = []
    for timestamp_s in (apex_time_s - delta_s, apex_time_s + delta_s):
        relative_s = timestamp_s - apex_time_s
        state_rows.append(
            {
                "timestamp": timestamp_s,
                "object.position": [0.0, 0.0, 1.0 - 0.5 * abs(scenario.gravity_m_s2[2]) * relative_s**2],
                "object.linear_velocity": [
                    scenario.object_initial_linear_velocity_m_s[0],
                    scenario.object_initial_linear_velocity_m_s[1],
                    scenario.gravity_m_s2[2] * relative_s,
                ],
                "object.motion_mode": "free_flight",
            }
        )
    event = _measured_visibility_key_event(scenario, state_rows, ())

    assert event["actual_key_event_name"] == "projectile_apex"
    assert event["actual_key_event_time_s"] == pytest.approx(apex_time_s)
    assert event["actual_key_event_source"] == "persisted_free_flight_apex"
    assert event["physical_contact_applicable"] is False
    timestamps = fixed_duration_frame_timestamps(
        scenario.duration_s, scenario.video_hz
    )
    strip = event_strip_frame_indices(
        timestamps, event["actual_key_event_time_s"]
    )
    assert len(strip) == 5
    assert len(set(strip.values())) == 5


def test_visibility_replay_accepts_persisted_passive_event_sources() -> None:
    surface = _passing_visibility(actuated=False)
    surface.update(
        actual_key_event_name="task_surface_contact_onset",
        actual_key_event_source="persisted_task_surface_contact",
        physical_contact_applicable=True,
        contact_counterpart_geom_ids=[3],
        contact_exact_fixture_geom_ids=[3],
        contact_visible_fixture_geom_ids_by_view={
            "main": [3],
            "secondary": [3],
        },
        contact_fixture_pixel_counts_by_view={"main": 16, "secondary": 16},
        actual_contact_counterpart_visible_at_key_event=True,
        contact_occluded_both_views=False,
    )
    surface_result = _replay_visibility(
        surface,
        actuated=False,
        event_rows=[
            {
                "timestamp": surface["actual_key_event_time_s"],
                "contact_category": "task_surface",
                "counterpart_geom_id": 3,
            }
        ],
    )
    assert surface_result.passed, surface_result.hard_failures

    apex = _passing_visibility(actuated=False)
    apex.update(
        actual_key_event_name="projectile_apex",
        actual_key_event_source="persisted_free_flight_apex",
    )
    apex_result = _replay_visibility(apex, actuated=False)
    assert apex_result.passed, apex_result.hard_failures


def test_contact_collision_geom_uses_only_same_body_rendered_proxy() -> None:
    evidence = _contact_body_proxy_visibility(
        counterpart_geom_ids=[10],
        tool_geom_body_ids={"10": 7, "11": 7, "20": 8},
        key_geom_pixels={
            "main": {"10": 0, "11": 16, "20": 200},
            "secondary": {"10": 0, "11": 0, "20": 200},
        },
        key_target_visible={"main": True, "secondary": True},
        required_views=("main", "secondary"),
        minimum_area_px=16,
    )

    assert evidence["resolution_complete"] is True
    assert evidence["counterpart_body_ids"] == [7]
    assert evidence["proxy_geom_ids"] == [10, 11]
    assert evidence["visible_proxy_geom_ids_by_view"]["main"] == [11]
    assert evidence["proxy_pixel_counts_by_view"] == {
        "main": 16,
        "secondary": 0,
    }
    assert evidence["visible"] is True


def test_tool_render_masks_include_hash_bound_same_body_visual_geoms() -> None:
    left, right = _tool_render_geom_ids_by_side(
        {
            "tool_geom_body_ids": {
                "89": 17,
                "90": 17,
                "91": 17,
                "92": 17,
                "103": 23,
                "104": 23,
                "105": 23,
                "106": 23,
                "200": 99,
            },
            "left_tool_geom_ids": [103, 104],
            "right_tool_geom_ids": [89, 90],
            "left_tool_body_id": 23,
            "right_tool_body_id": 17,
        }
    )

    assert left == (103, 104, 105, 106)
    assert right == (89, 90, 91, 92)
    assert 200 not in left
    assert 200 not in right


def test_visibility_replay_recomputes_side_pixels_from_compiled_render_geoms() -> None:
    visibility = _passing_visibility(actuated=True)
    # Frame 2 is not duplicated into a checkpoint, so this isolates the
    # per-frame aggregate from all higher-level summary bindings.
    visibility["views"]["main"]["frames"][2]["left_tool_pixel_count"] = 17
    visibility["views"]["main"]["frames"][2]["tool_pixel_count"] = 33

    result = _replay_visibility(
        visibility,
        actuated=True,
        event_rows=[
            {
                "timestamp": visibility["actual_key_event_time_s"],
                "contact_category": "gripper",
                "counterpart_geom_id": geom_id,
            }
            for geom_id in (1, 2)
        ],
    )

    assert not result.passed
    assert any(
        "left-tool pixels differ from compiled same-body render geoms" in failure
        for failure in result.hard_failures
    )


@pytest.mark.parametrize(
    ("counterpart_geom_ids", "tool_geom_body_ids"),
    (
        ([99], {"10": 7, "20": 8}),
        ([10], {"10": 7, "20": 8}),
    ),
)
def test_contact_proxy_visibility_fails_closed_on_unresolved_or_other_body_pixels(
    counterpart_geom_ids: list[int], tool_geom_body_ids: dict[str, int]
) -> None:
    evidence = _contact_body_proxy_visibility(
        counterpart_geom_ids=counterpart_geom_ids,
        tool_geom_body_ids=tool_geom_body_ids,
        key_geom_pixels={
            "main": {"10": 0, "20": 200},
            "secondary": {"10": 0, "20": 200},
        },
        key_target_visible={"main": True, "secondary": True},
        required_views=("main", "secondary"),
        minimum_area_px=16,
    )

    assert evidence["visible"] is False


def test_persisted_source_qc_rejects_missing_visibility(tmp_path) -> None:
    result, _ = QCValidator(tmp_path, deep_video_checks=False)._validate_episode(
        _qc_record(None)
    )
    assert any(
        "lacks rendered visibility_qc metadata" in failure
        for failure in result.hard_failures
    )


def test_persisted_source_qc_rejects_tampered_visibility_hash(tmp_path) -> None:
    visibility = _passing_visibility(actuated=False)
    stale_hash = sha256_json(visibility)
    visibility["final_state_visible_in_any_view"] = False
    visibility["checkpoints"]["final"]["visible_in_any_view"] = False
    result, _ = QCValidator(tmp_path, deep_video_checks=False)._validate_episode(
        _qc_record(visibility, visibility_hash=stale_hash)
    )
    assert any("visibility QC hash binding changed" in value for value in result.hard_failures)
    assert any("final target state" in value for value in result.hard_failures)


def test_native_visibility_contract_remains_unchanged(tmp_path) -> None:
    native_visibility = {
        "schema_version": NATIVE_VISUAL_QC_SCHEMA,
        "evaluated": True,
        "key_event_visible_in_any_view": True,
        "critically_cropped": False,
        "contact_occluded_both_views": False,
        "target_visible_frame_fraction": float(
            NATIVE_VISUAL_THRESHOLDS["minimum_target_visible_frame_fraction"]
        ),
        "minimum_bbox_margin_px": float(
            NATIVE_VISUAL_THRESHOLDS["minimum_bbox_margin_px"]
        ),
        "key_event_object_area_px": int(
            NATIVE_VISUAL_THRESHOLDS["minimum_key_event_object_area_px"]
        ),
        "tool_visible_at_key_event": True,
        "fixture_visible_at_key_event": True,
        "camera_roles_correct": True,
        "maximum_underexposed_fraction": 0.0,
        "maximum_overexposed_fraction": 0.0,
    }
    result, _ = QCValidator(tmp_path, deep_video_checks=False)._validate_episode(
        _qc_record(native_visibility, backend="native_mujoco")
    )
    visibility_failures = [
        value
        for value in result.hard_failures
        if "visibility" in value or "visible" in value or "cropped" in value
    ]
    assert visibility_failures == []


@pytest.fixture(scope="module")
def rendered_visibility_results():
    backend = SourceMujocoBackend()
    return {
        "P0c-review-05": backend.run(_case("P0c-review-05"), render=True),
        "F1a-review-00": backend.run(_case("F1a-review-00"), render=True),
        "F1a-review-04": backend.run(_case("F1a-review-04"), render=True),
    }


@pytest.mark.integration
def test_rendered_p0_and_f1_visibility_use_their_physical_counterparts(
    rendered_visibility_results,
) -> None:
    passive = rendered_visibility_results["P0c-review-05"].visibility_qc
    assert passive["evaluated"] is True
    assert passive["target_visible_frame_fraction"] == 1.0
    assert passive["apex_visible_in_any_view"] is True
    assert passive["final_state_visible_in_any_view"] is True
    assert passive["tool_visibility_applicable"] is False
    assert passive["tool_visible_at_key_event"] is None
    assert passive["fixture_visibility_applicable"] is True
    assert passive["fixture_visible_at_key_event"] is True

    actuated = rendered_visibility_results["F1a-review-00"].visibility_qc
    assert actuated["evaluated"] is True
    assert actuated["target_visible_frame_fraction"] == 1.0
    assert actuated["apex_visible_in_any_view"] is True
    assert actuated["final_state_visible_in_any_view"] is True
    assert actuated["tool_visibility_applicable"] is True
    assert actuated["tool_visible_at_key_event"] is True
    assert actuated["fixture_visibility_applicable"] is False
    assert actuated["fixture_visible_at_key_event"] is None


@pytest.mark.integration
def test_rendered_generic_contact_resolves_collision_geom_to_visible_body_proxy(
    rendered_visibility_results,
) -> None:
    visibility = rendered_visibility_results["F1a-review-04"].visibility_qc
    counterpart_ids = set(visibility["contact_counterpart_geom_ids"])
    visible_proxy_ids = {
        geom_id
        for values in visibility[
            "contact_visible_proxy_geom_ids_by_view"
        ].values()
        for geom_id in values
    }

    assert visibility["actual_key_event_source"] == "persisted_contact_event"
    assert visibility["contact_body_proxy_resolution_complete"] is True
    assert visibility["unresolved_contact_counterpart_geom_ids"] == []
    assert visibility["contact_counterpart_body_ids"]
    assert visible_proxy_ids - counterpart_ids
    assert visibility["actual_contact_counterpart_visible_at_key_event"] is True
    assert visibility["contact_occluded_both_views"] is False


@pytest.mark.integration
def test_materialized_source_visibility_is_content_bound(
    rendered_visibility_results,
) -> None:
    case = _case("P0c-review-05")
    declaration = prepare_source_review_declaration(case, episode_index=0)
    materialization = materialize_source_mujoco_result(
        _entry(declaration), rendered_visibility_results[case.case_id]
    )
    record = materialization.record
    assert record.extras["visibility_qc"] == dict(
        rendered_visibility_results[case.case_id].visibility_qc
    )
    assert record.extras["visibility_qc_sha256"] == sha256_json(
        record.extras["visibility_qc"]
    )
    tampered = deepcopy(record.extras["visibility_qc"])
    tampered["final_state_visible_in_any_view"] = False
    assert sha256_json(tampered) != record.extras["visibility_qc_sha256"]
