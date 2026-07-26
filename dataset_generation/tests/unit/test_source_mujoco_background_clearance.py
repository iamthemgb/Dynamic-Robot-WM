from __future__ import annotations

from copy import deepcopy

import pytest

from dynamic_robot_dataset.backends.source_mujoco import (
    SourceMujocoBackend,
    prepare_review_case,
)
from dynamic_robot_dataset.backends.source_mujoco.backend import (
    SOURCE_MUJOCO_BACKGROUND_CLEARANCE_SCHEMA,
    _evaluate_background_clearance_rows,
)
from dynamic_robot_dataset.common.hashing import sha256_json
from dynamic_robot_dataset.common.qc import (
    EpisodeQC,
    _validate_source_mujoco_background_clearance,
    _validate_source_mujoco_visibility_qc,
)
from dynamic_robot_dataset.common.review_suite import build_review_suite_plan


def _aabb(
    minimum: tuple[float, float, float],
    maximum: tuple[float, float, float],
) -> dict:
    return {
        "minimum_m": list(minimum),
        "maximum_m": list(maximum),
        "method": "mujoco_compiled_local_aabb_transformed/v1",
    }


def _background(
    stable_id: str,
    minimum: tuple[float, float, float],
    maximum: tuple[float, float, float],
    *,
    collision_disabled: bool = True,
    anchored: bool = True,
) -> dict:
    return {
        "stable_id": stable_id,
        "geom_id": 10,
        "classification": "procedural",
        "source_name": stable_id,
        "catalog_slot": None,
        "body_id": 0,
        "body_name": "world",
        "body_weld_id": 0 if anchored else 1,
        "world_aabb": _aabb(minimum, maximum),
        "contype": 0 if collision_disabled else 1,
        "conaffinity": 0 if collision_disabled else 1,
        "collision_disabled": collision_disabled,
        "anchored": anchored,
    }


def _state(timestamp: float, position: tuple[float, float, float]) -> dict:
    return {"timestamp": timestamp, "object.position": list(position)}


def _fixture(
    fixture_id: str,
    role: str,
    minimum: tuple[float, float, float],
    maximum: tuple[float, float, float],
    *,
    geom_id: int,
    structural: bool = False,
) -> dict:
    return {
        "fixture_id": fixture_id,
        "role": role,
        "geom_id": geom_id,
        "fixture_class": "structural_support" if structural else "task_fixture",
        "expected_task_contact": not structural,
        "supports_fixture_id": "owned_table" if structural else None,
        "grounded_fixture_id": "floor" if structural else None,
        "body_id": 0,
        "body_weld_id": 0,
        "contype": 1,
        "conaffinity": 1,
        "ground_contact_distance_m": 0.0 if structural else None,
        "supported_contact_distance_m": -0.001 if structural else None,
        "support_interface_maximum_mismatch_m": 0.0011 if structural else None,
        "support_interface_tolerance_m": 0.002 if structural else None,
        "world_aabb": _aabb(minimum, maximum),
    }


def _case(case_id: str):
    return next(
        case
        for case in build_review_suite_plan().cases
        if case.case_id == case_id
    )


def _static_contract(clearance: dict, *, object_radius_m: float) -> dict:
    static_fields = (
        "stable_id",
        "geom_id",
        "classification",
        "source_name",
        "catalog_slot",
        "body_id",
        "body_name",
        "body_weld_id",
        "world_aabb",
        "contype",
        "conaffinity",
    )
    background_static_rows = sorted(
        (
            {name: row.get(name) for name in static_fields}
            for row in clearance["background_rows"]
        ),
        key=lambda row: row["stable_id"],
    )
    descriptor_rows = [
        {
            name: row[name]
            for name in (
                "stable_id",
                "geom_id",
                "classification",
                "source_name",
                "catalog_slot",
            )
        }
        for row in background_static_rows
    ]
    fixture_static_fields = (
        "fixture_id",
        "role",
        "geom_id",
        "fixture_class",
        "expected_task_contact",
        "supports_fixture_id",
        "grounded_fixture_id",
        "body_id",
        "body_weld_id",
        "contype",
        "conaffinity",
        "ground_contact_distance_m",
        "supported_contact_distance_m",
        "support_interface_maximum_mismatch_m",
        "support_interface_tolerance_m",
        "world_aabb",
    )
    fixture_rows = sorted(
        (
            {name: row.get(name) for name in fixture_static_fields}
            for row in clearance["fixture_rows"]
        ),
        key=lambda row: row["fixture_id"],
    )
    structural_rows = [
        row for row in fixture_rows if row["fixture_class"] == "structural_support"
    ]
    source_fixtures = [
        {
            "fixture_id": row["fixture_id"],
            "fixture_type": row["role"],
            "pose": {
                "position_m": [0.0, 0.0, 0.0],
                "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            },
            "physical": True,
            "anchored": True,
            "parameters": {
                "fixture_class": row["fixture_class"],
                "expected_task_contact": row["expected_task_contact"],
                "supports_fixture_id": row["supports_fixture_id"],
                "grounded_plane_z_m": (
                    0.0 if row["fixture_class"] == "structural_support" else None
                ),
                "support_interface_maximum_mismatch_m": row[
                    "support_interface_maximum_mismatch_m"
                ],
                "support_interface_tolerance_m": row[
                    "support_interface_tolerance_m"
                ],
                "collision_enabled": True,
            },
        }
        for row in fixture_rows
    ]
    clearance["classification"] = {
        "source": "CompiledSourceModel.background_geom_descriptors/v1",
        "descriptor_count": len(descriptor_rows),
        "descriptors_sha256": sha256_json(descriptor_rows),
        "background_static_rows_sha256": sha256_json(background_static_rows),
        "fixture_static_rows_sha256": sha256_json(fixture_rows),
        "exclusions": {},
        "exclusions_sha256": sha256_json({}),
    }
    return {
        "fixtures": source_fixtures,
        "physics": {
            "object_radius_m": object_radius_m,
            "background_clearance_static_row_count": len(background_static_rows),
            "background_clearance_static_rows_sha256": sha256_json(
                background_static_rows
            ),
            "fixture_clearance_static_row_count": len(fixture_rows),
            "fixture_clearance_static_rows_sha256": sha256_json(fixture_rows),
            "structural_support_fixture_ids": sorted(
                row["fixture_id"] for row in structural_rows
            ),
            "structural_support_geom_ids": sorted(
                int(row["geom_id"]) for row in structural_rows
            ),
            "structural_support_station_by_geom": {
                str(int(row["geom_id"])): str(row["fixture_id"]).rsplit(
                    "_y", 1
                )[0]
                for row in structural_rows
            },
            "task_contact_fixture_ids": sorted(
                row["fixture_id"]
                for row in fixture_rows
                if row["expected_task_contact"] is True
            ),
            "task_contact_geom_ids": sorted(
                int(row["geom_id"])
                for row in fixture_rows
                if row["expected_task_contact"] is True
            ),
        }
    }


def test_runtime_clearance_checks_every_high_rate_position() -> None:
    rows = [
        _state(0.0, (0.0, 0.0, 0.5)),
        _state(0.1, (1.0, 0.0, 0.5)),
        _state(0.2, (2.0, 0.0, 0.5)),
    ]
    clearance = _evaluate_background_clearance_rows(
        background_rows=[
            _background("procedural:late_obstacle", (1.85, -0.1, 0.4), (2.1, 0.1, 0.6))
        ],
        fixture_rows=[],
        high_rate_rows=rows,
        object_radius_m=0.05,
    )

    assert clearance["schema_version"] == SOURCE_MUJOCO_BACKGROUND_CLEARANCE_SCHEMA
    assert clearance["evaluated"] is True
    assert clearance["object_sweep"]["sample_count"] == 3
    assert clearance["object_swept_volume_clear"] is False
    assert clearance["clearance_pass"] is False
    evidence = clearance["background_rows"][0]
    assert evidence["object_swept_clear"] is False
    assert evidence["object_intersection_sample_count"] == 1
    assert evidence["first_object_intersection"] == {
        "sample_index": 2,
        "timestamp_s": 0.2,
        "object_position_m": [2.0, 0.0, 0.5],
    }

    changed = deepcopy(rows)
    changed[2]["object.position"][0] += 0.01
    changed_clearance = _evaluate_background_clearance_rows(
        background_rows=[
            _background("procedural:late_obstacle", (1.85, -0.1, 0.4), (2.1, 0.1, 0.6))
        ],
        fixture_rows=[],
        high_rate_rows=changed,
        object_radius_m=0.05,
    )
    assert (
        changed_clearance["object_sweep"]["exact_rows_sha256"]
        != clearance["object_sweep"]["exact_rows_sha256"]
    )


def test_runtime_clearance_rejects_collision_fixture_overlap_and_floating_background() -> None:
    clearance = _evaluate_background_clearance_rows(
        background_rows=[
            _background(
                "procedural:invalid",
                (0.8, 0.8, 0.8),
                (1.2, 1.2, 1.2),
                collision_disabled=False,
                anchored=False,
            )
        ],
        fixture_rows=[
            {
                "fixture_id": "owned_table",
                "role": "table",
                "geom_id": 3,
                "world_aabb": _aabb((0.9, 0.9, 0.9), (1.1, 1.1, 1.1)),
            }
        ],
        high_rate_rows=[_state(0.0, (0.0, 0.0, 0.0))],
        object_radius_m=0.05,
    )

    assert clearance["all_background_collision_disabled"] is False
    assert clearance["all_background_anchored"] is False
    assert clearance["fixture_intersection_clear"] is False
    assert clearance["collision_failure_ids"] == ["procedural:invalid"]
    assert clearance["anchoring_failure_ids"] == ["procedural:invalid"]
    assert clearance["fixture_intersection_failure_ids"] == [
        "procedural:invalid"
    ]
    assert clearance["background_rows"][0]["intersecting_fixture_ids"] == [
        "owned_table"
    ]


def test_structural_support_chain_is_physical_grounded_and_sweep_clear() -> None:
    fixtures = [
        _fixture(
            "owned_table",
            "table",
            (-0.5, -0.5, 0.70),
            (0.5, 0.5, 0.74),
            geom_id=3,
        ),
        _fixture(
            "owned_structural_leg_x0_y0",
            "structural_support",
            (0.40, 0.40, 0.0),
            (0.46, 0.46, 0.70),
            geom_id=4,
            structural=True,
        ),
    ]
    rows = [
        _state(0.0, (-0.2, 0.0, 0.8)),
        _state(0.1, (0.2, 0.0, 0.8)),
    ]

    clearance = _evaluate_background_clearance_rows(
        background_rows=[],
        fixture_rows=fixtures,
        high_rate_rows=rows,
        object_radius_m=0.05,
    )

    assert clearance["clearance_pass"] is True
    assert clearance["task_contact_fixture_ids"] == ["owned_table"]
    assert clearance["structural_support_fixture_ids"] == [
        "owned_structural_leg_x0_y0"
    ]
    support = next(
        row
        for row in clearance["fixture_rows"]
        if row["fixture_class"] == "structural_support"
    )
    assert support["expected_task_contact"] is False
    assert support["anchored"] is True
    assert support["collision_enabled"] is True
    assert support["ground_contact_within_tolerance"] is True
    assert support["supported_contact_within_tolerance"] is True
    assert support["declared_interface_within_tolerance"] is True
    assert support["support_target_valid"] is True
    assert support["support_chain_valid"] is True
    assert support["object_swept_clear"] is True

    floating = deepcopy(fixtures)
    floating[1]["ground_contact_distance_m"] = 0.01
    rejected_chain = _evaluate_background_clearance_rows(
        background_rows=[],
        fixture_rows=floating,
        high_rate_rows=rows,
        object_radius_m=0.05,
    )
    assert rejected_chain["structural_support_chain_pass"] is False
    assert rejected_chain["clearance_pass"] is False

    intersecting_rows = [_state(0.0, (0.43, 0.43, 0.35))]
    rejected_sweep = _evaluate_background_clearance_rows(
        background_rows=[],
        fixture_rows=fixtures,
        high_rate_rows=intersecting_rows,
        object_radius_m=0.05,
    )
    assert rejected_sweep["structural_support_swept_volume_clear"] is False
    assert rejected_sweep["structural_support_sweep_failure_ids"] == [
        "owned_structural_leg_x0_y0"
    ]
    assert rejected_sweep["clearance_pass"] is False


def test_persisted_clearance_qc_replays_and_rejects_tampered_sweep() -> None:
    rows = [
        _state(0.0, (0.0, 0.0, 0.5)),
        _state(0.1, (0.1, 0.0, 0.5)),
    ]
    clearance = _evaluate_background_clearance_rows(
        background_rows=[],
        fixture_rows=[],
        high_rate_rows=rows,
        object_radius_m=0.05,
    )
    source_scenario = _static_contract(clearance, object_radius_m=0.05)
    digest = sha256_json(clearance)
    result = EpisodeQC("episode", 0, False)
    _validate_source_mujoco_background_clearance(
        result,
        clearance,
        high_rate_rows=rows,
        source_scenario=source_scenario,
        backend_provenance={
            "background_clearance": clearance,
            "background_clearance_sha256": digest,
        },
        runtime_audit={"background_clearance_sha256": digest},
        stored_sha256=digest,
    )
    assert result.passed, result.hard_failures

    tampered = deepcopy(clearance)
    tampered["object_sweep"]["exact_rows_sha256"] = "0" * 64
    rejected = EpisodeQC("episode", 0, False)
    _validate_source_mujoco_background_clearance(
        rejected,
        tampered,
        high_rate_rows=rows,
        source_scenario=source_scenario,
        backend_provenance={
            "background_clearance": clearance,
            "background_clearance_sha256": digest,
        },
        runtime_audit={"background_clearance_sha256": digest},
        stored_sha256=digest,
    )
    assert any("persisted high-rate trajectory" in failure for failure in rejected.hard_failures)


def test_persisted_clearance_recomputes_rehashed_primitive_claims() -> None:
    rows = [_state(0.0, (0.0, 0.0, 0.5))]
    clearance = _evaluate_background_clearance_rows(
        background_rows=[
            _background(
                "procedural:far_wall",
                (2.0, 2.0, 2.0),
                (2.2, 2.2, 2.2),
            )
        ],
        fixture_rows=[],
        high_rate_rows=rows,
        object_radius_m=0.05,
    )
    tampered = deepcopy(clearance)
    tampered["background_rows"][0]["contype"] = 1
    tampered["background_rows"][0]["conaffinity"] = 1
    # Simulate an attacker updating every self-authored hash and even the
    # planned static hash while leaving the favorable derived claims intact.
    tampered["background_rows_sha256"] = sha256_json(tampered["background_rows"])
    source_scenario = _static_contract(tampered, object_radius_m=0.05)
    digest = sha256_json(tampered)
    result = EpisodeQC("episode", 0, False)

    _validate_source_mujoco_background_clearance(
        result,
        tampered,
        high_rate_rows=rows,
        source_scenario=source_scenario,
        backend_provenance={
            "background_clearance": tampered,
            "background_clearance_sha256": digest,
        },
        runtime_audit={"background_clearance_sha256": digest},
        stored_sha256=digest,
    )

    assert any(
        "background clearance replay changed" in failure
        for failure in result.hard_failures
    )


def test_persisted_support_primitives_remain_bound_to_source_scenario() -> None:
    fixtures = [
        _fixture(
            "owned_table",
            "table",
            (-0.5, -0.5, 0.70),
            (0.5, 0.5, 0.74),
            geom_id=3,
        ),
        _fixture(
            "owned_structural_leg_x0_y0",
            "structural_support",
            (0.40, 0.40, 0.0),
            (0.46, 0.46, 0.70),
            geom_id=4,
            structural=True,
        ),
    ]
    rows = [_state(0.0, (0.0, 0.0, 0.8))]
    clearance = _evaluate_background_clearance_rows(
        background_rows=[],
        fixture_rows=fixtures,
        high_rate_rows=rows,
        object_radius_m=0.05,
    )
    source_scenario = _static_contract(clearance, object_radius_m=0.05)
    tampered = deepcopy(clearance)
    support = next(
        row
        for row in tampered["fixture_rows"]
        if row["fixture_class"] == "structural_support"
    )
    support["body_weld_id"] = 1
    # Rehash every mutable envelope while retaining the favorable derived
    # claims.  The immutable SourceScenarioSpec static-row hash and pure replay
    # must still reject the support as no longer world-anchored.
    tampered["fixture_rows_sha256"] = sha256_json(tampered["fixture_rows"])
    digest = sha256_json(tampered)
    result = EpisodeQC("episode", 0, False)

    _validate_source_mujoco_background_clearance(
        result,
        tampered,
        high_rate_rows=rows,
        source_scenario=source_scenario,
        backend_provenance={
            "background_clearance": tampered,
            "background_clearance_sha256": digest,
        },
        runtime_audit={"background_clearance_sha256": digest},
        stored_sha256=digest,
    )

    assert any(
        "fixture geometry differs from SourceScenarioSpec" in failure
        or "not physically anchored" in failure
        or "background clearance replay changed" in failure
        for failure in result.hard_failures
    )


def test_persisted_clearance_radius_is_bound_to_source_scenario() -> None:
    rows = [_state(0.0, (0.0, 0.0, 0.5))]
    clearance = _evaluate_background_clearance_rows(
        background_rows=[],
        fixture_rows=[],
        high_rate_rows=rows,
        object_radius_m=0.005,
    )
    # Static geometry hashes are valid, but the immutable scenario says the
    # physical object radius is ten times larger.
    source_scenario = _static_contract(clearance, object_radius_m=0.05)
    digest = sha256_json(clearance)
    result = EpisodeQC("episode", 0, False)

    _validate_source_mujoco_background_clearance(
        result,
        clearance,
        high_rate_rows=rows,
        source_scenario=source_scenario,
        backend_provenance={
            "background_clearance": clearance,
            "background_clearance_sha256": digest,
        },
        runtime_audit={"background_clearance_sha256": digest},
        stored_sha256=digest,
    )

    assert any(
        "object radius differs from SourceScenarioSpec" in failure
        for failure in result.hard_failures
    )


@pytest.mark.integration
def test_p0d_runtime_clearance_binds_the_complete_rolling_trajectory() -> None:
    result = SourceMujocoBackend().run(_case("P0d-review-02"), render=False)
    clearance = result.background_clearance
    positions_x = [float(row["object.position"][0]) for row in result.high_rate_rows]

    assert clearance["evaluated"] is True
    assert clearance["object_sweep"]["sample_count"] == len(
        result.high_rate_rows
    )
    assert clearance["object_sweep"]["aggregate_world_aabb"]["maximum_m"][0] == pytest.approx(
        max(positions_x) + result.scenario.object_radius_m
    )
    assert max(positions_x) > result.scenario.object_initial_position_m[0] + 1.0
    assert clearance["classification"]["descriptor_count"] == len(
        clearance["background_rows"]
    )
    assert any(
        row["classification"] == "procedural"
        for row in clearance["background_rows"]
    )
    assert any(
        row["classification"] == "catalog"
        for row in clearance["background_rows"]
    )


@pytest.mark.integration
@pytest.mark.parametrize("case_id", ("P0c-review-02", "P0d-review-02"))
def test_randomized_elevated_passive_fixture_has_a_visible_grounded_support_frame(
    case_id: str,
) -> None:
    case = _case(case_id)
    spec = prepare_review_case(case)
    result = SourceMujocoBackend().run(case, render=True)
    clearance = result.background_clearance
    supports = [
        row
        for row in clearance["fixture_rows"]
        if row["fixture_class"] == "structural_support"
    ]

    assert len(supports) == 4
    assert clearance["all_physical_fixtures_anchored"] is True
    assert clearance["all_physical_fixtures_collision_enabled"] is True
    assert clearance["structural_support_chain_pass"] is True
    assert clearance["structural_support_swept_volume_clear"] is True
    assert clearance["clearance_pass"] is True
    assert all(row["expected_task_contact"] is False for row in supports)
    assert all(row["anchored"] is True for row in supports)
    assert all(row["collision_enabled"] is True for row in supports)
    assert all(row["ground_contact_within_tolerance"] is True for row in supports)
    assert all(row["supported_contact_within_tolerance"] is True for row in supports)
    assert all(row["support_chain_valid"] is True for row in supports)
    support_geom_ids = {int(row["geom_id"]) for row in supports}
    assert support_geom_ids == set(
        spec.physics["structural_support_geom_ids"]
    )
    assert sha256_json(
        [
            {
                name: row.get(name)
                for name in (
                    "fixture_id",
                    "role",
                    "geom_id",
                    "fixture_class",
                    "expected_task_contact",
                    "supports_fixture_id",
                    "grounded_fixture_id",
                    "body_id",
                    "body_weld_id",
                    "contype",
                    "conaffinity",
                    "ground_contact_distance_m",
                    "supported_contact_distance_m",
                    "support_interface_maximum_mismatch_m",
                    "support_interface_tolerance_m",
                    "world_aabb",
                )
            }
            for row in sorted(
                clearance["fixture_rows"], key=lambda value: value["fixture_id"]
            )
        ]
    ) == spec.physics["fixture_clearance_static_rows_sha256"]
    assert not any(
        row.get("contact_category") == "task_surface"
        and int(row.get("counterpart_geom_id", -1)) in support_geom_ids
        for row in result.contact_rows
    )
    support_visibility = result.visibility_qc["structural_support_visibility"]
    assert support_visibility["evaluated"] is True
    assert support_visibility["all_support_stations_visible"] is True
    assert len(support_visibility["visible_support_station_ids"]) == 2

    replay = EpisodeQC(case.episode_uuid, case.rollout_index, False)
    _validate_source_mujoco_background_clearance(
        replay,
        clearance,
        high_rate_rows=result.high_rate_rows,
        source_scenario=spec.to_dict(),
        backend_provenance=result.backend_provenance,
        runtime_audit=result.runtime_audit,
        stored_sha256=sha256_json(clearance),
    )
    assert replay.passed, replay.hard_failures

    visibility_replay = EpisodeQC(case.episode_uuid, case.rollout_index, False)
    _validate_source_mujoco_visibility_qc(
        visibility_replay,
        result.visibility_qc,
        end_effector="no_robot",
        expected_frame_count=len(result.frame_rows),
        frame_rows=result.frame_rows,
        event_rows=result.contact_rows,
        record_key_event_name=str(result.outcome["key_event_name"]),
        record_key_event_time_s=float(result.outcome["key_event_time_s"]),
        objective_key_event_source=str(result.outcome["key_event_source"]),
        event_time_tolerance_s=1.0 / result.scenario.simulation_hz,
        source_scenario=spec.to_dict(),
    )
    assert visibility_replay.passed, visibility_replay.hard_failures


@pytest.mark.integration
def test_f1_negative_runtime_clearance_is_hash_bound_and_catalog_manifest_uses_it() -> None:
    result = SourceMujocoBackend().run(_case("F1a-review-02"), render=False)
    clearance = result.background_clearance
    exact_rows = [
        {
            "sample_index": index,
            "timestamp_s": float(row["timestamp"]),
            "object_position_m": [
                float(value) for value in row["object.position"]
            ],
        }
        for index, row in enumerate(result.high_rate_rows)
    ]

    assert clearance["clearance_pass"] is True
    assert clearance["object_sweep"]["exact_rows_sha256"] == sha256_json(
        exact_rows
    )
    assert not any(
        str(row["stable_id"]).startswith("procedural:robot_table_")
        for row in clearance["background_rows"]
    )
    assert result.robocasa_asset_manifest
    for row in result.robocasa_asset_manifest:
        assert row["runtime_clearance_evaluated"] is True
        assert row["runtime_clearance_sha256"] == sha256_json(clearance)
        assert row["catalog_geom_count"] > 0
        assert row["swept_volume_clear"] is True
        assert row["fixture_intersection_clear"] is True
        assert row["blockers"] == ["rendered_occlusion_review_pending"]
