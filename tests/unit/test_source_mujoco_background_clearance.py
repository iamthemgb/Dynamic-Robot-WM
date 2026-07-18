from __future__ import annotations

from copy import deepcopy

import pytest

from dynamic_robot_dataset.backends.source_mujoco import SourceMujocoBackend
from dynamic_robot_dataset.backends.source_mujoco.backend import (
    SOURCE_MUJOCO_BACKGROUND_CLEARANCE_SCHEMA,
    _evaluate_background_clearance_rows,
)
from dynamic_robot_dataset.common.hashing import sha256_json
from dynamic_robot_dataset.common.qc import (
    EpisodeQC,
    _validate_source_mujoco_background_clearance,
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
    fixture_rows = sorted(
        (dict(row) for row in clearance["fixture_rows"]),
        key=lambda row: row["fixture_id"],
    )
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
        "physics": {
            "object_radius_m": object_radius_m,
            "background_clearance_static_row_count": len(background_static_rows),
            "background_clearance_static_rows_sha256": sha256_json(
                background_static_rows
            ),
            "fixture_clearance_static_row_count": len(fixture_rows),
            "fixture_clearance_static_rows_sha256": sha256_json(fixture_rows),
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
