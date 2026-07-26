from __future__ import annotations

from pathlib import Path

import pytest

from dynamic_robot_dataset.common.arrow_schema import (
    ARROW_SIDECAR_SCHEMA_VERSION,
    canonical_sidecar_table,
    normalize_extra_field_declarations,
)
from dynamic_robot_dataset.common.episode_writer import EpisodeWriter
from dynamic_robot_dataset.common.paths import ResumeMismatchError, atomic_write_json
from dynamic_robot_dataset.common.schema import DatasetInfo, EpisodeRecord


def _source_scenario() -> dict[str, object]:
    return {
        "embodiment": {
            "end_effector": "franka_hand",
            "action_names": [f"actuator_{index}" for index in range(8)],
            "action_semantics": "actual_actuator_command/v1",
        }
    }


def test_core_frame_types_are_explicit_and_source_fields_are_required() -> None:
    pa = pytest.importorskip("pyarrow")
    row = {
        "episode_index": 4,
        "task_index": 2,
        "frame_index": 0,
        "video_frame_index": 0,
        "timestamp": 0,
        "simulation_timestamp": 0.0,
        "synchronization_error_s": 0,
        "action.actuator_command": list(range(8)),
        "simulator.applied_actuator_ctrl": list(range(8)),
        "action.mode": "actual_actuator_command/v1",
    }
    table = canonical_sidecar_table(
        pa, "frame", [row], source_scenario=_source_scenario()
    )

    assert str(table.schema.field("episode_index").type) == "int64"
    assert str(table.schema.field("timestamp").type) == "double"
    assert str(table.schema.field("simulation_timestamp").type) == "double"
    assert str(table.schema.field("synchronization_error_s").type) == "double"
    assert str(table.schema.field("action.actuator_command").type) == "list<item: double>"
    assert str(table.schema.field("simulator.applied_actuator_ctrl").type) == "list<item: double>"
    assert table.schema.metadata[b"contract"] == b"dynamic-robot-frames/v1"
    assert table.schema.metadata[b"schema_strategy"] == ARROW_SIDECAR_SCHEMA_VERSION.encode()

    missing_timing = dict(row)
    missing_timing.pop("simulation_timestamp")
    with pytest.raises(ValueError, match="requires simulation_timestamp"):
        canonical_sidecar_table(
            pa, "frame", [missing_timing], source_scenario=_source_scenario()
        )

    wrong_action = {**row, "action.actuator_command": [0.0] * 7}
    with pytest.raises(ValueError, match="requires 8 actual actuator commands"):
        canonical_sidecar_table(
            pa, "frame", [wrong_action], source_scenario=_source_scenario()
        )

    wrong_echo = {**row, "simulator.applied_actuator_ctrl": [0.0] * 8}
    with pytest.raises(ValueError, match="differs from applied data.ctrl"):
        canonical_sidecar_table(
            pa, "frame", [wrong_echo], source_scenario=_source_scenario()
        )


def test_contact_penetration_and_category_have_stable_types_when_empty() -> None:
    pa = pytest.importorskip("pyarrow")
    table = canonical_sidecar_table(pa, "events", [])

    assert str(table.schema.field("penetration_depth_m").type) == "double"
    assert str(table.schema.field("contact_category").type) == "string"
    assert str(table.schema.field("counterpart_geom_id").type) == "int64"
    assert str(table.schema.field("normal_world").type) == "list<item: double>"
    assert table.schema.metadata[b"contract"] == b"dynamic-robot-contact-events/v2"


def test_contact_counterpart_geom_id_round_trips_as_nullable_int64() -> None:
    pa = pytest.importorskip("pyarrow")
    table = canonical_sidecar_table(
        pa,
        "events",
        [
            {"episode_index": 0, "timestamp": 0.1, "counterpart_geom_id": 17},
            {"episode_index": 0, "timestamp": 0.2, "counterpart_geom_id": None},
        ],
    )

    assert str(table.schema.field("counterpart_geom_id").type) == "int64"
    assert table.column("counterpart_geom_id").to_pylist() == [17, None]


def test_all_null_extension_requires_a_bound_declaration() -> None:
    pa = pytest.importorskip("pyarrow")
    rows = [{"episode_index": 0, "timestamp": 0.0, "custom.sensor": None}]

    with pytest.raises(ValueError, match="all-null"):
        canonical_sidecar_table(pa, "high_rate", rows)

    table = canonical_sidecar_table(
        pa,
        "high_rate",
        rows,
        declared_extra_fields={"custom.sensor": "list<float64>"},
    )
    assert str(table.schema.field("custom.sensor").type) == "list<item: double>"
    assert table.column("custom.sensor").to_pylist() == [None]


def test_extra_schema_declaration_is_part_of_resume_identity(tmp_path: Path) -> None:
    config = {
        "seed": 1,
        "arrow_extra_fields": {"frame": {"diagnostic.score": "float64"}},
    }
    writer = EpisodeWriter(tmp_path / "run", config)
    assert writer.writer_settings["arrow_schema"]["version"] == ARROW_SIDECAR_SCHEMA_VERSION
    assert writer.writer_settings["arrow_schema"]["extra_fields"]["frame"] == {
        "diagnostic.score": "float64"
    }

    changed = {
        "seed": 1,
        "arrow_extra_fields": {"frame": {"diagnostic.score": "string"}},
    }
    with pytest.raises(ResumeMismatchError):
        EpisodeWriter(tmp_path / "run", changed, resume=True)


def test_extra_declarations_cannot_override_core_fields() -> None:
    with pytest.raises(ValueError, match="overrides a core field"):
        normalize_extra_field_declarations({"frame": {"timestamp": "string"}})


def test_metadata_validation_failure_does_not_publish_irreversible_seal(
    tmp_path: Path,
) -> None:
    writer = EpisodeWriter(tmp_path / "run", {"seed": 7})
    record = EpisodeRecord(
        episode_uuid="00000000-0000-4000-8000-000000000007",
        episode_index=0,
        counterfactual_bundle_id="bundle-7",
        physics_counterfactual_family_id="physics-7",
        split_group_id="group-7",
        scene_seed=7,
        branch_seed=8,
        family="falling_catch",
        subfamily="centered_vertical_drop",
        intended_branch="success_seeking",
        actual_outcome="success",
        task_success=True,
        failure_mode="none",
        source_generator="test",
        source_generator_version="1",
        config_hash=writer.config_hash,
        simulator_name="test",
        simulator_version="1",
        renderer="test",
        task_index=0,
    )
    record.validate()
    atomic_write_json(
        writer._marker(record.episode_uuid),
        {"config_hash": writer.config_hash, "episode": record.to_dict()},
    )

    with pytest.raises(ValueError, match="No task row"):
        writer.finalize(
            DatasetInfo(name="invalid-finalize"),
            tasks=[{"task_index": 0, "family": "wrong", "subfamily": "wrong"}],
        )
    assert not writer.seal_path.exists()
