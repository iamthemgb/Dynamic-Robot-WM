from __future__ import annotations

import argparse
from collections import defaultdict
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import dynamic_robot_dataset.backends as backend_registry
import dynamic_robot_dataset.cli as cli
from dynamic_robot_dataset.common.contract_v2 import (
    CounterfactualFamilyRecord,
    CounterfactualRelation,
    validate_counterfactual_family_records,
)
from dynamic_robot_dataset.common.hashing import sha256_json
from dynamic_robot_dataset.common.native_suite import (
    build_planned_counterfactual_family_records,
    plan_suite_cases,
)
from dynamic_robot_dataset.common.qc import _objective_metric_success_reference
from dynamic_robot_dataset.common.schema import EpisodeRecord
from dynamic_robot_dataset.common.suites import expand_suite
from dynamic_robot_dataset.smoke_runner import (
    _frame_rows as smoke_frame_rows,
    _record as smoke_record,
)


ROOT = Path(__file__).resolve().parents[2]
SUITE = ROOT / "configs/families/native_acceptance_160.yaml"


def _sweep_requirements():
    config = yaml.safe_load(SUITE.read_text(encoding="utf-8"))
    return config["requirements"]["require_measured_physics_sweeps"]


def _arguments(output: Path | None, *, resume: bool = False, dry_run: bool = False):
    return argparse.Namespace(
        config=str(SUITE),
        output=None if output is None else str(output),
        resume=resume,
        dry_run=dry_run,
    )


def _record(
    item,
    index: int,
    *,
    success: bool = True,
    actual_outcome_class: str | None = None,
) -> EpisodeRecord:
    outcome = actual_outcome_class or ("success" if success else "no_op")
    failure_codes = {
        "success": "none",
        "near_miss": "spatial_near_miss",
        "contact_failure": "contact_without_completion",
        "no_op": "no_op",
        "wrong_action": "wrong_action",
    }
    success = outcome == "success"
    sweep_checks = {}
    if item.case.subfamily == "gravity_sweep":
        sweep_checks = {
            "measured_gravity_magnitude_m_s2": abs(
                float(item.episode_plan.physics["gravity"]["value"][2])
            ),
            "free_flight_pair_count": 20,
        }
    elif item.case.subfamily == "friction_sweep":
        sweep_checks = {
            "measured_effective_dynamic_friction": float(
                item.episode_plan.physics["surface_dynamic_friction"]["value"]
            ),
            "friction_response_sample_count": 20,
        }
    elif item.case.subfamily == "restitution_sweep":
        sweep_checks = {
            "measured_effective_restitution": float(
                item.episode_plan.physics["effective_restitution_target"]["value"]
            ),
            "rebound_measurement_count": 1,
        }
    return EpisodeRecord(
        episode_uuid=item.episode_uuid,
        episode_index=index,
        counterfactual_bundle_id=item.episode_plan.counterfactual_bundle_id,
        physics_counterfactual_family_id=item.episode_plan.physics_counterfactual_family_id,
        split_group_id=item.episode_plan.split_group_id,
        scene_seed=item.episode_plan.scene_seed,
        branch_seed=item.episode_plan.branch_seed,
        family=item.episode_plan.family,
        subfamily=item.episode_plan.subfamily,
        intended_branch=item.episode_plan.intended_branch,
        actual_outcome=outcome,
        task_success=success,
        failure_mode=failure_codes[outcome],
        source_generator="test.suite",
        source_generator_version="1",
        config_hash=item.episode_plan.config_hash,
        simulator_name="test",
        simulator_version="1",
        renderer="test",
        physics=cli._physics_metadata(item.episode_plan.physics),
        randomization={
            "scene_style": item.episode_plan.scene_style,
            "visual_seed": item.episode_plan.scene_seed,
        },
        camera_stream_calibration_ids={
            "observation.images.main": "main@test",
            "observation.images.secondary": "secondary@test",
        },
        extras={
            "family_plan_config_hash": item.episode_plan.config_hash,
            "planned_counterfactual_fixed_hashes": dict(
                item.episode_plan.options.get(
                    "planned_counterfactual_fixed_hashes", {}
                )
            ),
            "action_hash": item.episode_plan.action_hash,
            "initial_state_hash": sha256_json({"shared_initial_state": 1}),
            "derived_action_hash": sha256_json(
                {"persisted_action": item.episode_plan.intended_branch}
            ),
            "derived_initial_state_hash": sha256_json(
                {"persisted_shared_initial_state": 1}
            ),
            "physics_qc": {"physics_qc_pass": True, "checks": sweep_checks},
        },
        video_paths={
            "observation.images.main": f"videos/main/{index}.mp4",
            "observation.images.secondary": f"videos/secondary/{index}.mp4",
        },
    )


def test_generate_suite_dry_run_reports_mixed_truth(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli._command_generate_suite(_arguments(None, dry_run=True)) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["case_count"] == 160
    assert payload["execution_backend_counts"] == {
        "native_mujoco": 128,
        "diagnostic_quarantine": 32,
    }
    assert payload["full_native"] is False
    assert len(payload["planned_episode_uuids"]) == 160
    assert len(payload["counterfactual_families"]) == 36


def test_execution_routes_native_and_diagnostic_without_relabeling() -> None:
    planned = plan_suite_cases(expand_suite(SUITE))
    native = next(item for item in planned if item.execution_backend == "native_mujoco")
    diagnostic = next(
        item for item in planned if item.execution_backend == "diagnostic_quarantine"
    )

    class FakeNativeBackend:
        def __init__(self) -> None:
            self.calls = []

        def run(self, plan):
            self.calls.append(plan)
            return SimpleNamespace(
                simulation=SimpleNamespace(plan=plan),
                backend_provenance={"native_mujoco": True},
                transition_events=[{"timestamp": 0.1, "event_type": "test"}],
                as_renderer_payload=lambda: {
                    "videos": {"main": [object()], "secondary": [object()]},
                },
            )

    backend = FakeNativeBackend()
    result, rendered = cli._execute_suite_item(native, backend)
    assert backend.calls == [native.episode_plan]
    assert result.plan.episode_uuid == native.episode_uuid
    assert rendered["transition_rows"] == [{"timestamp": 0.1, "event_type": "test"}]

    result, rendered = cli._execute_suite_item(diagnostic, None)
    assert result.plan.episode_uuid == diagnostic.episode_uuid
    assert rendered["production_eligible"] is False
    assert rendered["backend_provenance"]["execution_backend"] == "diagnostic_quarantine"
    assert set(rendered["videos"]) == {
        "observation.images.main",
        "observation.images.secondary",
    }


def test_runtime_merge_retains_distinct_legacy_proxy_simulator_lineages(
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / ".suite_runtime"
    runtime_root.mkdir()
    config_hash = "c" * 64
    simulator_names = (
        "legacy_quarantine.scripted_bounce",
        "legacy_quarantine.legacy_motion_proxy",
        "legacy_quarantine.assisted_latch",
        "legacy_quarantine.equality_grasp",
    )
    for case_index, simulator_name in enumerate(simulator_names):
        provenance = {
            "source_generator": (
                "dynamic_robot_dataset.families.legacy_proxy_quarantine.adapter."
                "LegacyProxyQuarantineAdapter"
            ),
            "source_generator_version": "1.0.0",
            "generator_git_commit": "d" * 40,
            "config_hash": config_hash,
            "simulator_name": simulator_name,
            "simulator_version": "legacy-proxy-v1",
            "renderer": "dynamic_robot_dataset.diagnostic_state_renderer/v1",
            "execution_backend": "diagnostic_quarantine",
            "integrated_native_backend": False,
            "release_eligibility_is_per_episode": True,
            "asset_roots": {},
        }
        (runtime_root / f"case-{case_index:06d}.json").write_text(
            json.dumps(
                {
                    "schema_version": "dynamic-robot-suite-runtime-context/v1",
                    "config_hash": config_hash,
                    "cameras": [],
                    "provenance": [provenance],
                }
            ),
            encoding="utf-8",
        )

    cameras, provenance_rows = cli._merge_suite_runtime_rows(
        tmp_path,
        config_hash=config_hash,
    )

    assert cameras == {}
    assert len(provenance_rows) == len(simulator_names)
    assert {row["simulator_name"] for row in provenance_rows.values()} == set(simulator_names)
    assert all(key == cli._suite_provenance_key(row) for key, row in provenance_rows.items())


@pytest.mark.parametrize(
    ("family", "subfamily", "branch"),
    [
        ("cloth", "poke_cloth", "contact_failure"),
        ("rope", "drag_endpoint", "contact_failure"),
        ("cloth", "poke_cloth", "success_seeking"),
        ("legacy_proxy_quarantine", "scripted_bounce", "success_seeking"),
    ],
)
def test_diagnostic_candidates_write_honest_unverified_v2_records(
    family: str,
    subfamily: str,
    branch: str,
) -> None:
    item = next(
        value
        for value in plan_suite_cases(expand_suite(SUITE))
        if value.execution_backend == "diagnostic_quarantine"
        and value.case.family == family
        and value.case.subfamily == subfamily
        and value.episode_plan.intended_branch == branch
    )
    result, rendered = cli._execute_suite_item(item, None)

    record = cli._episode_record(result, item.case.case_index, "e" * 40, rendered)
    record.validate()

    assert record.actual_outcome == "unverified"
    assert record.actual_outcome_class.value == "unverified"
    assert record.task_success is False
    assert record.partial_success_score is None
    assert record.label_confidence is None
    assert record.label_status.value == "unverified"
    assert record.release_eligible is False
    expected_code = (
        "legacy_proxy_quarantined"
        if result.outcome.failure_mode == "legacy_proxy_quarantined"
        else "label_unverified"
    )
    assert record.failure_mode == record.primary_failure_code == expected_code
    candidate = record.objective_evidence["diagnostic_candidate_outcome"]
    assert candidate["actual_outcome"] == result.actual_outcome
    assert candidate["task_success"] == result.outcome.task_success
    assert candidate["failure_mode"] == result.outcome.failure_mode
    assert candidate["partial_success_score"] == result.outcome.partial_success_score
    assert record.objective_metrics == result.outcome.metrics
    assert record.objective_evidence["independently_recomputed"] is False
    expected_success, subject = _objective_metric_success_reference(record)
    assert expected_success is result.outcome.task_success
    assert subject == "diagnostic candidate task_success"
    smoke_projection = smoke_record(result, item.case.case_index, "e" * 40)
    assert (
        smoke_projection.actual_outcome,
        smoke_projection.task_success,
        smoke_projection.partial_success_score,
        smoke_projection.failure_mode,
        smoke_projection.label_confidence,
    ) == (
        record.actual_outcome,
        record.task_success,
        record.partial_success_score,
        record.failure_mode,
        record.label_confidence,
    )
    assert (
        smoke_projection.objective_evidence["diagnostic_candidate_outcome"]
        == record.objective_evidence["diagnostic_candidate_outcome"]
    )


def test_diagnostic_action_bundles_preserve_initial_state_and_vary_saved_actions() -> None:
    grouped: dict[str, list[EpisodeRecord]] = defaultdict(list)
    for item in plan_suite_cases(expand_suite(SUITE)):
        if item.case.family not in {"cloth", "rope"}:
            continue
        result, rendered = cli._execute_suite_item(item, None)
        grouped[item.episode_plan.counterfactual_bundle_id].append(
            cli._episode_record(result, item.case.case_index, "e" * 40, rendered)
        )

    assert sum(len(records) for records in grouped.values()) == 28
    for records in grouped.values():
        assert len({record.extras["derived_initial_state_hash"] for record in records}) == 1
        assert len({record.extras["derived_action_hash"] for record in records}) == len(records)


def test_diagnostic_cloth_assistance_intervals_match_frame_mechanism_ids() -> None:
    for item in plan_suite_cases(expand_suite(SUITE)):
        if item.case.family != "cloth":
            continue
        result, _ = cli._execute_suite_item(item, None)
        for rows in (
            cli._default_frame_rows(result, item.case.case_index),
            smoke_frame_rows(result, item.case.case_index),
        ):
            for row in rows:
                mechanism_active = bool(row["assistance.mechanism_ids"])
                assert mechanism_active == bool(row["assistance.active"])


def test_expected_declarations_use_persisted_hash_contract_and_keep_missing_members() -> None:
    cases = [
        case
        for case in expand_suite(SUITE)
        if case.family == "falling_catch" and case.subfamily == "centered_vertical_drop"
    ]
    planned = plan_suite_cases(cases)
    expected = build_planned_counterfactual_family_records(planned)
    assert len(expected) == 1 and expected[0].relation == CounterfactualRelation.ACTION
    records = [_record(item, index) for index, item in enumerate(planned)]

    rows = cli._expected_counterfactual_rows(expected, records)
    declaration = CounterfactualFamilyRecord.from_dict(rows[0])
    assert declaration.expected_episode_uuids == sorted(item.episode_uuid for item in planned)
    derived = {
        record.episode_uuid: {
            "derived_initial_state_hash": record.extras["derived_initial_state_hash"],
            "derived_action_hash": record.extras["derived_action_hash"],
        }
        for record in records
    }
    assert validate_counterfactual_family_records(
        [declaration], records, derived_by_uuid=derived
    ) == []

    incomplete_rows = cli._expected_counterfactual_rows(expected, records[:-1])
    incomplete = CounterfactualFamilyRecord.from_dict(incomplete_rows[0])
    assert incomplete.expected_member_count == 4
    problems = validate_counterfactual_family_records(
        [incomplete],
        records[:-1],
        derived_by_uuid={key: value for key, value in derived.items() if key != records[-1].episode_uuid},
    )
    assert any("missing expected members" in problem for problem in problems)


def test_planned_hash_copy_cannot_hide_tampered_runtime_metadata() -> None:
    cases = [
        case
        for case in expand_suite(SUITE)
        if case.family == "falling_catch"
        and case.subfamily == "centered_vertical_drop"
    ]
    planned = plan_suite_cases(cases)
    declaration = build_planned_counterfactual_family_records(planned)[0]
    records = [_record(item, index) for index, item in enumerate(planned)]
    derived = {
        record.episode_uuid: {
            "derived_initial_state_hash": record.extras["derived_initial_state_hash"],
            "derived_action_hash": record.extras["derived_action_hash"],
        }
        for record in records
    }
    assert validate_counterfactual_family_records(
        [declaration], records, derived_by_uuid=derived
    ) == []

    # The copied planned hash map is unchanged.  Validation must still hash
    # actual committed metadata and reject the modified sibling.
    records[0].scene_seed += 1
    records[0].randomization["scene_style"] = "forged_after_planning"
    mass = records[0].physics.parameters["mass"]
    records[0].physics.parameters["mass"] = replace(
        mass, value=float(mass.value) * 2.0
    )
    problems = validate_counterfactual_family_records(
        [declaration], records, derived_by_uuid=derived
    )
    assert any("changes runtime field scene_seed" in problem for problem in problems)
    assert any("changes invariant appearance" in problem for problem in problems)
    assert any("changes invariant physics" in problem for problem in problems)


def test_physics_sweeps_ignore_derived_solver_hash_and_id_ood_partition() -> None:
    cases = [
        case
        for case in expand_suite(SUITE)
        if case.family == "projectile_rebound"
        and case.subfamily == "gravity_sweep"
    ]
    planned = plan_suite_cases(cases)
    declaration = next(
        value
        for value in build_planned_counterfactual_family_records(planned)
        if value.relation == CounterfactualRelation.PHYSICS
    )
    records = [_record(item, index) for index, item in enumerate(planned)]
    for index, record in enumerate(records):
        record.physics.solver_settings = {"model_hash": f"model-{index}"}
        record.physics.parameter_range_provenance["partition"] = (
            "test_ood" if index in {0, len(records) - 1} else "train_id"
        )
    derived = {
        record.episode_uuid: {
            "derived_initial_state_hash": record.extras["derived_initial_state_hash"],
            "derived_action_hash": record.extras["derived_action_hash"],
        }
        for record in records
    }

    assert validate_counterfactual_family_records(
        [declaration], records, derived_by_uuid=derived
    ) == []

    records[0].physics.solver_settings["contact_resolution"] = "tampered"
    problems = validate_counterfactual_family_records(
        [declaration], records, derived_by_uuid=derived
    )
    assert any("changes invariant nonintervened_physics" in problem for problem in problems)


def test_acceptance_gates_require_exact_160_views_styles_categories_and_rigid_outcomes() -> None:
    planned = plan_suite_cases(expand_suite(SUITE))
    measured_class_for_branch = {
        "success_seeking": "success",
        "near_miss": "near_miss",
        "contact_failure": "contact_failure",
        "no_op": "no_op",
        "wrong_action": "wrong_action",
    }
    records = [
        _record(
            item,
            index,
            actual_outcome_class=(
                measured_class_for_branch[item.case.branch]
                if item.case.family
                in {"falling_catch", "rolling_interception", "projectile_rebound"}
                else "success"
            ),
        )
        for index, item in enumerate(planned)
    ]
    required = ("falling_catch", "rolling_interception", "projectile_rebound")
    gates = cli._suite_acceptance_gates(
        planned,
        records,
        required_rigid_outcome_families=required,
        required_physics_sweeps=_sweep_requirements(),
    )
    # The rigid portion satisfies structural/outcome gates, but the current
    # 32 diagnostic-quarantine cases deliberately prevent a full-native pass.
    assert not gates["passed"]
    assert not gates["checks"]["all_acceptance_cases_use_native_mujoco"]
    assert gates["checks"]["all_native_rigid_physics_qc_pass"]
    assert gates["checks"]["all_required_physics_sweeps_measured_and_monotonic"]
    assert gates["blockers"] == ["all_acceptance_cases_use_native_mujoco"]
    assert gates["planned_category_counts"] == gates["committed_category_counts"]
    assert not gates["missing_required_outcome_families"]
    assert not gates["missing_required_actual_outcome_classes"]

    missing = cli._suite_acceptance_gates(
        planned,
        records[:-1],
        required_rigid_outcome_families=required,
        required_physics_sweeps=_sweep_requirements(),
    )
    assert not missing["passed"]
    assert "exact_160_committed" in missing["blockers"]

    all_success = [_record(item, index) for index, item in enumerate(planned)]
    no_failures = cli._suite_acceptance_gates(
        planned,
        all_success,
        required_rigid_outcome_families=required,
        required_physics_sweeps=_sweep_requirements(),
    )
    assert not no_failures["passed"]
    assert set(no_failures["missing_required_outcome_families"]) == set(required)
    assert no_failures["missing_required_actual_outcome_classes"] == {
        "falling_catch": ["contact_failure", "near_miss", "no_op"],
        "rolling_interception": ["contact_failure", "near_miss", "no_op"],
        "projectile_rebound": [
            "contact_failure",
            "near_miss",
            "no_op",
            "wrong_action",
        ],
    }


@pytest.mark.parametrize("corruption", ["zero_samples", "constant", "inverted", "missing"])
def test_sweep_acceptance_rejects_nonmeasured_or_nonresponsive_families(
    corruption: str,
) -> None:
    planned = plan_suite_cases(expand_suite(SUITE))
    records = [_record(item, index) for index, item in enumerate(planned)]
    if corruption == "zero_samples":
        member = next(
            record for record in records if record.subfamily == "gravity_sweep"
        )
        member.extras["physics_qc"]["checks"]["free_flight_pair_count"] = 0
    elif corruption == "constant":
        for record in records:
            if record.subfamily == "friction_sweep":
                record.extras["physics_qc"]["checks"][
                    "measured_effective_dynamic_friction"
                ] = 0.3
    elif corruption == "inverted":
        restitution = [
            record for record in records if record.subfamily == "restitution_sweep"
        ]
        measured = [
            record.extras["physics_qc"]["checks"]["measured_effective_restitution"]
            for record in restitution
        ]
        for record, value in zip(restitution, reversed(measured)):
            record.extras["physics_qc"]["checks"][
                "measured_effective_restitution"
            ] = value
    else:
        missing = next(
            record for record in records if record.subfamily == "restitution_sweep"
        )
        records.remove(missing)
    evidence = cli._sweep_acceptance_evidence(
        planned,
        records,
        _sweep_requirements(),
    )
    assert not evidence["passed"]


def test_suite_acceptance_rejects_false_native_physics_qc() -> None:
    planned = plan_suite_cases(expand_suite(SUITE))
    records = [_record(item, index) for index, item in enumerate(planned)]
    native = next(
        record
        for record, item in zip(records, planned)
        if item.execution_backend == "native_mujoco"
    )
    native.physics_qc_pass = False
    gates = cli._suite_acceptance_gates(
        planned,
        records,
        required_rigid_outcome_families=(),
        required_physics_sweeps=_sweep_requirements(),
    )
    assert not gates["checks"]["all_native_rigid_physics_qc_pass"]


def test_failed_suite_persists_preplan_expected_context_and_never_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cases = [
        case
        for case in expand_suite(SUITE)
        if case.family == "falling_catch" and case.subfamily == "centered_vertical_drop"
    ]
    output = tmp_path / "failed-suite"

    class FakeWriter:
        def __init__(self, dataset_root, config, *, resume=False, **_kwargs):
            self.root = Path(dataset_root)
            self.root.mkdir(parents=True, exist_ok=True)
            (self.root / ".records").mkdir(exist_ok=True)
            self.config_hash = sha256_json(config)
            self.resume = resume

        def records(self):
            return []

        def write_episode(self, *_args, **_kwargs):  # pragma: no cover - all injected failures
            raise AssertionError("writer must not be reached")

    monkeypatch.setattr(cli, "expand_suite", lambda _path: cases)
    monkeypatch.setattr(cli, "EpisodeWriter", FakeWriter)
    monkeypatch.setattr(cli, "get_git_commit", lambda _path: "test-commit")
    monkeypatch.setattr(backend_registry, "get_backend", lambda _name: object())
    attempts = []

    def injected_failure(item, _backend):
        assert (output / ".suite_plan.json").is_file()
        attempts.append(item.episode_uuid)
        raise RuntimeError("injected native execution failure")

    monkeypatch.setattr(cli, "_execute_suite_item", injected_failure)
    assert cli._command_generate_suite(_arguments(output)) == 1
    first_report = json.loads(capsys.readouterr().out)
    assert len(attempts) == 4
    assert first_report["failed_case_count"] == 4
    # This deliberately reduced failure fixture contains only rigid cases, so
    # it is fully native even though execution itself failed.
    assert first_report["full_native"] is True
    assert first_report["outcome_mismatch_retry_count"] == 0

    plan_ledger = json.loads((output / ".suite_plan.json").read_text())
    assert plan_ledger["retry_policy"]["retry_on_outcome_mismatch"] is False
    assert len(plan_ledger["counterfactual_families"]) == 1
    context = json.loads((output / ".finalize_context.json").read_text())
    declaration = CounterfactualFamilyRecord.from_dict(
        context["counterfactual_families"][0]
    )
    assert declaration.expected_member_count == 4
    assert declaration.expected_episode_uuids == sorted(item.episode_uuid for item in plan_suite_cases(cases))
    assert len(list((output / ".suite_attempts").glob("case-*.json"))) == 4

    monkeypatch.setattr(
        cli,
        "_execute_suite_item",
        lambda *_args: (_ for _ in ()).throw(AssertionError("failed cases were retried")),
    )
    assert cli._command_generate_suite(_arguments(output, resume=True)) == 1
    resumed_report = json.loads(capsys.readouterr().out)
    assert resumed_report == first_report
