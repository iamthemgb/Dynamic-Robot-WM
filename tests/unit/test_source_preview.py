from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from dynamic_robot_dataset.common.review_suite import write_review_suite_bundle
from dynamic_robot_dataset.common.source_preview import (
    _prepare_source_review_declarations_isolated,
    _run_preview_shards_isolated,
    select_source_review_cases,
)


def test_source_preview_selection_preserves_fixed_order_and_filters_clean_r0(
    tmp_path,
) -> None:
    bundle = write_review_suite_bundle(tmp_path / "suite")

    selected = select_source_review_cases(
        bundle,
        leaf_ids=("F1d", "P0a"),
        clean_r0_only=True,
    )

    assert [case.case_id for case in selected] == [
        "P0a-review-00",
        "F1d-review-00",
    ]
    assert all(case.execution_eligible for case in selected)


def test_source_preview_selection_rejects_unknown_and_blocked_cases(tmp_path) -> None:
    bundle = write_review_suite_bundle(tmp_path / "suite")

    with pytest.raises(ValueError, match="unknown fixed review selectors"):
        select_source_review_cases(bundle, case_ids=("not-a-fixed-case",))
    with pytest.raises(ValueError, match="blocked review cases"):
        select_source_review_cases(bundle, case_ids=("F3c-review-00",))


def test_source_preview_executes_one_owned_isolated_process_per_shard(
    tmp_path, monkeypatch
) -> None:
    commands = []

    def fake_run(command, **options):
        commands.append((tuple(command), options))
        return SimpleNamespace(
            returncode=0,
            stderr="",
            stdout=json.dumps(
                {
                    "planned_count": 1,
                    "committed_count": 1,
                    "recovered_count": 0,
                    "failed_count": 0,
                    "remaining_count": 0,
                }
            ),
        )

    monkeypatch.setattr(
        "dynamic_robot_dataset.common.source_preview.subprocess.run", fake_run
    )
    _run_preview_shards_isolated(tmp_path, 3)

    assert len(commands) == 3
    assert [command[-1] for command, _ in commands] == ["0", "1", "2"]
    assert all("--unsafe-executor" not in command for command, _ in commands)
    assert all(options["capture_output"] is True for _, options in commands)


def test_source_preview_isolated_worker_fails_closed(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "dynamic_robot_dataset.common.source_preview.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stderr="",
            stdout=json.dumps(
                {
                    "planned_count": 1,
                    "committed_count": 0,
                    "recovered_count": 0,
                    "failed_count": 0,
                    "remaining_count": 1,
                }
            ),
        ),
    )
    with pytest.raises(RuntimeError, match="incomplete isolated shard"):
        _run_preview_shards_isolated(tmp_path, 1)


def test_source_preview_prepares_one_declaration_per_isolated_process(
    tmp_path, monkeypatch
) -> None:
    bundle = write_review_suite_bundle(tmp_path / "suite")
    cases = select_source_review_cases(bundle, leaf_ids=("F1a",))[:3]
    commands = []

    def fake_run(command, **options):
        request = json.loads(options["input"])
        case = request["review_case"]
        commands.append((tuple(command), options, request))
        return SimpleNamespace(
            returncode=0,
            stderr="",
            stdout=json.dumps(
                {
                    "episode_index": request["episode_index"],
                    "episode_uuid": case["episode_uuid"],
                    "review_suite_episode_index": case["episode_index"],
                    "review_case": case,
                    "review_case_sha256": case["case_sha256"],
                    "generator_git_commit": request["generator_git_commit"],
                    "source_scenario_spec": {"scenario_id": case["case_id"]},
                    "source_scenario_spec_sha256": f"spec-{case['case_id']}",
                }
            ),
        )

    monkeypatch.setattr(
        "dynamic_robot_dataset.common.source_preview.subprocess.run", fake_run
    )
    monkeypatch.setattr(
        "dynamic_robot_dataset.common.source_preview.SourceScenarioSpec.from_dict",
        lambda value: SimpleNamespace(
            scenario_id=value["scenario_id"],
            spec_hash=f"spec-{value['scenario_id']}",
        ),
    )

    declarations = _prepare_source_review_declarations_isolated(
        cases,
        generator_git_commit="a" * 40,
    )

    assert len(declarations) == len(cases) == len(commands)
    assert [value["episode_index"] for value in declarations] == [0, 1, 2]
    for index, (command, options, request) in enumerate(commands):
        assert command[-2:] == (
            "-m",
            "dynamic_robot_dataset.common.source_preview_prepare_worker",
        )
        assert options["capture_output"] is True
        assert options["check"] is False
        assert request["episode_index"] == index
        assert request["review_case"] == cases[index].to_dict()
        assert request["generator_git_commit"] == "a" * 40


@pytest.mark.parametrize(
    ("completed", "failure"),
    (
        (
            SimpleNamespace(returncode=7, stderr="worker failed", stdout=""),
            "preparation failed",
        ),
        (
            SimpleNamespace(returncode=0, stderr="", stdout="not-json"),
            "malformed JSON",
        ),
        (
            SimpleNamespace(returncode=0, stderr="", stdout="[]"),
            "non-mapping",
        ),
    ),
)
def test_source_preview_isolated_declaration_worker_fails_closed(
    tmp_path, monkeypatch, completed, failure
) -> None:
    bundle = write_review_suite_bundle(tmp_path / "suite")
    case = select_source_review_cases(bundle, case_ids=("F1a-review-00",))
    monkeypatch.setattr(
        "dynamic_robot_dataset.common.source_preview.subprocess.run",
        lambda *args, **kwargs: completed,
    )

    with pytest.raises(RuntimeError, match=failure):
        _prepare_source_review_declarations_isolated(
            case,
            generator_git_commit="a" * 40,
        )


def test_source_preview_isolated_declaration_rejects_identity_change(
    tmp_path, monkeypatch
) -> None:
    bundle = write_review_suite_bundle(tmp_path / "suite")
    case = select_source_review_cases(bundle, case_ids=("F1a-review-00",))[0]
    declaration = {
        "episode_index": 1,
        "episode_uuid": case.episode_uuid,
        "review_suite_episode_index": case.episode_index,
        "review_case": case.to_dict(),
        "review_case_sha256": case.case_sha256,
        "generator_git_commit": "a" * 40,
        "source_scenario_spec": {"scenario_id": case.case_id},
        "source_scenario_spec_sha256": "spec",
    }
    monkeypatch.setattr(
        "dynamic_robot_dataset.common.source_preview.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stderr="",
            stdout=json.dumps(declaration),
        ),
    )

    with pytest.raises(RuntimeError, match="identity differs"):
        _prepare_source_review_declarations_isolated(
            (case,),
            generator_git_commit="a" * 40,
        )
