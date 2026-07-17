from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from dynamic_robot_dataset.common.review_suite import write_review_suite_bundle
from dynamic_robot_dataset.common.source_preview import (
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
