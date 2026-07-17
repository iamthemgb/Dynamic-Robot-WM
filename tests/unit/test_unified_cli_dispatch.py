from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from dynamic_robot_dataset import cli, orchestration_cli, review_suite_cli


class _PreviewResult:
    def __init__(self, *, passed: bool = True) -> None:
        self.strict_qc_passed = passed

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_root": "/canonical/preview",
            "strict_qc_passed": self.strict_qc_passed,
            "case_ids": ["F1a-review-00"],
        }


def test_run_shard_defaults_to_owned_executor_without_import_spec(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sentinel = object()
    captured: dict[str, object] = {}

    monkeypatch.setattr(orchestration_cli, "_owned_shard_executor", lambda root: sentinel)

    def fake_run_shard(dataset, shard_id, executor, *, max_episodes=None):
        captured.update(
            dataset=dataset,
            shard_id=shard_id,
            executor=executor,
            max_episodes=max_episodes,
        )
        return SimpleNamespace(
            failed_count=0,
            to_dict=lambda: {"shard_id": shard_id, "failed_count": 0},
        )

    monkeypatch.setattr(orchestration_cli, "run_shard", fake_run_shard)
    dataset = tmp_path / "dataset"
    assert cli.main(
        [
            "run-shard",
            "--dataset",
            str(dataset),
            "--shard-id",
            "2",
            "--max-episodes",
            "3",
        ]
    ) == 0
    assert captured == {
        "dataset": str(dataset),
        "shard_id": 2,
        "executor": sentinel,
        "max_episodes": 3,
    }
    assert json.loads(capsys.readouterr().out)["failed_count"] == 0


def test_arbitrary_shard_executor_requires_explicit_unsafe_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = argparse.Namespace(
        dataset="unused",
        executor="example.module:execute",
        unsafe_executor=False,
    )
    with pytest.raises(ValueError, match="--unsafe-executor"):
        orchestration_cli._resolve_shard_executor(arguments)

    sentinel = object()
    monkeypatch.setattr(orchestration_cli, "_load_executor", lambda value: sentinel)
    arguments.unsafe_executor = True
    assert orchestration_cli._resolve_shard_executor(arguments) is sentinel

    arguments.executor = None
    with pytest.raises(ValueError, match="requires --executor"):
        orchestration_cli._resolve_shard_executor(arguments)


def test_owned_executor_is_bound_to_source_backend_and_plan_declarations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    (root / ".generation.json").write_text(
        json.dumps({"resolved_config": {"backend": "source_mujoco"}}),
        encoding="utf-8",
    )
    plan = SimpleNamespace(
        episodes=(SimpleNamespace(declaration={"backend": "source_mujoco"}),)
    )
    monkeypatch.setattr(orchestration_cli, "load_run_plan", lambda _root: plan)

    from dynamic_robot_dataset.common.source_execution import (
        execute_source_mujoco_episode,
    )

    assert orchestration_cli._owned_shard_executor(root) is execute_source_mujoco_episode

    (root / ".generation.json").write_text(
        json.dumps({"resolved_config": {"backend": "diagnostic"}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="No owned run-shard executor"):
        orchestration_cli._owned_shard_executor(root)


def test_review_suite_execute_forwards_fixed_selectors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, object] = {}

    def fake_execute(review_root, dataset_root, **options):
        captured.update(
            review_root=Path(review_root),
            dataset_root=dataset_root,
            options=options,
        )
        return _PreviewResult()

    monkeypatch.setattr(review_suite_cli, "_execute_source_preview", fake_execute)
    suite = tmp_path / "review-suite"
    dataset = tmp_path / "preview-dataset"
    assert cli.main(
        [
            "review-suite",
            "--output",
            str(suite),
            "--execute",
            "--dataset-output",
            str(dataset),
            "--case-id",
            "F1a-review-00",
            "--case-id",
            "P0a-review-00",
            "--leaf-id",
            "F1a",
            "--clean-r0-only",
            "--resume",
        ]
    ) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["execution"]["strict_qc_passed"] is True
    assert captured == {
        "review_root": suite.resolve(),
        "dataset_root": str(dataset),
        "options": {
            "case_ids": ("F1a-review-00", "P0a-review-00"),
            "leaf_ids": ("F1a",),
            "clean_r0_only": True,
            "resume": True,
        },
    }


def test_source_generate_needs_no_family_and_uses_canonical_preview(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, object] = {}

    def fake_execute(review_root, dataset_root, **options):
        captured.update(
            review_root=review_root,
            dataset_root=dataset_root,
            options=options,
        )
        return _PreviewResult()

    monkeypatch.setattr(cli, "_execute_source_preview", fake_execute)
    suite = tmp_path / "review-suite"
    dataset = tmp_path / "preview-dataset"
    assert cli.main(
        [
            "generate",
            "--backend",
            "source_mujoco",
            "--review-suite-root",
            str(suite),
            "--review-case",
            "F1a-review-00",
            "--review-leaf",
            "F1a",
            "--clean-r0-only",
            "--output",
            str(dataset),
            "--resume",
        ]
    ) == 0
    assert json.loads(capsys.readouterr().out)["strict_qc_passed"] is True
    assert captured == {
        "review_root": str(suite),
        "dataset_root": str(dataset),
        "options": {
            "case_ids": ("F1a-review-00",),
            "leaf_ids": ("F1a",),
            "clean_r0_only": True,
            "resume": True,
        },
    }


def test_source_generate_requires_review_suite_root(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["generate", "--backend", "source_mujoco", "--dry-run"]) == 2
    assert "--review-suite-root" in capsys.readouterr().err

