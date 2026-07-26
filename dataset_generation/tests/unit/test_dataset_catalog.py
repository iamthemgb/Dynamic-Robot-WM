from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _catalog_module():
    script = Path(__file__).resolve().parents[2] / "tools" / "catalog_datasets.py"
    spec = importlib.util.spec_from_file_location("dataset_catalog_tool", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_catalog_never_releases_without_completed_human_review(tmp_path: Path) -> None:
    module = _catalog_module()
    run = tmp_path / "sealed-run"
    _write_json(run / ".run_plan.json", {"episodes": ["episode-0"]})
    _write_json(run / ".seal.json", {"sealed": True})
    _write_json(run / "meta" / ".complete.json", {"complete": True})
    _write_json(
        run / "qc" / "dataset_report.json",
        {
            "passed": True,
            "episode_count": 1,
            "failed_episode_count": 0,
            "release_eligible_count": 1,
        },
    )
    _write_json(run / "reviews" / "human_review_ledger.pending.json", {})

    pending = module._canonical_run(run)
    assert pending["human_review"] == "pending"
    assert pending["release_eligible"] is False

    (run / "reviews" / "human_review_ledger.pending.json").unlink()
    missing = module._canonical_run(run)
    assert missing["human_review"] == "missing"
    assert missing["release_eligible"] is False
