#!/usr/bin/env python3
"""Build a non-destructive catalog of robotics datasets and run artifacts.

The catalog is deliberately an index, not a migration tool.  It never moves,
renames, unlinks, or deletes a source artifact.  With ``--build-symlink-view``
it creates a content-addressed directory of symlinks, so later refreshes leave
older views intact rather than mutating them in place.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "dynamic-robot-dataset-catalog/v1"
DEFAULT_CATALOG_ROOT = Path("/gpfs/radev/project/sous/zl664/dataset_catalog")
TSV_COLUMNS = (
    "artifact_id",
    "path",
    "category",
    "kind",
    "lifecycle",
    "schema_version",
    "episodes",
    "videos",
    "size_bytes",
    "automated_qc",
    "human_review",
    "release_tier",
    "release_eligible",
    "seal_sha256",
    "plan_sha256",
    "source_commit",
    "duplicate_group",
    "notes",
)


def _json(path: Path) -> dict[str, Any] | list[Any] | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_stats(root: Path) -> tuple[int, int, int, int]:
    """Return size, file count, MP4 count, and episode-parquet count."""
    if root.is_file():
        size = root.stat().st_size
        return size, 1, int(root.suffix.lower() == ".mp4"), 0

    size_bytes = 0
    file_count = 0
    video_count = 0
    episode_parquets = 0
    for directory, _, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        directory_parts = directory_path.relative_to(root).parts
        for filename in filenames:
            path = directory_path / filename
            try:
                metadata = os.stat(path, follow_symlinks=False)
                if stat.S_ISLNK(metadata.st_mode):
                    continue
                size_bytes += metadata.st_size
            except OSError:
                continue
            file_count += 1
            suffix = path.suffix.lower()
            video_count += int(suffix == ".mp4")
            if suffix == ".parquet" and "data" in directory_parts:
                episode_parquets += 1
    return size_bytes, file_count, video_count, episode_parquets


def _git_commit(path: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip()


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-.")
    return slug or "artifact"


def _record(
    *,
    artifact_id: str,
    path: Path,
    category: str,
    kind: str,
    lifecycle: str,
    notes: str = "",
    scan_mode: str = "recursive",
) -> dict[str, Any]:
    if scan_mode == "recursive":
        size_bytes, file_count, videos, episode_parquets = _tree_stats(path)
    elif scan_mode == "metadata_only":
        size_bytes, file_count, videos, episode_parquets = None, None, 0, 0
        notes = "; ".join(
            filter(None, (notes, "recursive dependency size intentionally not scanned"))
        )
    else:
        raise ValueError(f"unsupported scan_mode={scan_mode!r} for {path}")
    return {
        "artifact_id": _slug(artifact_id),
        "path": str(path.resolve()),
        "category": category,
        "kind": kind,
        "lifecycle": lifecycle,
        "schema_version": "",
        "episodes": episode_parquets or 0,
        "videos": videos,
        "size_bytes": size_bytes,
        "file_count": file_count,
        "automated_qc": "not_run",
        "human_review": "not_applicable",
        "release_tier": "",
        "release_eligible": None,
        "seal_sha256": "",
        "plan_sha256": "",
        "source_commit": "",
        "duplicate_group": "",
        "notes": notes,
        "modified_at": datetime.fromtimestamp(
            path.stat().st_mtime, tz=timezone.utc
        ).isoformat(),
    }


def _first_record_metadata(run: Path) -> tuple[str, str, bool | None]:
    records_dir = run / ".records"
    if not records_dir.is_dir():
        return "", "", None
    release_tiers: set[str] = set()
    commits: set[str] = set()
    release_values: set[bool] = set()
    # A run is normally homogeneous for source commit and release tier.  Bound
    # reads because attempt records can contain large high-rate diagnostics.
    for path in sorted(records_dir.glob("*.json"))[:32]:
        payload = _json(path)
        if not isinstance(payload, dict):
            continue
        episode = payload.get("episode", {})
        if not isinstance(episode, dict):
            continue
        tier = episode.get("release_tier")
        commit = episode.get("generator_git_commit")
        eligible = episode.get("release_eligible")
        if isinstance(tier, str) and tier:
            release_tiers.add(tier)
        if isinstance(commit, str) and commit:
            commits.add(commit)
        if isinstance(eligible, bool):
            release_values.add(eligible)
    release_eligible: bool | None
    if release_values == {True}:
        release_eligible = True
    elif release_values:
        release_eligible = False
    else:
        release_eligible = None
    return ",".join(sorted(release_tiers)), ",".join(sorted(commits)), release_eligible


def _canonical_run(root: Path) -> dict[str, Any]:
    name = root.name
    run_plan = root / ".run_plan.json"
    seal = root / ".seal.json"
    complete = root / "meta" / ".complete.json"
    report_path = root / "qc" / "dataset_report.json"
    human_pending = root / "reviews" / "human_review_ledger.pending.json"
    is_derived_qc = (
        name.endswith("_qc_external")
        or name.endswith("_qc_recheck")
        or ((root / "dataset_report.json").is_file() and not run_plan.is_file())
    )
    is_plan = (root / "review_suite_plan.json").is_file() and not run_plan.is_file()

    if is_plan:
        record = _record(
            artifact_id=name,
            path=root,
            category="plans/history",
            kind="review_suite_plan",
            lifecycle="planned",
        )
        plan_path = root / "review_suite_plan.json"
        payload = _json(plan_path)
        if isinstance(payload, dict):
            episodes = payload.get("episodes", payload.get("cases", []))
            if isinstance(episodes, list):
                record["episodes"] = len(episodes)
            record["schema_version"] = str(payload.get("schema_version", ""))
        record["plan_sha256"] = _sha256(plan_path)
        record["_plan_only"] = True
        return record

    if is_derived_qc:
        record = _record(
            artifact_id=name,
            path=root,
            category="runs/derived_qc",
            kind="derived_qc",
            lifecycle="derived",
        )
        report = _json(root / "dataset_report.json")
        if isinstance(report, dict):
            record["episodes"] = int(report.get("episode_count", 0) or 0)
            record["schema_version"] = str(report.get("schema_version", ""))
            record["automated_qc"] = "pass" if report.get("passed") else "fail"
        return record

    sealed = seal.is_file() and complete.is_file()
    if sealed:
        lifecycle = "sealed"
        report = _json(report_path)
        failed = None
        passed = None
        if isinstance(report, dict):
            failed = int(report.get("failed_episode_count", 0) or 0)
            passed = bool(report.get("passed", failed == 0))
        if failed is not None and (failed > 0 or passed is False):
            category = "runs/sealed_qc_fail"
        elif human_pending.is_file():
            category = "runs/sealed_qc_pass_human_pending"
        elif name.startswith("preview_"):
            category = "previews/canonical_sealed"
        else:
            category = "runs/sealed_qc_pass_human_missing"
    else:
        lifecycle = "partial"
        category = "runs/partial"

    record = _record(
        artifact_id=name,
        path=root,
        category=category,
        kind="canonical_run",
        lifecycle=lifecycle,
    )
    if run_plan.is_file():
        record["plan_sha256"] = _sha256(run_plan)
    if seal.is_file():
        record["seal_sha256"] = _sha256(seal)
    report = _json(report_path)
    if isinstance(report, dict):
        record["episodes"] = int(report.get("episode_count", record["episodes"]) or 0)
        record["schema_version"] = str(report.get("schema_version", ""))
        failed = int(report.get("failed_episode_count", 0) or 0)
        record["automated_qc"] = "pass" if report.get("passed", failed == 0) else "fail"
        eligible_count = int(report.get("release_eligible_count", 0) or 0)
        record["release_eligible"] = bool(
            record["episodes"] and eligible_count == record["episodes"]
        )
        if failed:
            record["notes"] = f"{failed}/{record['episodes']} automated-QC failures"
    elif run_plan.is_file():
        plan = _json(run_plan)
        committed_records = len(list((root / ".records").glob("*.json")))
        record["episodes"] = committed_records
        if isinstance(plan, dict):
            planned = plan.get("episodes", plan.get("episode_membership", []))
            if isinstance(planned, list):
                record["notes"] = (
                    f"partial: {record['episodes']}/{len(planned)} planned episodes present"
                )
    record["human_review"] = "pending" if human_pending.is_file() else "missing"
    release_tier, source_commit, record_eligible = _first_record_metadata(root)
    record["release_tier"] = release_tier
    record["source_commit"] = source_commit
    if record["release_eligible"] is None:
        record["release_eligible"] = record_eligible
    # Automated QC and sealed metadata cannot authorize training while the
    # required human review is pending or absent.  The catalog is intentionally
    # fail-closed even if an older dataset report claimed release eligibility.
    if record["human_review"] != "approved":
        record["release_eligible"] = False
    record["_canonical_media"] = True
    return record


def _external_metadata(record: dict[str, Any], path: Path) -> None:
    info = _json(path / "meta" / "info.json")
    summary = _json(path / "meta" / "dataset_summary.json")
    complete = _json(path / "meta" / ".complete.json")
    target = _json(path / "_TARGET.json")
    report = _json(path / "qc" / "dataset_report.json")
    if not isinstance(report, dict):
        report = _json(path / "dataset_report.json")

    for payload in (info, summary, complete, target):
        if not isinstance(payload, dict):
            continue
        for key in ("total_episodes", "episode_count"):
            if payload.get(key) is not None:
                record["episodes"] = int(payload[key])
                break
        if payload.get("schema_version") and not record["schema_version"]:
            record["schema_version"] = str(payload["schema_version"])

    if isinstance(info, dict):
        extras = info.get("extras", {})
        if isinstance(extras, dict):
            record["release_tier"] = str(extras.get("release_tier_default", ""))
    if isinstance(summary, dict) and not record["release_tier"]:
        record["release_tier"] = str(summary.get("release_tier_default", ""))
    if isinstance(report, dict):
        if report.get("episode_count") is not None:
            record["episodes"] = int(report["episode_count"])
        failed = int(report.get("failed_episode_count", 0) or 0)
        record["automated_qc"] = "pass" if report.get("passed", failed == 0) else "fail"
        eligible_count = int(report.get("release_eligible_count", 0) or 0)
        record["release_eligible"] = bool(
            record["episodes"] and eligible_count == record["episodes"]
        )

    if isinstance(target, dict) and isinstance(complete, dict):
        if target.get("complete") is False:
            suffix = "_TARGET.json says complete=false while meta/.complete.json exists"
            record["notes"] = "; ".join(filter(None, (record["notes"], suffix)))


def _apply_rule(entry: dict[str, Any], name: str) -> dict[str, Any]:
    values = {
        "category": entry.get("category", "legacy/assisted"),
        "kind": entry.get("kind", "external_artifact"),
        "lifecycle": entry.get("lifecycle", "unsealed"),
        "notes": entry.get("notes", ""),
        "duplicate_group": "",
    }
    for rule in entry.get("rules", []):
        if fnmatch.fnmatch(name, str(rule.get("pattern", ""))):
            for key in values:
                if key in rule:
                    values[key] = rule[key]
    hints = entry.get("duplicate_group_hints", {})
    if isinstance(hints, dict) and name in hints:
        values["duplicate_group"] = hints[name]
    return values


def _external_self(entry: dict[str, Any], path: Path) -> dict[str, Any]:
    values = _apply_rule(entry, path.name)
    artifact_id = str(entry.get("artifact_id") or f"{entry['id_prefix']}-{path.name}")
    record = _record(
        artifact_id=artifact_id,
        path=path,
        category=str(values["category"]),
        kind=str(values["kind"]),
        lifecycle=str(values["lifecycle"]),
        notes=str(values["notes"]),
        scan_mode=str(entry.get("scan_mode", "recursive")),
    )
    record["duplicate_group"] = str(values["duplicate_group"])
    _external_metadata(record, path)
    if not record["release_tier"] and entry.get("release_tier"):
        record["release_tier"] = str(entry["release_tier"])
    if record["category"] == "legacy/assisted":
        record["release_eligible"] = False
    if record["kind"] == "source_dependency":
        record["source_commit"] = _git_commit(path)
    return record


def _scan_external(entry: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    root = Path(str(entry["path"]))
    if not root.exists():
        return [], [f"missing external root: {root}"]
    discovery = str(entry.get("discovery", "self"))
    paths: list[Path]
    if discovery == "self":
        paths = [root]
    elif discovery == "children":
        paths = sorted(path for path in root.iterdir() if path.is_dir())
    elif discovery == "corpus_leaves":
        paths = sorted(
            leaf
            for family in root.iterdir()
            if family.is_dir() and not family.name.startswith("_")
            for leaf in family.iterdir()
            if leaf.is_dir()
        )
    else:
        return [], [f"unsupported discovery={discovery!r} for {root}"]

    records: list[dict[str, Any]] = []
    for path in paths:
        local_entry = dict(entry)
        if discovery == "corpus_leaves":
            if (path / "meta" / ".complete.json").is_file():
                local_entry.update(
                    category="legacy/assisted",
                    kind="converted_legacy_dataset",
                    lifecycle="legacy_complete",
                )
            elif (path / "_TARGET.json").is_file():
                local_entry.update(
                    category="plans/history",
                    kind="external_target_scaffold",
                    lifecycle="planned",
                )
        record = _external_self(local_entry, path)
        if discovery != "self":
            relative = path.relative_to(root)
            record["artifact_id"] = _slug(f"{entry['id_prefix']}-{relative}")
        records.append(record)
    return records, []


def _media_aggregate(root: Path) -> str:
    videos = sorted((root / "videos").rglob("*.mp4")) if (root / "videos").is_dir() else []
    if not videos:
        return ""
    digest = hashlib.sha256()
    for path in videos:
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _mark_duplicates(records: list[dict[str, Any]]) -> None:
    plan_counts = Counter(
        record["plan_sha256"]
        for record in records
        if record.get("_plan_only") and record.get("plan_sha256")
    )
    for record in records:
        plan_hash = record.get("plan_sha256", "")
        if record.get("_plan_only") and plan_hash and plan_counts[plan_hash] > 1:
            record["duplicate_group"] = f"exact-plan:{plan_hash[:16]}"

    media_hashes: dict[str, str] = {}
    for record in records:
        if not record.get("_canonical_media"):
            continue
        aggregate = _media_aggregate(Path(record["path"]))
        if aggregate:
            media_hashes[record["artifact_id"]] = aggregate
    media_counts = Counter(media_hashes.values())
    for record in records:
        aggregate = media_hashes.get(record["artifact_id"], "")
        if aggregate and media_counts[aggregate] > 1:
            record["duplicate_group"] = f"exact-media:{aggregate[:16]}"


def _write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _tsv(records: list[dict[str, Any]]) -> str:
    from io import StringIO

    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=TSV_COLUMNS, delimiter="\t", extrasaction="ignore")
    writer.writeheader()
    for record in records:
        row = dict(record)
        if row["release_eligible"] is None:
            row["release_eligible"] = ""
        else:
            row["release_eligible"] = str(row["release_eligible"]).lower()
        writer.writerow(row)
    return output.getvalue()


def _build_view(
    catalog_root: Path, records: list[dict[str, Any]], content_hash: str
) -> tuple[Path, list[str]]:
    view_root = catalog_root / "views" / f"snapshot-{content_hash[:16]}"
    conflicts: list[str] = []
    for record in records:
        category_dir = view_root / record["category"]
        category_dir.mkdir(parents=True, exist_ok=True)
        link = category_dir / record["artifact_id"]
        target = record["path"]
        if link.is_symlink():
            if os.readlink(link) != target:
                conflicts.append(f"symlink conflict: {link} -> {os.readlink(link)}")
            continue
        if link.exists():
            conflicts.append(f"non-symlink view entry exists: {link}")
            continue
        link.symlink_to(target, target_is_directory=True)
    return view_root, conflicts


def _parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--roots-config",
        type=Path,
        default=repo_root / "migration" / "dataset_catalog_roots.json",
    )
    parser.add_argument("--catalog-root", type=Path, default=DEFAULT_CATALOG_ROOT)
    parser.add_argument(
        "--readme",
        type=Path,
        default=repo_root / "migration" / "dataset_catalog_README.md",
    )
    parser.add_argument("--build-symlink-view", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = _json(args.roots_config)
    if not isinstance(config, dict):
        raise SystemExit(f"invalid roots config: {args.roots_config}")

    runs_root = Path(str(config["canonical_runs_root"]))
    if not runs_root.is_dir():
        raise SystemExit(f"canonical runs root does not exist: {runs_root}")

    records = [
        _canonical_run(path)
        for path in sorted(runs_root.iterdir())
        if path.is_dir()
    ]
    scan_errors: list[str] = []
    for entry in config.get("external_roots", []):
        if not isinstance(entry, dict):
            scan_errors.append(f"invalid external root entry: {entry!r}")
            continue
        print(f"scan external: {entry.get('path', '<missing path>')}", file=sys.stderr)
        external_records, errors = _scan_external(entry)
        records.extend(external_records)
        scan_errors.extend(errors)

    by_artifact_id = {record["artifact_id"]: record for record in records}
    overrides = config.get("canonical_overrides", {})
    if not isinstance(overrides, dict):
        scan_errors.append("canonical_overrides must be a mapping")
    else:
        for artifact_id, override in sorted(overrides.items()):
            record = by_artifact_id.get(str(artifact_id))
            if record is None:
                scan_errors.append(f"canonical override target is absent: {artifact_id}")
                continue
            if not isinstance(override, dict):
                scan_errors.append(f"canonical override is not a mapping: {artifact_id}")
                continue
            for key in ("category", "human_review", "release_eligible"):
                if key in override:
                    record[key] = override[key]
            notes_append = str(override.get("notes_append", "")).strip()
            if notes_append:
                record["notes"] = "; ".join(
                    filter(None, (str(record.get("notes", "")), notes_append))
                )

    plan_records = [record for record in records if record.get("_plan_only")]
    if plan_records:
        newest = max(plan_records, key=lambda record: record["modified_at"])
        newest["category"] = "plans/current"
        newest["notes"] = "; ".join(
            filter(None, (newest["notes"], "newest canonical review-suite plan"))
        )

    _mark_duplicates(records)
    for record in records:
        record.pop("_plan_only", None)
        record.pop("_canonical_media", None)
    records.sort(key=lambda record: (record["category"], record["artifact_id"]))

    canonical_records = json.dumps(
        records, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    content_hash = hashlib.sha256(canonical_records).hexdigest()
    category_counts = dict(sorted(Counter(r["category"] for r in records).items()))
    lifecycle_counts = dict(sorted(Counter(r["lifecycle"] for r in records).items()))

    args.catalog_root.mkdir(parents=True, exist_ok=True)
    view_path = ""
    view_conflicts: list[str] = []
    if args.build_symlink_view:
        view_root, view_conflicts = _build_view(args.catalog_root, records, content_hash)
        view_path = str(view_root)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "catalog_root": str(args.catalog_root.resolve()),
        "roots_config": str(args.roots_config.resolve()),
        "content_sha256": content_hash,
        "symlink_view": view_path,
        "summary": {
            "artifact_count": len(records),
            "category_counts": category_counts,
            "lifecycle_counts": lifecycle_counts,
            "scan_error_count": len(scan_errors),
            "view_conflict_count": len(view_conflicts),
        },
        "scan_errors": scan_errors,
        "view_conflicts": view_conflicts,
        "artifacts": records,
    }
    _write_atomic(
        args.catalog_root / "manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )
    _write_atomic(args.catalog_root / "manifest.tsv", _tsv(records))
    if args.readme.is_file():
        _write_atomic(
            args.catalog_root / "README.md",
            args.readme.read_text(encoding="utf-8"),
        )

    print(json.dumps(manifest["summary"], indent=2, sort_keys=True))
    print(f"manifest: {args.catalog_root / 'manifest.json'}")
    print(f"table:    {args.catalog_root / 'manifest.tsv'}")
    if view_path:
        print(f"view:     {view_path}")
    return 1 if scan_errors or view_conflicts else 0


if __name__ == "__main__":
    sys.exit(main())
