#!/usr/bin/env python3
"""Create review contact sheets and an exact sampled-path ledger."""

from __future__ import annotations

import argparse
import io
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import av
from PIL import Image, ImageDraw

from dynamic_robot_dataset.common.episode_writer import load_episode_records
from dynamic_robot_dataset.common.paths import (
    atomic_write_bytes,
    atomic_write_json,
    ensure_not_source_path,
    resolve_dataset_path,
)
from dynamic_robot_dataset.common.schema import EpisodeRecord


def _indices(frame_count: int, count: int = 6) -> list[int]:
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if count <= 1:
        return [frame_count // 2]
    return [round(index * (frame_count - 1) / (count - 1)) for index in range(count)]


def _sample_frames(path: Path, indices: Iterable[int]) -> dict[int, Image.Image]:
    wanted = set(indices)
    samples: dict[int, Image.Image] = {}
    with av.open(str(path), mode="r") as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index in wanted:
                samples[index] = frame.to_image().convert("RGB")
            if len(samples) == len(wanted):
                break
    missing = wanted - samples.keys()
    if missing:
        raise RuntimeError(f"Could not decode sampled frames {sorted(missing)} from {path}")
    return samples


def _select(records: list[EpisodeRecord], per_family: int) -> dict[str, list[EpisodeRecord]]:
    grouped: dict[str, list[EpisodeRecord]] = defaultdict(list)
    for record in records:
        grouped[record.family].append(record)
    selected: dict[str, list[EpisodeRecord]] = {}
    for family, values in sorted(grouped.items()):
        values.sort(key=lambda record: (record.actual_outcome, record.scene_seed, record.episode_index))
        chosen: list[EpisodeRecord] = []
        seen_outcomes: set[str] = set()
        seen_styles: set[str] = set()
        for record in values:
            style = str(record.randomization.get("scene_style", "unknown"))
            if record.actual_outcome not in seen_outcomes or style not in seen_styles:
                chosen.append(record)
                seen_outcomes.add(record.actual_outcome)
                seen_styles.add(style)
            if len(chosen) == per_family:
                break
        for record in values:
            if len(chosen) == per_family:
                break
            if record not in chosen:
                chosen.append(record)
        selected[family] = chosen
    return selected


def make_contact_sheets(dataset_root: Path, output: Path, *, per_family: int = 4, resume: bool = False) -> dict:
    records = load_episode_records(dataset_root)
    selected = _select(records, per_family)
    output = ensure_not_source_path(output)
    output.mkdir(parents=True, exist_ok=True)
    ledger: dict = {
        "schema_version": "dynamic-robot-contact-sheet-samples/v1",
        "dataset_root": str(dataset_root.resolve()),
        "families": {},
    }
    tile_width, tile_height, label_height = 208, 120, 22
    sample_count = 6
    for family, family_records in selected.items():
        row_count = sum(len(record.video_paths) for record in family_records)
        sheet = Image.new("RGB", (sample_count * tile_width, row_count * (tile_height + label_height)), (24, 26, 29))
        draw = ImageDraw.Draw(sheet)
        row = 0
        ledger_rows: list[dict] = []
        for record in family_records:
            frame_indices = _indices(int(record.frame_count or 0), sample_count)
            for camera, relative in sorted(record.video_paths.items()):
                video_path = resolve_dataset_path(dataset_root, relative)
                samples = _sample_frames(video_path, frame_indices)
                label = (
                    f"ep={record.episode_index} {record.actual_outcome} "
                    f"{record.randomization.get('scene_style', 'unknown')} {camera.rsplit('.', 1)[-1]}"
                )
                y = row * (tile_height + label_height)
                draw.rectangle((0, y, sheet.width, y + label_height), fill=(24, 26, 29))
                draw.text((5, y + 4), label, fill=(238, 240, 242))
                for column, frame_index in enumerate(frame_indices):
                    tile = samples[frame_index].resize((tile_width, tile_height), Image.Resampling.LANCZOS)
                    sheet.paste(tile, (column * tile_width, y + label_height))
                ledger_rows.append(
                    {
                        "episode_uuid": record.episode_uuid,
                        "episode_index": record.episode_index,
                        "actual_outcome": record.actual_outcome,
                        "intended_branch": record.intended_branch,
                        "scene_style": record.randomization.get("scene_style"),
                        "camera": camera,
                        "video_path": relative,
                        "sampled_frame_indices": frame_indices,
                    }
                )
                row += 1
        destination = output / f"{family}.png"
        if not destination.exists():
            buffer = io.BytesIO()
            sheet.save(buffer, format="PNG", optimize=True)
            atomic_write_bytes(destination, buffer.getvalue())
        elif not resume:
            raise FileExistsError(f"Contact sheet exists: {destination}")
        ledger["families"][family] = {
            "contact_sheet": destination.relative_to(dataset_root).as_posix(),
            "samples": ledger_rows,
        }
    ledger_path = output / "samples.json"
    if not ledger_path.exists():
        atomic_write_json(ledger_path, ledger)
    elif not resume:
        raise FileExistsError(f"Sample ledger exists: {ledger_path}")
    return ledger


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--per-family", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output = args.output or args.dataset_root / "qc" / "contact_sheets"
    ledger = make_contact_sheets(args.dataset_root, output, per_family=args.per_family, resume=args.resume)
    print(f"wrote {len(ledger['families'])} family contact sheets to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
