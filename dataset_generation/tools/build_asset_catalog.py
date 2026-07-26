#!/usr/bin/env python3
"""Build a hash-addressed catalog of referenced external assets without copying them."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml

from dynamic_robot_dataset.common.episode_writer import write_parquet_atomic
from dynamic_robot_dataset.common.hashing import sha256_file, sha256_json

ASSET_EXTENSIONS = {
    ".xml": "mjcf_or_scene",
    ".mjcf": "mjcf",
    ".urdf": "urdf",
    ".obj": "mesh",
    ".stl": "mesh",
    ".dae": "mesh",
    ".ply": "mesh",
    ".glb": "mesh",
    ".gltf": "mesh",
    ".png": "texture",
    ".jpg": "texture",
    ".jpeg": "texture",
    ".exr": "texture",
    ".hdr": "texture",
}
PRUNED_DIRECTORIES = {".git", ".venv", "__pycache__", ".cache", "videos", "outputs"}


def _notice(root: Path) -> str | None:
    candidates = sorted(
        path
        for pattern in ("LICENSE*", "NOTICE*", "COPYING*")
        for path in root.glob(pattern)
        if path.is_file()
    )
    return candidates[0].relative_to(root).as_posix() if candidates else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/assets/asset_roots.example.yaml")
    parser.add_argument("--output", default="migration/asset_catalog.refresh.parquet")
    parser.add_argument("--max-files", type=int, default=250_000)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    unavailable: list[dict[str, str]] = []
    for project, declaration in sorted(config["roots"].items()):
        environment = str(declaration["environment"])
        root = Path(os.environ.get(environment, declaration["default"])).resolve()
        if not root.is_dir():
            unavailable.append({"source_project": project, "root": str(root), "error": "not a directory"})
            continue
        notice = _notice(root)
        for directory, child_directories, files in os.walk(root, followlinks=False):
            child_directories[:] = [name for name in child_directories if name not in PRUNED_DIRECTORIES]
            for name in files:
                path = Path(directory) / name
                asset_type = ASSET_EXTENSIONS.get(path.suffix.lower())
                if asset_type is None:
                    continue
                relative = path.relative_to(root).as_posix()
                rows.append(
                    {
                        "asset_id": f"{project}-{sha256_json(relative)[:20]}",
                        "source_project": project,
                        "relative_path": relative,
                        "asset_type": asset_type,
                        "category": path.parent.name,
                        "license_or_notice_path": notice or "missing",
                        "compatible_simulator": "mujoco" if path.suffix.lower() in {".xml", ".mjcf"} else "generic",
                        "scale": 1.0,
                        "known_issues": "missing license/notice at configured root" if notice is None else "",
                        "sha256": sha256_file(path),
                        "size_bytes": path.stat().st_size,
                        "configured_root_environment": environment,
                    }
                )
                if len(rows) > args.max_files:
                    raise RuntimeError(
                        f"Asset catalog exceeded --max-files={args.max_files}; narrow configured roots"
                    )
    rows.sort(key=lambda value: (str(value["source_project"]), str(value["relative_path"])))
    write_parquet_atomic(args.output, rows)
    print(
        json.dumps(
            {"output": str(Path(args.output).resolve()), "asset_count": len(rows), "unavailable_roots": unavailable},
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())

