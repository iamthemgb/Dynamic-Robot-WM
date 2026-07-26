from __future__ import annotations

import os
import importlib.util
from dataclasses import dataclass
from pathlib import Path

from .utils import repo_root


@dataclass(frozen=True)
class AssetInfo:
    robot_xml: Path
    source: str
    asset_dir: Path
    note: str


def _candidate_paths() -> list[tuple[str, Path, str]]:
    root = repo_root()
    menagerie = root / "demo_mujoco_arm_gripper" / "third_party" / "mujoco_menagerie"
    candidates: list[tuple[str, Path, str]] = []
    env_path = os.environ.get("FRANKA_MJCF_PATH")
    if env_path:
        candidates.append(("env", Path(env_path), "FRANKA_MJCF_PATH"))
    candidates.extend(
        [
            ("menagerie", menagerie / "franka_emika_panda" / "panda.xml", "MuJoCo Menagerie Panda"),
            ("menagerie", menagerie / "franka_fr3" / "fr3.xml", "MuJoCo Menagerie FR3"),
        ]
    )
    genesis_spec = importlib.util.find_spec("genesis")
    if genesis_spec and genesis_spec.origin:
        genesis_root = Path(genesis_spec.origin).resolve().parent
        candidates.append(
            (
                "genesis_package",
                genesis_root / "assets" / "xml" / "franka_emika_panda" / "panda.xml",
                "Installed Genesis Panda MJCF fallback",
            )
        )
    return candidates


def locate_franka_asset() -> AssetInfo:
    checked: list[str] = []
    for source, path, note in _candidate_paths():
        checked.append(str(path))
        if path.exists():
            return AssetInfo(robot_xml=path.resolve(), source=source, asset_dir=path.resolve().parent / "assets", note=note)
    raise FileNotFoundError(
        "Could not find a Franka/Panda MJCF. Install MuJoCo Menagerie with franka_emika_panda, "
        "or set FRANKA_MJCF_PATH to a Panda/FR3 XML. Checked:\n"
        + "\n".join(checked)
    )


def locate_franka_nohand_asset() -> AssetInfo:
    franka = locate_franka_asset()
    nohand_xml = franka.robot_xml.with_name("panda_nohand.xml")
    if nohand_xml.exists():
        return AssetInfo(
            robot_xml=nohand_xml.resolve(),
            source=franka.source,
            asset_dir=franka.asset_dir,
            note=f"{franka.note} no-hand attachment variant",
        )
    raise FileNotFoundError(
        "Could not find panda_nohand.xml next to the Franka/Panda MJCF. "
        f"Checked: {nohand_xml}"
    )
