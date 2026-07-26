"""Production embodiment allowlist and read-only source-adapter inventory.

The first native rigid backend in this repository mounted a synthetic tray,
bin, or paddle on the Franka flange.  That is not the embodiment used by the
source datasets and must never be presented as Panda-hand or Robotiq data.

This module is intentionally small and dependency-free.  It establishes the
hard production boundary while the existing, separately maintained generators
are adapted into the canonical v2 writer:

* ``franka_hand`` means the MuJoCo Menagerie Panda hand and its two fingers.
* ``robotiq_2f85_thick_pad`` means the Menagerie Robotiq 2F-85 mounted on a
  no-hand Panda, including the generator's declared fingertip pads.

Source trees are inspected read-only.  Presence and hashes establish source
provenance; they do not by themselves establish release readiness.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from .hashing import combined_manifest_hash, sha256_file


FRANKA_HAND = "franka_hand"
ROBOTIQ_2F85_THICK_PAD = "robotiq_2f85_thick_pad"
PRODUCTION_END_EFFECTORS = frozenset({FRANKA_HAND, ROBOTIQ_2F85_THICK_PAD})

# These names describe the removed custom attachment path, not allowed
# aliases for either real gripper.  Keep this list explicit so new configs and
# converted records fail closed instead of being silently normalized.
FORBIDDEN_CUSTOM_ATTACHMENTS = frozenset(
    {
        "native_task_tool",
        "shallow_tray",
        "deep_tray",
        "small_bin",
        "flat_paddle",
        "angled_paddle",
    }
)

RETIRED_CUSTOM_TOOL_BACKEND = "native_mujoco"
CORRECTED_SOURCE_BACKEND = "source_mujoco"


class EmbodimentContractError(ValueError):
    """A request would misrepresent a custom attachment as a real gripper."""


def validate_production_end_effector(value: str) -> str:
    """Return a canonical allowed end-effector name or fail closed."""

    normalized = str(value).strip().lower().replace("-", "_")
    if normalized in FORBIDDEN_CUSTOM_ATTACHMENTS:
        raise EmbodimentContractError(
            f"{value!r} is a retired custom attachment, not a production "
            "Franka/Robotiq end effector"
        )
    if normalized not in PRODUCTION_END_EFFECTORS:
        allowed = ", ".join(sorted(PRODUCTION_END_EFFECTORS))
        raise EmbodimentContractError(
            f"unsupported production end effector {value!r}; allowed: {allowed}"
        )
    return normalized


def reject_retired_custom_tool_backend(backend: str) -> None:
    """Reject the old public execution path before it can create output."""

    normalized = str(backend).strip().lower().replace("-", "_")
    if normalized == RETIRED_CUSTOM_TOOL_BACKEND:
        raise EmbodimentContractError(
            "backend 'native_mujoco' is retired for generation because it mounts "
            "synthetic tray/paddle/bin geometry. Use the real Franka-hand or "
            "Robotiq source adapters after canonical-v2 integration; the old "
            "backend is retained only for quarantined regression tests."
        )


@dataclass(frozen=True)
class SourceGeneratorAdapterSpec:
    """Read-only contract for one existing embodiment generator lineage."""

    adapter_id: str
    end_effector: str
    robot_model: str
    module: str
    source_root_env: str
    source_root_candidates: tuple[str, ...]
    required_files: tuple[str, ...]
    families: tuple[str, ...]
    subfamilies: tuple[str, ...]
    state_dim: int | None
    action_dim: int | None
    action_semantics: str
    production_role: str
    release_candidate: bool
    notes: str

    def __post_init__(self) -> None:
        validate_production_end_effector(self.end_effector)
        if self.production_role not in {
            "rigid_source_candidate",
            "rigid_source_assisted_only",
            "deformable_preview_assisted",
        }:
            raise ValueError(f"unknown source-adapter role {self.production_role!r}")
        if not self.required_files:
            raise ValueError("source adapters require at least one provenance file")

    def resolve_root(self) -> Path | None:
        override = os.environ.get(self.source_root_env)
        candidates: Iterable[str] = (override,) if override else self.source_root_candidates
        first_existing: Path | None = None
        for raw in candidates:
            path = Path(raw).expanduser().resolve()
            if path.is_dir():
                if first_existing is None:
                    first_existing = path
                if all((path / relative).is_file() for relative in self.required_files):
                    return path
        return first_existing

    def inspect(self) -> dict[str, Any]:
        root = self.resolve_root()
        missing: list[str] = []
        manifest: dict[str, str] = {}
        if root is None:
            missing.extend(self.required_files)
        else:
            for relative in self.required_files:
                path = root / relative
                if not path.is_file():
                    missing.append(relative)
                else:
                    manifest[relative] = sha256_file(path)
        return {
            **asdict(self),
            "source_root": None if root is None else str(root),
            "accessible": root is not None and not missing,
            "missing_required_files": missing,
            "required_file_sha256": dict(sorted(manifest.items())),
            "source_manifest_sha256": (
                combined_manifest_hash(manifest) if manifest and not missing else None
            ),
            "source_tree_read_only": True,
            "release_ready": False,
        }


SOURCE_GENERATOR_ADAPTERS: Mapping[str, SourceGeneratorAdapterSpec] = {
    "franka_hand_catch": SourceGeneratorAdapterSpec(
        adapter_id="franka_hand_catch",
        end_effector=FRANKA_HAND,
        robot_model="franka_panda",
        module="franka_catch.generate",
        source_root_env="FRANKA_HAND_SOURCE_ROOT",
        source_root_candidates=(
            "/gpfs/radev/project/sous/zss8/dataset-generation/demo_mujoco_arm_gripper",
        ),
        required_files=(
            "franka_catch/generate.py",
            "franka_catch/scene_builder.py",
            "franka_catch/controller.py",
            "third_party/mujoco_menagerie/franka_emika_panda/panda.xml",
        ),
        families=("falling_catch",),
        subfamilies=("centered_vertical_drop",),
        state_dim=18,
        action_dim=8,
        action_semantics="7 Panda joint-position commands plus one finger command",
        production_role="rigid_source_assisted_only",
        release_candidate=False,
        notes="Real Panda hand/finger geometry, but successful capture rewrites/holds ball state; requires actuator-only and free-contact repair before v2 candidacy.",
    ),
    "franka_hand_bounce": SourceGeneratorAdapterSpec(
        adapter_id="franka_hand_bounce",
        end_effector=FRANKA_HAND,
        robot_model="franka_panda",
        module="franka_bounce.generate",
        source_root_env="FRANKA_HAND_SOURCE_ROOT",
        source_root_candidates=(
            "/gpfs/radev/project/sous/zss8/dataset-generation/demo_mujoco_arm_gripper",
        ),
        required_files=(
            "franka_bounce/generate.py",
            "franka_bounce/scene_builder.py",
            "franka_bounce/controller.py",
            "third_party/mujoco_menagerie/franka_emika_panda/panda.xml",
        ),
        families=("projectile_rebound", "falling_catch"),
        subfamilies=("table_bounce", "bounce_to_robot_interception"),
        state_dim=18,
        action_dim=8,
        action_semantics="7 Panda joint-position commands plus one finger command",
        production_role="rigid_source_assisted_only",
        release_candidate=False,
        notes="Real Panda hand, but bounce response and successful capture rewrite ball state; requires native-contact/free-grasp repair before v2 candidacy.",
    ),
    "robotiq_2f85_catch": SourceGeneratorAdapterSpec(
        adapter_id="robotiq_2f85_catch",
        end_effector=ROBOTIQ_2F85_THICK_PAD,
        robot_model="franka_panda_nohand_plus_robotiq_2f85",
        module="robotiq_catch.generate",
        source_root_env="ROBOTIQ_SOURCE_ROOT",
        source_root_candidates=(
            "/gpfs/radev/home/zl664/project/demo_mujoco_arm_gripper",
            "/gpfs/radev/project/sous/zl664/demo_mujoco_arm_gripper",
        ),
        required_files=(
            "robotiq_catch/generate.py",
            "robotiq_catch/controller.py",
            "scripts_mujoco/scene_builder.py",
            "third_party/mujoco_menagerie/robotiq_2f85/2f85.xml",
        ),
        families=("falling_catch", "projectile_rebound"),
        subfamilies=("centered_vertical_drop", "direct_projectile_interception"),
        state_dim=24,
        action_dim=15,
        action_semantics="7 Panda arm plus 8 native/coupled Robotiq joint-position commands",
        production_role="rigid_source_assisted_only",
        release_candidate=False,
        notes="Real Robotiq 2F-85 mounted on no-hand Panda, but successful capture holds ball qpos at the captured offset; requires free-contact repair before v2 candidacy.",
    ),
    "franka_hand_cloth_preview": SourceGeneratorAdapterSpec(
        adapter_id="franka_hand_cloth_preview",
        end_effector=FRANKA_HAND,
        robot_model="franka_panda",
        module="mujoco_franka_cloth_previews.generate_previews",
        source_root_env="FRANKA_DEFORMABLE_SOURCE_ROOT",
        source_root_candidates=(
            "/gpfs/radev/home/zl664/project/demo_mujoco_deformable",
            "/gpfs/radev/project/sous/zl664/demo_mujoco_deformable",
        ),
        required_files=(
            "mujoco_franka_cloth_previews/README.md",
            "mujoco_franka_cloth_previews/generate_previews.py",
            "mujoco_franka_cloth_previews/scene_builder.py",
        ),
        families=("cloth",),
        subfamilies=("poke_cloth", "lift_corner_release", "fold_edge_fixed_line"),
        state_dim=None,
        action_dim=None,
        action_semantics="Panda joint targets plus gripper command/force and cloth state sidecars",
        production_role="deformable_preview_assisted",
        release_candidate=False,
        notes="Preview lineage. Poke is contact-based; lift/fold use equality-connect proxy grasps and remain assisted/quarantined.",
    ),
}


def inspect_source_generator_adapters(
    adapter_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Inspect the selected source contracts without importing or executing them."""

    selected = sorted(adapter_ids or SOURCE_GENERATOR_ADAPTERS)
    unknown = sorted(set(selected) - set(SOURCE_GENERATOR_ADAPTERS))
    if unknown:
        raise KeyError(f"unknown source adapter(s): {', '.join(unknown)}")
    adapters = [SOURCE_GENERATOR_ADAPTERS[name].inspect() for name in selected]
    rigid_candidates = [
        value
        for value in adapters
        if str(value["production_role"]).startswith("rigid_source_")
    ]
    return {
        "schema_version": "dynamic-robot-embodiment-sources/v1",
        "allowed_end_effectors": sorted(PRODUCTION_END_EFFECTORS),
        "forbidden_custom_attachments": sorted(FORBIDDEN_CUSTOM_ATTACHMENTS),
        "retired_backend": RETIRED_CUSTOM_TOOL_BACKEND,
        "canonical_integration_backend": CORRECTED_SOURCE_BACKEND,
        "adapter_count": len(adapters),
        "all_selected_sources_accessible": all(
            bool(value["accessible"]) for value in adapters
        ),
        "all_rigid_candidate_sources_accessible": bool(rigid_candidates)
        and all(bool(value["accessible"]) for value in rigid_candidates),
        "adapters": adapters,
    }


__all__ = [
    "CORRECTED_SOURCE_BACKEND",
    "EmbodimentContractError",
    "FORBIDDEN_CUSTOM_ATTACHMENTS",
    "FRANKA_HAND",
    "PRODUCTION_END_EFFECTORS",
    "RETIRED_CUSTOM_TOOL_BACKEND",
    "ROBOTIQ_2F85_THICK_PAD",
    "SOURCE_GENERATOR_ADAPTERS",
    "SourceGeneratorAdapterSpec",
    "inspect_source_generator_adapters",
    "reject_retired_custom_tool_backend",
    "validate_production_end_effector",
]
