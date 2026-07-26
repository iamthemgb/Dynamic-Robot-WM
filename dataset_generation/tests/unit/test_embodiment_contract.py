from __future__ import annotations

import argparse
from pathlib import Path

import pytest
import yaml

from dynamic_robot_dataset import cli
from dynamic_robot_dataset.common.embodiments import (
    FORBIDDEN_CUSTOM_ATTACHMENTS,
    FRANKA_HAND,
    PRODUCTION_END_EFFECTORS,
    ROBOTIQ_2F85_THICK_PAD,
    SOURCE_GENERATOR_ADAPTERS,
    EmbodimentContractError,
    SourceGeneratorAdapterSpec,
    reject_retired_custom_tool_backend,
    validate_production_end_effector,
)


ROOT = Path(__file__).resolve().parents[2]


def test_production_allowlist_is_real_franka_and_robotiq_only() -> None:
    assert PRODUCTION_END_EFFECTORS == {FRANKA_HAND, ROBOTIQ_2F85_THICK_PAD}
    assert validate_production_end_effector("franka-hand") == FRANKA_HAND
    assert (
        validate_production_end_effector("robotiq_2f85_thick_pad")
        == ROBOTIQ_2F85_THICK_PAD
    )
    for value in FORBIDDEN_CUSTOM_ATTACHMENTS:
        with pytest.raises(EmbodimentContractError, match="retired custom attachment"):
            validate_production_end_effector(value)


def test_retired_custom_tool_backend_fails_before_generation() -> None:
    with pytest.raises(EmbodimentContractError, match="tray/paddle/bin"):
        reject_retired_custom_tool_backend("native-mujoco")


def test_every_registered_source_adapter_uses_allowed_end_effector() -> None:
    assert set(SOURCE_GENERATOR_ADAPTERS) == {
        "franka_hand_catch",
        "franka_hand_bounce",
        "robotiq_2f85_catch",
        "franka_hand_cloth_preview",
    }
    assert {
        value.end_effector for value in SOURCE_GENERATOR_ADAPTERS.values()
    } == PRODUCTION_END_EFFECTORS
    for adapter_id in (
        "franka_hand_catch",
        "franka_hand_bounce",
        "robotiq_2f85_catch",
    ):
        adapter = SOURCE_GENERATOR_ADAPTERS[adapter_id]
        assert adapter.production_role == "rigid_source_assisted_only"
        assert adapter.release_candidate is False
    cloth = SOURCE_GENERATOR_ADAPTERS["franka_hand_cloth_preview"]
    assert cloth.release_candidate is False
    assert cloth.production_role == "deformable_preview_assisted"


def test_versioned_embodiment_config_matches_code_allowlist() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/embodiments/production_end_effectors.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert {value["id"] for value in config["allowed_end_effectors"]} == (
        PRODUCTION_END_EFFECTORS
    )
    assert set(config["forbidden_custom_attachments"]) == (
        FORBIDDEN_CUSTOM_ATTACHMENTS
    )
    assert config["retired_backend"] == "native_mujoco"
    assert config["canonical_integration_backend"] == "source_mujoco"


def test_source_adapter_inspection_hashes_required_files_without_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "generator.py").write_text("# source\n", encoding="utf-8")
    monkeypatch.setenv("TEST_END_EFFECTOR_SOURCE", str(tmp_path))
    spec = SourceGeneratorAdapterSpec(
        adapter_id="test",
        end_effector=FRANKA_HAND,
        robot_model="franka_panda",
        module="generator",
        source_root_env="TEST_END_EFFECTOR_SOURCE",
        source_root_candidates=(),
        required_files=("generator.py",),
        families=("falling_catch",),
        subfamilies=("centered_vertical_drop",),
        state_dim=18,
        action_dim=8,
        action_semantics="test",
        production_role="rigid_source_assisted_only",
        release_candidate=False,
        notes="test",
    )
    before = (tmp_path / "generator.py").read_bytes()
    report = spec.inspect()
    assert report["accessible"] is True
    assert len(report["required_file_sha256"]["generator.py"]) == 64
    assert report["source_tree_read_only"] is True
    assert report["release_ready"] is False
    assert (tmp_path / "generator.py").read_bytes() == before


def test_previous_acceptance_suite_is_retired_and_public_execution_is_blocked() -> None:
    path = ROOT / "configs/families/native_acceptance_160.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert config["execution_allowed"] is False
    assert config["status"] == "retired_custom_attachment_definition"
    with pytest.raises(EmbodimentContractError, match="synthetic tray/paddle/bin"):
        cli._command_generate_suite(
            argparse.Namespace(
                config=str(path), output=None, resume=False, dry_run=True
            )
        )


@pytest.mark.parametrize(
    "name", ("10h.yaml", "100h.yaml", "300h_blocked.yaml", "1000h_blocked.yaml")
)
def test_release_gates_cannot_accept_the_retired_suite(name: str) -> None:
    config = yaml.safe_load(
        (ROOT / "configs/release_gates" / name).read_text(encoding="utf-8")
    )
    acceptance = config["acceptance_suite"]
    assert acceptance["suite_name"] == "real_gripper_acceptance_160_v2"
    assert set(acceptance["required_end_effectors"]) == PRODUCTION_END_EFFECTORS
    assert acceptance["forbid_custom_flange_attachments"] is True
