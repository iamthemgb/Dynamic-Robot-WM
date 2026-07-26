from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dynamic_robot_dataset.common.assets import load_robocasa_catalog_policy


ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "configs/assets/robocasa_catalog_v1.yaml"


def _write_catalog(tmp_path: Path, value: object) -> Path:
    path = tmp_path / "robocasa_catalog.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def test_nonempty_content_bound_candidates_do_not_claim_release_readiness() -> None:
    policy = load_robocasa_catalog_policy(CATALOG)

    assert policy.configured_asset_count > 0
    assert policy.review_ready
    assert not policy.release_ready
    assert policy.admitted_asset_count == 0
    assert all(asset.review_candidate for asset in policy.assets)
    assert all(not asset.occlusion_validated for asset in policy.assets)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    (
        ("descriptor_sha256", "0" * 64, "descriptor hash changed"),
        ("license_notice_sha256", "0" * 64, "license binding changed"),
        ("scale_to_meters", None, "scale is not validated"),
        ("visual_only", False, "visual-only with collisions disabled"),
        ("collision_enabled", True, "visual-only with collisions disabled"),
    ),
)
def test_catalog_loader_rejects_unbound_or_unsafe_candidate_rows(
    tmp_path: Path,
    field: str,
    replacement: object,
    message: str,
) -> None:
    value = yaml.safe_load(CATALOG.read_text(encoding="utf-8"))
    value["assets"][0][field] = replacement

    with pytest.raises(ValueError, match=message):
        load_robocasa_catalog_policy(_write_catalog(tmp_path, value))


def test_catalog_loader_rejects_missing_or_changed_referenced_content(
    tmp_path: Path,
) -> None:
    value = yaml.safe_load(CATALOG.read_text(encoding="utf-8"))
    manifest = value["assets"][0]["referenced_file_sha256"]
    manifest.pop(next(path for path in manifest if path.endswith(".png")))

    with pytest.raises(ValueError, match="exactly bind"):
        load_robocasa_catalog_policy(_write_catalog(tmp_path, value))


def test_release_readiness_requires_admission_and_occlusion_for_every_profile(
    tmp_path: Path,
) -> None:
    value = yaml.safe_load(CATALOG.read_text(encoding="utf-8"))
    for asset in value["assets"]:
        asset["scene_clearance_validated"] = True
        asset["fixture_intersection_validated"] = True
        asset["occlusion_validated"] = True
        asset["admitted"] = True
        asset["blockers"] = []

    policy = load_robocasa_catalog_policy(_write_catalog(tmp_path, value))

    assert policy.review_ready
    assert policy.release_ready
    assert policy.admitted_asset_count == policy.configured_asset_count
    assert set(policy.release_ready_profiles) == set(policy.profiles)


def test_false_admission_claim_is_rejected(tmp_path: Path) -> None:
    value = yaml.safe_load(CATALOG.read_text(encoding="utf-8"))
    value["assets"][0]["admitted"] = True

    with pytest.raises(ValueError, match="false admission claim"):
        load_robocasa_catalog_policy(_write_catalog(tmp_path, value))
