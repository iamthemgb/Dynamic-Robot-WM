from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from dynamic_robot_dataset.backends.source_mujoco import (
    SourceMujocoBackend,
    compile_review_case,
    prepare_review_case,
)
from dynamic_robot_dataset.backends.source_mujoco.backend import (
    _require_runtime_dependencies,
    _robocasa_manifest,
)
from dynamic_robot_dataset.backends.source_mujoco.model import (
    _build_external_sample,
    _load_external_scene_builder,
    compile_source_model,
)
from dynamic_robot_dataset.common.assets import (
    load_robocasa_catalog_policy,
    validate_robocasa_asset_manifest,
)
from dynamic_robot_dataset.common.hashing import sha256_file
from dynamic_robot_dataset.common.review_suite import build_review_suite_plan


ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "configs/assets/robocasa_catalog_v1.yaml"


def _kitchen_case():
    return next(
        case
        for case in build_review_suite_plan().cases
        if case.corpus_leaf_id == "P0c" and case.rollout_index == 2
    )


@pytest.fixture(scope="module")
def kitchen_runtime():
    mujoco, _ = _require_runtime_dependencies()
    backend = SourceMujocoBackend()
    scenario = compile_review_case(_kitchen_case())
    compiled = compile_source_model(
        scenario,
        source_dependency=backend.source_dependency,
        robocasa_dependency=backend.robocasa_dependency,
    )
    mujoco.mj_forward(compiled.model, compiled.data)
    manifest = _robocasa_manifest(
        mujoco, compiled, scenario, backend.robocasa_dependency
    )
    return mujoco, backend, scenario, compiled, manifest


def test_compiled_r1_uses_only_its_content_bound_catalog_candidate(
    kitchen_runtime,
) -> None:
    _, _, scenario, compiled, manifest = kitchen_runtime
    policy = load_robocasa_catalog_policy(CATALOG)
    expected = next(asset for asset in policy.assets if "kitchen" in asset.profiles)

    assert scenario.scene_profile == "robocasa_kitchen"
    assert len(compiled.robocasa_assets) == 1
    selected = compiled.robocasa_assets[0]
    assert selected["catalog_asset_id"] == expected.asset_id
    assert selected["descriptor_sha256"] == expected.descriptor_sha256
    assert selected["manifest_sha256"] == expected.manifest_sha256
    assert selected["license_notice_sha256"] == policy.license_notice_sha256
    assert selected["selection_source"] == "owned_content_bound_catalog"
    assert selected["catalog_release_ready"] is False
    assert manifest[0]["asset_id"] == expected.asset_id
    assert manifest[0]["catalog_version"] == policy.catalog_version
    assert manifest[0]["catalog_sha256"] == policy.catalog_sha256


def test_compiled_catalog_candidate_is_visual_only_and_swept_volume_clear(
    kitchen_runtime,
) -> None:
    mujoco, _, _, compiled, manifest = kitchen_runtime
    row = manifest[0]

    assert row["collision_enabled"] is False
    assert row["swept_volume_clear"] is True
    assert row["fixture_intersection_clear"] is True
    assert row["occlusion_validated"] is False
    assert row["admitted"] is False
    assert row["blockers"] == ["rendered_occlusion_review_pending"]
    assert all(
        int(compiled.model.geom_contype[geom_id]) == 0
        and int(compiled.model.geom_conaffinity[geom_id]) == 0
        for geom_id in range(compiled.model.ngeom)
        if str(
            mujoco.mj_id2name(
                compiled.model,
                mujoco.mjtObj.mjOBJ_BODY,
                int(compiled.model.geom_bodyid[geom_id]),
            )
            or ""
        ).startswith("rc_catalog_kitchen_")
    )


def test_runtime_candidate_is_review_valid_but_not_release_admitted(
    kitchen_runtime,
) -> None:
    manifest = kitchen_runtime[-1]
    with pytest.raises(ValueError, match="not admitted"):
        validate_robocasa_asset_manifest(manifest)
    validate_robocasa_asset_manifest(
        manifest,
        required_asset_ids=("kitchen-toaster-review-candidate-v1",),
        allow_pending_render_review=True,
    )


def test_external_random_robocasa_imports_and_kitchen_appliances_are_absent(
    kitchen_runtime,
) -> None:
    _, _, _, compiled, _ = kitchen_runtime
    xml_root = ET.fromstring(compiled.xml)
    rc_names = tuple(
        str(element.get("name"))
        for element in xml_root.iter()
        if str(element.get("name") or "").startswith("rc_")
    )

    assert compiled.stripped_external_robocasa_element_count > 0
    assert rc_names
    assert all(name.startswith("rc_catalog_kitchen_") for name in rc_names)
    assert not any(
        token in name.lower()
        for name in rc_names
        for token in ("refrigerator", "range")
    )
    camera_assets = compiled.external_camera_metadata["main_camera"][
        "robocasa_background_assets"
    ]
    assert camera_assets == [dict(compiled.robocasa_assets[0])]


def test_source_scenario_spec_retains_exact_catalog_identity_and_hash() -> None:
    backend = SourceMujocoBackend()
    spec = prepare_review_case(_kitchen_case(), backend=backend)
    policy = load_robocasa_catalog_policy(CATALOG)
    expected = next(asset for asset in policy.assets if "kitchen" in asset.profiles)

    assert spec.robocasa_manifest.catalog_version == policy.catalog_version
    assert spec.robocasa_manifest.catalog_sha256 == sha256_file(CATALOG)
    assert spec.robocasa_manifest.license_manifest_sha256 == (
        policy.license_notice_sha256
    )
    assert len(spec.robocasa_manifest.assets) == 1
    selected = spec.robocasa_manifest.assets[0]
    assert selected.asset_id == expected.asset_id
    assert selected.xml_sha256 == expected.descriptor_sha256
    assert selected.source_xml == expected.descriptor_path
    assert selected.collision_enabled is False


def test_external_sample_rng_streams_are_independent() -> None:
    backend = SourceMujocoBackend()
    scenario = compile_review_case(_kitchen_case())
    source_root = Path(backend.source_dependency.source_root)
    original_builder = _load_external_scene_builder(source_root)

    class RecordingBuilder:
        @staticmethod
        def sample_episode(rng, *args, **kwargs):
            sample = original_builder.sample_episode(rng, *args, **kwargs)
            return replace(sample, projectile={"scene_marker": float(rng.random())})

    def changed(stream: str):
        subseeds = dict(scenario.rng_subseeds)
        subseeds[stream] += 1
        return _build_external_sample(
            RecordingBuilder, replace(scenario, rng_subseeds=subseeds)
        )

    baseline = _build_external_sample(RecordingBuilder, scenario)
    camera_changed = changed("camera")
    assets_changed = changed("assets")
    construction_changed = changed("scene_construction")

    appearance = lambda sample: (
        sample.seed,
        sample.ball_color,
        sample.lighting_intensity,
        sample.floor_material_jitter,
    )
    assert camera_changed.camera_jitter != baseline.camera_jitter
    assert appearance(camera_changed) == appearance(baseline)
    assert camera_changed.projectile == baseline.projectile

    assert appearance(assets_changed) != appearance(baseline)
    assert assets_changed.camera_jitter == baseline.camera_jitter
    assert assets_changed.projectile == baseline.projectile

    assert construction_changed.projectile != baseline.projectile
    assert appearance(construction_changed) == appearance(baseline)
    assert construction_changed.camera_jitter == baseline.camera_jitter
