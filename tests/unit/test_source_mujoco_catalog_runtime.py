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
    _add_secondary_camera,
    _build_external_sample,
    _compiled_world_geom_aabb,
    _load_external_scene_builder,
    _positive_aabb_overlap,
    _relocate_external_visual_boundaries,
    _remove_external_visual_work_surfaces,
    _repair_task_camera,
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


def _kitchen_f1_case():
    return next(
        case
        for case in build_review_suite_plan().cases
        if case.corpus_leaf_id == "F1a" and case.rollout_index == 2
    )


def _case(leaf_id: str, rollout_index: int):
    return next(
        case
        for case in build_review_suite_plan().cases
        if case.corpus_leaf_id == leaf_id
        and case.rollout_index == rollout_index
    )


def _camera_root() -> ET.Element:
    root = ET.Element("mujoco")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "camera", name="main_camera")
    return root


def _camera_forward(camera: ET.Element) -> tuple[float, float, float]:
    values = tuple(float(value) for value in str(camera.get("xyaxes") or "").split())
    assert len(values) == 6
    x_axis = values[:3]
    y_axis = values[3:]
    camera_z = (
        x_axis[1] * y_axis[2] - x_axis[2] * y_axis[1],
        x_axis[2] * y_axis[0] - x_axis[0] * y_axis[2],
        x_axis[0] * y_axis[1] - x_axis[1] * y_axis[0],
    )
    return tuple(-value for value in camera_z)


def _direction(
    position: tuple[float, float, float],
    target: tuple[float, float, float],
) -> tuple[float, float, float]:
    delta = tuple(right - left for left, right in zip(position, target))
    norm = sum(value * value for value in delta) ** 0.5
    return tuple(value / norm for value in delta)


def _camera_pose(camera: ET.Element) -> tuple[float, float, float]:
    return tuple(float(value) for value in str(camera.get("pos") or "").split())


def _assert_camera_looks_at(
    camera: ET.Element,
    *,
    position: tuple[float, float, float],
    target: tuple[float, float, float],
    fovy: float,
) -> None:
    assert _camera_pose(camera) == pytest.approx(position, abs=1e-8)
    assert float(camera.get("fovy", "nan")) == pytest.approx(fovy)
    assert _camera_forward(camera) == pytest.approx(
        _direction(position, target), abs=1e-8
    )


def _background_root() -> ET.Element:
    root = ET.Element("mujoco")
    world = ET.SubElement(root, "worldbody")
    for name in (
        "robot_table_top",
        "robot_table_leg_0",
        "back_counter_base",
        "back_counter_top",
    ):
        ET.SubElement(
            world,
            "geom",
            name=name,
            type="box",
            pos="0 0 0.5",
            size="0.1 0.1 0.1",
        )
    ET.SubElement(
        world,
        "geom",
        name="left_wall",
        type="box",
        pos="-0.95 0.1 1.1",
        size="0.03 1.1 1.1",
    )
    ET.SubElement(
        world,
        "geom",
        name="backsplash_left_return",
        type="box",
        pos="-0.94 0.5 0.9",
        size="0.02 0.2 0.2",
    )
    return root


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
    manifest = backend.run(scenario, render=False).robocasa_asset_manifest
    return mujoco, backend, scenario, compiled, manifest


@pytest.fixture(scope="module")
def kitchen_f1_runtime_manifest():
    return SourceMujocoBackend().run(
        _kitchen_f1_case(), render=False
    ).robocasa_asset_manifest


@pytest.fixture(scope="module")
def f2b_compiled_review_suite():
    mujoco, _ = _require_runtime_dependencies()
    backend = SourceMujocoBackend()
    rows = []
    for rollout_index in range(6):
        scenario = compile_review_case(_case("F2b", rollout_index))
        compiled = compile_source_model(
            scenario,
            source_dependency=backend.source_dependency,
            robocasa_dependency=backend.robocasa_dependency,
        )
        mujoco.mj_forward(compiled.model, compiled.data)
        rows.append((scenario, compiled))
    return mujoco, tuple(rows)


def test_f2b_clean_r0_removes_exact_fixture_intersecting_visuals(
    f2b_compiled_review_suite,
) -> None:
    _, rows = f2b_compiled_review_suite
    scenario, compiled = rows[0]
    assert scenario.scene_profile == "clean_R0"
    assert scenario.requires_real_robocasa is False
    assert compiled.removed_fixture_intersection_background_names == (
        "lab_bench_leg_a",
        "lab_workbench",
    )
    remaining = {
        str(row["source_name"]) for row in compiled.background_geom_descriptors
    }
    assert "lab_bench_leg_a" not in remaining
    assert "lab_workbench" not in remaining
    assert {"back_wall", "lab_bench_leg_b", "left_wall"} <= remaining


def test_all_six_f2b_backgrounds_clear_every_owned_fixture(
    f2b_compiled_review_suite,
) -> None:
    mujoco, rows = f2b_compiled_review_suite
    assert len(rows) == 6
    for scenario, compiled in rows:
        fixture_aabbs = tuple(
            _compiled_world_geom_aabb(
                compiled.model,
                compiled.data,
                int(
                    mujoco.mj_name2id(
                        compiled.model,
                        mujoco.mjtObj.mjOBJ_GEOM,
                        surface.name,
                    )
                ),
            )
            for surface in scenario.surfaces
        )
        for background in compiled.background_geom_descriptors:
            background_aabb = _compiled_world_geom_aabb(
                compiled.model,
                compiled.data,
                int(background["geom_id"]),
            )
            assert not any(
                _positive_aabb_overlap(background_aabb, fixture_aabb)
                for fixture_aabb in fixture_aabbs
            ), (
                scenario.scene_profile,
                background["stable_id"],
            )


def test_f2b_clean_r0_cleanup_does_not_change_accepted_f2c() -> None:
    backend = SourceMujocoBackend()
    scenario = compile_review_case(_case("F2c", 0))
    compiled = compile_source_model(
        scenario,
        source_dependency=backend.source_dependency,
        robocasa_dependency=backend.robocasa_dependency,
    )
    assert scenario.requires_real_robocasa is False
    assert compiled.removed_fixture_intersection_background_names == ()


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


def test_compiled_catalog_candidate_preserves_measured_runtime_clearance(
    kitchen_runtime,
) -> None:
    mujoco, _, _, compiled, manifest = kitchen_runtime
    row = manifest[0]

    assert row["collision_enabled"] is False
    assert row["swept_volume_clear"] is True
    assert row["fixture_intersection_clear"] is True
    assert "background_intersects_physical_fixture" not in row["blockers"]
    assert row["occlusion_validated"] is False
    assert row["admitted"] is False
    assert "rendered_occlusion_review_pending" in row["blockers"]
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


def test_planning_manifest_never_claims_uncomputed_runtime_clearance(
    kitchen_runtime,
) -> None:
    mujoco, backend, scenario, compiled, _ = kitchen_runtime
    planning_manifest = _robocasa_manifest(
        mujoco,
        compiled,
        scenario,
        backend.robocasa_dependency,
    )
    row = planning_manifest[0]

    assert row["runtime_clearance_evaluated"] is False
    assert row["runtime_clearance_sha256"] is None
    assert row["swept_volume_clear"] is None
    assert row["fixture_intersection_clear"] is None
    assert "runtime_clearance_pending" in row["blockers"]


def test_runtime_candidate_is_review_valid_but_not_release_admitted(
    kitchen_f1_runtime_manifest,
) -> None:
    manifest = kitchen_f1_runtime_manifest
    with pytest.raises(ValueError, match="not admitted"):
        validate_robocasa_asset_manifest(manifest)
    validate_robocasa_asset_manifest(
        manifest,
        required_asset_ids=("kitchen-toaster-review-candidate-v1",),
        allow_pending_render_review=True,
    )


@pytest.mark.parametrize("rollout_index", range(6))
def test_f2b_uses_scoped_complete_ramp_launch_cameras(
    rollout_index: int,
) -> None:
    scenario = compile_review_case(_case("F2b", rollout_index))
    assert scenario.motion_kind == "ramp_launch_pickup_interception"
    anchor = scenario.physical_target_position_m
    assert anchor is not None
    root = _camera_root()

    _repair_task_camera(root, scenario, height_offset=0.0)
    _add_secondary_camera(root, scenario, height_offset=0.0)

    main = root.find(".//camera[@name='main_camera']")
    secondary = root.find(".//camera[@name='secondary_camera']")
    assert main is not None
    assert secondary is not None
    _assert_camera_looks_at(
        main,
        position=(anchor[0] - 1.15, anchor[1] - 1.18, anchor[2] + 0.65),
        target=(0.15, 0.0, 0.30),
        fovy=64.0,
    )
    _assert_camera_looks_at(
        secondary,
        position=(1.52, -0.65, 1.35),
        target=(0.50, 0.0, 0.30),
        fovy=62.0,
    )


def test_f2b_ramp_launch_cameras_are_outcome_branch_invariant() -> None:
    nominal = compile_review_case(_case("F2b", 1))
    assert nominal.controller_target_position_m is not None
    negative = replace(
        nominal,
        branch_role="deterministic_negative_controller_timing",
        intended_outcome="failure",
        controller_target_position_m=(
            nominal.controller_target_position_m[0],
            nominal.controller_target_position_m[1] - 0.10,
            nominal.controller_target_position_m[2],
        ),
    )

    serialized: list[tuple[bytes, bytes]] = []
    for scenario in (nominal, negative):
        root = _camera_root()
        _repair_task_camera(root, scenario, height_offset=0.0)
        _add_secondary_camera(root, scenario, height_offset=0.0)
        main = root.find(".//camera[@name='main_camera']")
        secondary = root.find(".//camera[@name='secondary_camera']")
        assert main is not None
        assert secondary is not None
        serialized.append((ET.tostring(main), ET.tostring(secondary)))

    assert serialized[0] == serialized[1]


@pytest.mark.parametrize(
    ("rollout_index", "secondary_offset", "secondary_target_z", "secondary_fovy"),
    (
        (0, (-1.15, -1.18, 0.65), 0.17, 62.0),
        (1, (0.20, -1.10, 0.75), 0.02, 52.0),
    ),
)
def test_f2e_floor_to_wall_uses_scoped_overview_and_embodiment_event_camera(
    rollout_index: int,
    secondary_offset: tuple[float, float, float],
    secondary_target_z: float,
    secondary_fovy: float,
) -> None:
    scenario = compile_review_case(_case("F2e", rollout_index))
    assert scenario.task_variant == "floor_to_wall"
    anchor = scenario.physical_target_position_m
    assert anchor is not None
    root = _camera_root()

    _repair_task_camera(root, scenario, height_offset=0.0)
    _add_secondary_camera(root, scenario, height_offset=0.0)

    main = root.find(".//camera[@name='main_camera']")
    secondary = root.find(".//camera[@name='secondary_camera']")
    assert main is not None
    assert secondary is not None
    _assert_camera_looks_at(
        main,
        position=(-1.30, -2.20, 2.60),
        target=(-0.05, -0.20, 0.70),
        fovy=70.0,
    )
    secondary_position = tuple(
        anchor[index] + secondary_offset[index] for index in range(3)
    )
    _assert_camera_looks_at(
        secondary,
        position=secondary_position,
        target=(anchor[0], anchor[1], anchor[2] + secondary_target_z),
        fovy=secondary_fovy,
    )


def test_f2e_floor_to_wall_camera_is_outcome_branch_invariant() -> None:
    nominal = compile_review_case(_case("F2e", 1))
    assert nominal.controller_target_position_m is not None
    negative = replace(
        nominal,
        branch_role="deterministic_negative_controller_timing",
        intended_outcome="failure",
        controller_target_position_m=(
            nominal.controller_target_position_m[0],
            nominal.controller_target_position_m[1] - 0.10,
            nominal.controller_target_position_m[2],
        ),
    )

    serialized: list[tuple[bytes, bytes]] = []
    for scenario in (nominal, negative):
        root = _camera_root()
        _repair_task_camera(root, scenario, height_offset=0.0)
        _add_secondary_camera(root, scenario, height_offset=0.0)
        main = root.find(".//camera[@name='main_camera']")
        secondary = root.find(".//camera[@name='secondary_camera']")
        assert main is not None
        assert secondary is not None
        serialized.append((ET.tostring(main), ET.tostring(secondary)))

    assert serialized[0] == serialized[1]


@pytest.mark.parametrize(
    ("leaf_id", "rollout_index"),
    (("F2e", 2), ("F2c", 0)),
)
def test_f2e_camera_repair_does_not_change_other_variants_or_good_leaves(
    leaf_id: str,
    rollout_index: int,
) -> None:
    scenario = compile_review_case(_case(leaf_id, rollout_index))
    if leaf_id == "F2e":
        assert scenario.task_variant == "flight_to_table_bounce"
    anchor = scenario.physical_target_position_m
    assert anchor is not None
    root = _camera_root()

    _repair_task_camera(root, scenario, height_offset=0.0)
    _add_secondary_camera(root, scenario, height_offset=0.0)

    main = root.find(".//camera[@name='main_camera']")
    secondary = root.find(".//camera[@name='secondary_camera']")
    assert main is not None
    assert secondary is not None
    _assert_camera_looks_at(
        main,
        position=(anchor[0] - 1.15, anchor[1] - 1.18, anchor[2] + 0.65),
        target=(anchor[0], anchor[1], anchor[2] + 0.17),
        fovy=62.0,
    )
    _assert_camera_looks_at(
        secondary,
        position=(anchor[0] + 1.05, anchor[1] - 0.35, anchor[2] + 0.45),
        target=(anchor[0], anchor[1], anchor[2] + 0.05),
        fovy=55.0,
    )


def test_f2e_floor_to_wall_removes_only_swept_bulk_and_relocates_boundary() -> None:
    scenario = compile_review_case(_case("F2e", 1))
    root = _background_root()

    visual, task_volume = _remove_external_visual_work_surfaces(root, scenario)
    relocation = _relocate_external_visual_boundaries(root, scenario)

    world = root.find("worldbody")
    assert world is not None
    names = {str(geom.get("name")) for geom in world.findall("./geom")}
    assert visual == ()
    assert task_volume == tuple(sorted(task_volume))
    assert set(task_volume) == {
        "back_counter_base",
        "robot_table_leg_0",
        "robot_table_top",
    }
    assert "back_counter_base" not in names
    assert "back_counter_top" in names
    assert not any(name.startswith("robot_table_") for name in names)
    by_name = {str(row["name"]): row for row in relocation}
    assert set(by_name) == {"left_wall", "backsplash_left_return"}
    assert all(
        row["maximum_x_m"] == pytest.approx(-1.20)
        and row["reason"] == "clear_owned_f2e_floor_to_wall_negative_sweep"
        for row in by_name.values()
    )
    left_wall = world.find("./geom[@name='left_wall']")
    backsplash = world.find("./geom[@name='backsplash_left_return']")
    assert left_wall is not None
    assert backsplash is not None
    assert _camera_pose(left_wall)[0] == pytest.approx(-1.23)
    assert _camera_pose(backsplash)[0] == pytest.approx(-1.22)


@pytest.mark.parametrize(
    ("leaf_id", "rollout_index"),
    (("F2e", 2), ("F2c", 1)),
)
def test_f2e_background_repair_does_not_change_other_variants_or_good_leaves(
    leaf_id: str,
    rollout_index: int,
) -> None:
    scenario = compile_review_case(_case(leaf_id, rollout_index))
    root = _background_root()

    _, task_volume = _remove_external_visual_work_surfaces(root, scenario)
    relocation = _relocate_external_visual_boundaries(root, scenario)

    world = root.find("worldbody")
    assert world is not None
    names = {str(geom.get("name")) for geom in world.findall("./geom")}
    assert "back_counter_base" in names
    assert "back_counter_base" not in task_volume
    assert relocation == ()
    left_wall = world.find("./geom[@name='left_wall']")
    assert left_wall is not None
    assert _camera_pose(left_wall) == pytest.approx((-0.95, 0.1, 1.1))


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


def test_r1_physical_fixture_is_the_only_visible_work_surface(
    kitchen_runtime,
) -> None:
    _, _, scenario, compiled, _ = kitchen_runtime
    xml_root = ET.fromstring(compiled.xml)
    world = xml_root.find("worldbody")
    assert world is not None
    names = {
        str(geom.get("name") or "") for geom in world.findall("./geom")
    }

    assert compiled.removed_visual_work_surface_names == (
        "back_counter_top",
        "robot_table_top",
    )
    assert not set(compiled.removed_visual_work_surface_names) & names
    assert {surface.name for surface in scenario.surfaces} <= names
    # The lower cabinetry remains as anchored visual support/context; only its
    # conflicting top plane is removed.
    assert "back_counter_base" in names
    assert "robot_table_front" in names
    assert compiled.external_camera_metadata["main_camera"][
        "removed_visual_work_surface_names"
    ] == ["back_counter_top", "robot_table_top"]


def test_fixture_free_f1_r1_removes_entire_central_table_and_keeps_remote_context() -> None:
    case = next(
        case
        for case in build_review_suite_plan().cases
        if case.corpus_leaf_id == "F1a" and case.rollout_index == 2
    )
    backend = SourceMujocoBackend()
    scenario = compile_review_case(case)
    assert not scenario.surfaces
    compiled = compile_source_model(
        scenario,
        source_dependency=backend.source_dependency,
        robocasa_dependency=backend.robocasa_dependency,
    )
    xml_root = ET.fromstring(compiled.xml)
    names = {
        str(geom.get("name") or "")
        for geom in xml_root.findall("./worldbody/geom")
    }

    assert compiled.removed_visual_work_surface_names == ()
    assert compiled.removed_fixture_intersection_background_names == ()
    removed = compiled.removed_task_volume_background_names
    assert "robot_table_top" in removed
    assert removed == tuple(sorted(removed))
    assert not any(name.startswith("robot_table_") for name in names)
    assert not any(
        str(element.get("name") or "").startswith("robot_table_")
        for element in xml_root.iter()
    )
    assert "back_counter_top" in names
    assert "back_counter_base" in names
    assert compiled.robot_base_position_m == (0.0, 0.0, 0.0)
    assert compiled.robot_base_quaternion_wxyz == pytest.approx((1.0, 0.0, 0.0, 0.0))
    assert compiled.external_camera_metadata["main_camera"][
        "removed_task_volume_background_names"
    ] == list(removed)

    procedural = tuple(
        row
        for row in compiled.background_geom_descriptors
        if row["classification"] == "procedural"
    )
    catalog = tuple(
        row
        for row in compiled.background_geom_descriptors
        if row["classification"] == "catalog"
    )
    assert procedural
    assert catalog
    assert compiled.background_geom_exclusions == {"floor": "physical_room_support"}
    assert len({row["stable_id"] for row in compiled.background_geom_descriptors}) == len(
        compiled.background_geom_descriptors
    )
    assert all(
        int(compiled.model.geom_contype[int(row["geom_id"])]) == 0
        and int(compiled.model.geom_conaffinity[int(row["geom_id"])]) == 0
        for row in compiled.background_geom_descriptors
    )


@pytest.mark.parametrize("leaf_id", ("F2c", "F2d"))
def test_rebound_f2_r1_recipes_are_floor_rooted_with_owned_supported_fixtures(
    leaf_id: str,
) -> None:
    case = next(
        case
        for case in build_review_suite_plan().cases
        if case.corpus_leaf_id == leaf_id and case.rollout_index == 1
    )
    backend = SourceMujocoBackend()
    scenario = compile_review_case(case)
    compiled = compile_source_model(
        scenario,
        source_dependency=backend.source_dependency,
        robocasa_dependency=backend.robocasa_dependency,
    )
    xml_root = ET.fromstring(compiled.xml)
    names = {
        str(geom.get("name") or "")
        for geom in xml_root.findall("./worldbody/geom")
    }

    assert scenario.robot_base_position_m == (0.0, 0.0, 0.0)
    assert compiled.robot_base_position_m == (0.0, 0.0, 0.0)
    assert "robot_table_top" not in names
    assert not any(name.startswith("robot_table_") for name in names)
    assert compiled.removed_visual_work_surface_names == ()
    assert "robot_table_top" in compiled.removed_task_volume_background_names

    task_fixture = next(
        surface for surface in scenario.surfaces if surface.expected_task_contact
    )
    if leaf_id == "F2c":
        supports = tuple(
            surface
            for surface in scenario.surfaces
            if surface.role == "structural_support"
        )
        assert task_fixture.name == "owned_bounce_pad"
        assert len(supports) == 4
        for support in supports:
            assert support.supports_fixture_id == task_fixture.name
            assert support.position_m[2] - support.half_size_m[2] == pytest.approx(
                0.0, abs=1e-12
            )
            assert support.position_m[2] + support.half_size_m[2] == pytest.approx(
                task_fixture.position_m[2] - task_fixture.half_size_m[2],
                abs=1e-12,
            )
    else:
        assert task_fixture.name == "supported_wall_rebound_barrier"
        assert (
            task_fixture.position_m[2] - task_fixture.half_size_m[2]
            == pytest.approx(0.0, abs=1e-12)
        )


def test_f2a_shares_f1_floor_rooting_in_r1_scenes() -> None:
    # Counter-rooted F2a misses measurably fell 2.2 m out of frame and into
    # background furniture; F2a is a fixtureless ballistic interception and
    # uses F1's floor rooting so the rendered visibility QC geometry holds.
    case = next(
        case
        for case in build_review_suite_plan().cases
        if case.corpus_leaf_id == "F2a" and case.rollout_index == 1
    )
    scenario = compile_review_case(case)
    assert scenario.robot_base_position_m == (0.0, 0.0, 0.0)
    f1d = next(
        case
        for case in build_review_suite_plan().cases
        if case.corpus_leaf_id == "F1d" and case.rollout_index == 1
    )
    assert scenario.controller_target_position_m == compile_review_case(
        f1d
    ).controller_target_position_m


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
