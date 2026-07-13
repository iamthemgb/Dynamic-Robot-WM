"""Exact 120-branch, non-production smoke orchestration.

The renderer in this module visualizes the saved state with calibrated pinhole
cameras.  It is intentionally labelled ``diagnostic_state_renderer`` and every
episode carries a ``smoke_diagnostic_renderer`` quality flag, so these format
fixtures can never enter the default training manifest by accident.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml
from PIL import Image, ImageDraw

from .common.cameras import CameraCalibration, invert_rigid_transform
from .common.contacts import normalize_assistance, select_task_event_time
from .common.episode_writer import EpisodeWriter
from .common.hashing import sha256_json
from .common.paths import atomic_write_json
from .common.provenance import GenerationProvenance, environment_hash, get_git_commit
from .common.qc import validate_dataset
from .common.schema import (
    DatasetInfo,
    DynamicsMode,
    EpisodeRecord,
    LabelStatus,
    NamedFeature,
    PhysicsMetadata,
    PhysicsValue,
    PhysicsValueKind,
    ReleaseTier,
    Split,
    TimeBase,
)
from .common.splits import SplitAssigner
from .common.video_writer import VideoSpec
from .common.wan_export import export_wan
from .families.base import SimulationResult
from .families.smoke import EXPECTED_SMOKE_COUNTS, plan_smoke
from .families import get_family


SMOKE_RENDERER = "dynamic_robot_dataset.diagnostic_state_renderer/v1"
CAMERA_NAMES = ("observation.images.main", "observation.images.secondary")


def _normalize(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if length <= 1e-12:
        raise ValueError("Cannot normalize a zero vector")
    return vector / length


def _camera(
    name: str,
    position: Sequence[float],
    target: Sequence[float],
    *,
    focal_px: float = 360.0,
) -> CameraCalibration:
    """Build a right-handed, +Z-world pinhole camera with image Y downward."""

    eye = np.asarray(position, dtype=np.float64)
    aim = np.asarray(target, dtype=np.float64)
    forward = _normalize(aim - eye)
    right = _normalize(np.cross(forward, np.asarray((0.0, 0.0, 1.0))))
    down = _normalize(np.cross(forward, right))
    transform = np.eye(4, dtype=np.float64)
    transform[0, :3] = right
    transform[1, :3] = down
    transform[2, :3] = forward
    transform[:3, 3] = -transform[:3, :3] @ eye
    world_to_camera = tuple(float(value) for value in transform.reshape(-1))
    camera_to_world = invert_rigid_transform(world_to_camera)
    calibration = CameraCalibration(
        camera_name=name,
        intrinsic_matrix=(focal_px, 0.0, 416.0, 0.0, focal_px, 240.0, 0.0, 0.0, 1.0),
        world_to_camera=world_to_camera,
        camera_to_world=camera_to_world,
        width=832,
        height=480,
        fps=30.0,
        near_m=0.01,
        far_m=50.0,
        renderer=SMOKE_RENDERER,
    )
    calibration.validate()
    return calibration


def smoke_cameras() -> dict[str, CameraCalibration]:
    """Return two fixed, synchronized cameras covering all smoke task bounds."""

    return {
        CAMERA_NAMES[0]: _camera(CAMERA_NAMES[0], (4.2, -5.2, 3.0), (0.8, 0.0, 0.55)),
        CAMERA_NAMES[1]: _camera(CAMERA_NAMES[1], (0.8, -4.2, 4.8), (0.8, 0.0, 0.35), focal_px=330.0),
    }


def render_result(
    *, result: SimulationResult, episode_index: int, views: Sequence[str] = ("main", "secondary")
) -> dict[str, Any]:
    """CLI-compatible diagnostic renderer, always quarantined from release."""

    del episode_index  # Identity is encoded by the writer, not burned into pixels.
    calibrations = smoke_cameras()
    selected: dict[str, CameraCalibration] = {}
    for view in views:
        name = str(view)
        if not name.startswith("observation.images."):
            name = f"observation.images.{name}"
        if name not in calibrations:
            raise ValueError(f"Diagnostic renderer has no camera named {view!r}")
        selected[name] = calibrations[name]
    return {
        "videos": {
            name: DiagnosticRenderer(result, calibration).frames()
            for name, calibration in selected.items()
        },
        "renderer": SMOKE_RENDERER,
        "production_eligible": False,
        "quality_flags": ["smoke_diagnostic_renderer"],
        "camera_calibrations": [
            {"camera_id": name, **calibration.to_dict()}
            for name, calibration in selected.items()
        ],
    }


_STYLE_COLORS: dict[str, tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]] = {
    "clean_franka_lab": ((226, 235, 241), (191, 205, 214), (116, 132, 142)),
    "robocasa_kitchen_tabletop": ((242, 225, 197), (194, 154, 108), (112, 79, 52)),
    "robotwin_cluttered_tabletop": ((105, 119, 130), (151, 141, 123), (58, 68, 76)),
    # Backward-compatible aliases for pre-contract development runs.
    "neutral": ((226, 235, 241), (191, 205, 214), (116, 132, 142)),
    "kitchen": ((242, 225, 197), (194, 154, 108), (112, 79, 52)),
    "industrial": ((105, 119, 130), (151, 141, 123), (58, 68, 76)),
}


@dataclass(slots=True)
class DiagnosticRenderer:
    """Calibrated state visualization used only for pipeline smoke tests."""

    result: SimulationResult
    camera: CameraCalibration

    def _project(self, point: Sequence[float]) -> tuple[int, int] | None:
        try:
            x, y, _ = self.camera.project_world(point)
        except ValueError:
            return None
        if not (-200 <= x <= self.camera.width + 200 and -200 <= y <= self.camera.height + 200):
            return None
        return int(round(x)), int(round(y))

    def _base(self) -> Image.Image:
        wall, table, accent = _STYLE_COLORS.get(
            self.result.plan.scene_style, _STYLE_COLORS["clean_franka_lab"]
        )
        image = Image.new("RGB", (self.camera.width, self.camera.height), wall)
        draw = ImageDraw.Draw(image)
        corners = [self._project(point) for point in ((-1.0, -0.8, 0.0), (5.5, -0.8, 0.0), (5.5, 0.8, 0.0), (-1.0, 0.8, 0.0))]
        if all(point is not None for point in corners):
            draw.polygon(corners, fill=table, outline=accent, width=3)  # type: ignore[arg-type]
        for x in np.linspace(-0.8, 5.2, 9):
            endpoints = (self._project((x, -0.7, 0.002)), self._project((x, 0.7, 0.002)))
            if all(point is not None for point in endpoints):
                draw.line(endpoints, fill=accent, width=1)  # type: ignore[arg-type]
        return image

    def _line3(self, draw: ImageDraw.ImageDraw, points: Sequence[Sequence[float]], *, fill: tuple[int, int, int], width: int) -> None:
        projected = [self._project(point) for point in points]
        segments: list[tuple[int, int]] = []
        for point in projected:
            if point is None:
                if len(segments) >= 2:
                    draw.line(segments, fill=fill, width=width, joint="curve")
                segments = []
            else:
                segments.append(point)
        if len(segments) >= 2:
            draw.line(segments, fill=fill, width=width, joint="curve")

    def _sphere(self, draw: ImageDraw.ImageDraw, position: Sequence[float], *, color: tuple[int, int, int], radius: int = 11) -> None:
        point = self._project(position)
        if point is None:
            return
        x, y = point
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=(25, 30, 35), width=2)

    def frame(self, index: int) -> np.ndarray:
        state = self.result.states[index]
        image = self._base()
        draw = ImageDraw.Draw(image)

        if "cloth.vertices" in state:
            vertices = state["cloth.vertices"]
            side = int(round(math.sqrt(len(vertices))))
            for row in range(side):
                self._line3(draw, vertices[row * side : (row + 1) * side], fill=(38, 102, 188), width=4)
            for column in range(side):
                self._line3(draw, vertices[column::side], fill=(38, 102, 188), width=3)
        if "rope.points" in state:
            self._line3(draw, state["rope.points"], fill=(244, 176, 45), width=8)
            self._sphere(draw, state["rope.points"][-1], color=(220, 66, 61), radius=8)
        if "soft_body.centroid" in state:
            deformation = float(state.get("soft_body.deformation_metric", 0.0))
            self._sphere(
                draw,
                state["soft_body.centroid"],
                color=(132, 79, 186),
                radius=max(12, int(round(22 + 150 * deformation))),
            )
        # Draw tools first so the target remains visible at the critical contact
        # rather than being hidden by the schematic receptacle/paddle glyph.
        for key in ("robot.receptacle_position", "robot.tool_position", "robot.paddle_position"):
            if key in state:
                point = self._project(state[key])
                if point is not None:
                    x, y = point
                    draw.rounded_rectangle((x - 32, y - 11, x + 32, y + 11), radius=6, fill=(47, 186, 188), outline=(20, 65, 70), width=3)
        for key in ("object.position", "legacy.proxy_position"):
            if key in state:
                self._sphere(draw, state[key], color=(239, 91, 64), radius=18)

        # A neutral physical-clock marker makes temporal progress visible in
        # diagnostic clips without exposing branch intent or measured labels.
        progress = index / max(1, len(self.result.states) - 1)
        draw.rectangle((24, 454, 808, 461), fill=(80, 84, 88))
        draw.rectangle((24, 454, 24 + int(round(784 * progress)), 461), fill=(242, 192, 58))
        return np.asarray(image, dtype=np.uint8)

    def frames(self) -> Iterable[np.ndarray]:
        for index in range(len(self.result.states)):
            yield self.frame(index)


def _physics_metadata(result: SimulationResult) -> PhysicsMetadata:
    kinds = {
        "physical": PhysicsValueKind.PHYSICAL,
        "calibrated_effective": PhysicsValueKind.CALIBRATED_EFFECTIVE,
        "proxy": PhysicsValueKind.SIMULATOR_PROXY,
        "simulator_proxy": PhysicsValueKind.SIMULATOR_PROXY,
        "unknown": PhysicsValueKind.UNKNOWN,
        "not_implemented": PhysicsValueKind.NOT_IMPLEMENTED,
    }
    parameters: dict[str, PhysicsValue] = {}
    gravity = (0.0, 0.0, -9.81)
    gravity_valid = False
    timestep: float | None = None
    substeps: int | None = None
    for name, raw in result.plan.physics.items():
        field = dict(raw) if isinstance(raw, Mapping) else {
            "value": raw,
            "unit": "unknown",
            "valid": raw is not None,
            "implemented": True,
            "interpretation": "physical",
        }
        value = field.get("value")
        valid = bool(field.get("valid", value is not None))
        implemented = bool(field.get("implemented", True))
        interpretation = str(field.get("interpretation", "physical"))
        kind = kinds.get(interpretation, PhysicsValueKind.UNKNOWN)
        if not implemented and interpretation == "not_implemented":
            kind = PhysicsValueKind.NOT_IMPLEMENTED
        # A few legacy proxy flags describe whether source physics exists. They
        # are known booleans despite the source's generic "unknown" tag; do not
        # misrepresent a known value as an unknown physical parameter.
        if kind == PhysicsValueKind.UNKNOWN and valid:
            kind = PhysicsValueKind.SIMULATOR_PROXY
        parameters[name] = PhysicsValue(
            name=name,
            value=value,
            unit=str(field.get("unit", "unknown")),
            valid=valid,
            implemented=implemented,
            kind=kind,
            source="family_adapter_config",
        )
        if name == "gravity" and valid and isinstance(value, Sequence):
            gravity = tuple(float(component) for component in value)  # type: ignore[assignment]
            gravity_valid = True
        elif name == "simulation_timestep" and valid:
            timestep = float(value)
        elif name == "substeps" and valid:
            substeps = int(round(float(value)))
    return PhysicsMetadata(
        parameters=parameters,
        gravity_world_m_s2=gravity,
        gravity_valid=gravity_valid,
        simulation_timestep_s=timestep,
        substeps=substeps,
        solver_settings={"solver": result.simulator.get("name", "unknown")},
    )


def _label_status(value: str) -> LabelStatus:
    return LabelStatus.VERIFIED if value in {"verified", "verified_objective"} else (
        LabelStatus.PROXY if value == "proxy" else LabelStatus.UNVERIFIED
    )


def _record(result: SimulationResult, episode_index: int, git_commit: str) -> EpisodeRecord:
    status = _label_status(result.outcome.label_status)
    dynamics = DynamicsMode(result.dynamics_mode)
    release = ReleaseTier(result.release_tier)
    if dynamics == DynamicsMode.SCRIPTED_MOTION:
        release = ReleaseTier.SCRIPTED_MOTION
    elif dynamics == DynamicsMode.ASSISTED_CONTACT:
        release = ReleaseTier.ASSISTED_CONTACT
    elif status != LabelStatus.VERIFIED and release == ReleaseTier.FREE_CONTACT:
        release = ReleaseTier.UNVERIFIED
    event_time = select_task_event_time(result.contacts)
    return EpisodeRecord(
        episode_uuid=result.plan.episode_uuid,
        episode_index=episode_index,
        counterfactual_bundle_id=result.plan.counterfactual_bundle_id,
        physics_counterfactual_family_id=result.plan.physics_counterfactual_family_id,
        split_group_id=result.plan.split_group_id,
        scene_seed=result.plan.scene_seed,
        branch_seed=result.plan.branch_seed,
        family=result.plan.family,
        subfamily=result.plan.subfamily,
        variant=result.plan.variant,
        robot_model=result.plan.robot_model,
        tool_type=result.plan.tool_type,
        action_mode=(
            "task_space_position_absolute"
            if result.plan.family in {"falling_catch", "rolling_interception", "projectile_rebound"}
            else "family_specific_scripted_proxy"
        ),
        intended_branch=result.plan.intended_branch,
        actual_outcome=result.actual_outcome,
        task_success=result.outcome.task_success,
        partial_success_score=result.outcome.partial_success_score,
        failure_mode=result.outcome.failure_mode,
        label_confidence=result.outcome.label_confidence,
        label_status=status,
        dynamics_mode=dynamics,
        release_tier=release,
        physics_qc_pass=bool(result.physics_qc.get("physics_qc_pass", False)),
        source_generator=result.plan.source_generator,
        source_generator_version=result.plan.source_generator_version,
        generator_git_commit=git_commit,
        config_hash=result.plan.config_hash,
        simulator_name=str(result.simulator.get("name", "unknown")),
        simulator_version=str(result.simulator.get("version", "unknown")),
        renderer=SMOKE_RENDERER,
        event_time_s=event_time,
        physics=_physics_metadata(result),
        assistance=normalize_assistance(result.assistance),
        objective_metrics=dict(result.outcome.metrics),
        randomization={
            "scene_style": result.plan.scene_style,
            "randomization_level": result.plan.randomization_level,
            "visual_seed": result.plan.scene_parameters.get("visual_seed", result.plan.scene_seed),
        },
        quality_flags=["smoke_diagnostic_renderer"],
        extras={
            "family_plan_config_hash": result.plan.config_hash,
            "action_hash": result.plan.action_hash,
            "physics_hash": result.plan.physics_hash,
            "counterfactual_invariant_hash": result.plan.invariant_hash,
            # Split grouping consumes these identities before media are
            # written.  Excluding timestamps and assistance/QC flags makes the
            # hashes describe physical state rather than branch bookkeeping.
            "initial_state_hash": sha256_json(
                {
                    key: value
                    for key, value in result.states[0].items()
                    if key != "timestamp"
                    and not key.startswith("assistance.")
                    and not key.endswith("_flag")
                }
            ),
            "trajectory_hash": sha256_json(
                [
                    {
                        key: value
                        for key, value in state.items()
                        if key != "timestamp"
                        and not key.startswith("assistance.")
                        and not key.endswith("_flag")
                    }
                    for state in result.states
                ]
            ),
            "physics_variant": result.plan.physics_variant,
            "scene_parameters": dict(result.plan.scene_parameters),
            "branch_parameters": dict(result.plan.branch_parameters),
            "physics_qc": dict(result.physics_qc),
            "simulator": dict(result.simulator),
            "notes": list(result.notes),
            "training_default_exclusion": "diagnostic smoke renderer; regenerate with a validated native renderer",
            "event_time_semantics": "first_non_fixture_task_contact",
        },
    )


def _frame_rows(result: SimulationResult, episode_index: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    contact_times = [float(event["timestamp"]) for event in result.contacts]
    half_frame = 0.5 / result.plan.rates_hz["video"]
    for frame_index, (timestamp, state) in enumerate(zip(result.frame_times_s, result.states)):
        action = result.actions[min(frame_index, len(result.actions) - 1)] if result.actions else {}
        assistance_active = bool(state.get("assistance.active", False))
        mechanism_ids = [
            str(mechanism["mechanism_id"])
            for mechanism in result.assistance.get("mechanisms", ())
            if any(
                float(interval["start_time_s"]) <= float(timestamp)
                and (
                    interval.get("end_time_s") is None
                    or float(timestamp) <= float(interval["end_time_s"])
                )
                for interval in mechanism.get("activation_intervals", ())
            )
        ]
        row: dict[str, Any] = {
            "episode_index": episode_index,
            "frame_index": frame_index,
            "video_frame_index": frame_index,
            "timestamp": float(timestamp),
            "contact.active": any(abs(float(timestamp) - event_time) <= half_frame for event_time in contact_times),
            "event.contact": any(abs(float(timestamp) - event_time) <= half_frame for event_time in contact_times),
            "assistance.active": assistance_active,
            "assistance.assisted_grasp": assistance_active and bool(result.assistance.get("assisted_grasp", False)),
            "assistance.assisted_retention": assistance_active and bool(result.assistance.get("assisted_retention", False)),
            "assistance.equality_constraint_active": assistance_active and bool(result.assistance.get("equality_constraint_active", False)),
            "assistance.latch_active": assistance_active and bool(result.assistance.get("latch_active", False)),
            "assistance.mechanism_ids": mechanism_ids,
        }
        row.update({key: value for key, value in state.items() if key != "timestamp"})
        row.update({f"action.{key}": value for key, value in action.items() if key != "timestamp"})
        rows.append(row)
    return rows


def _high_rate_rows(result: SimulationResult, episode_index: int) -> list[dict[str, Any]]:
    return [{"episode_index": episode_index, **dict(row)} for row in result.high_rate_states]


def _event_rows(result: SimulationResult, episode_index: int) -> list[dict[str, Any]]:
    return [{"episode_index": episode_index, **dict(row)} for row in result.contacts]


def _task_rows(results: Sequence[SimulationResult]) -> list[dict[str, Any]]:
    values = sorted({(result.plan.family, result.plan.subfamily) for result in results})
    return [
        {"task_index": index, "family": family, "subfamily": subfamily}
        for index, (family, subfamily) in enumerate(values)
    ]


def _validate_spec(config: Mapping[str, Any]) -> None:
    counts = {key: int(value) for key, value in dict(config.get("counts", {})).items()}
    if counts != EXPECTED_SMOKE_COUNTS:
        raise ValueError(f"Smoke counts differ from implemented matrix: {counts} != {EXPECTED_SMOKE_COUNTS}")
    total = sum(counts.values())
    maximum = int(config.get("max_episode_branches", 0))
    if total != 120 or maximum > 200 or total > maximum:
        raise ValueError(f"Smoke suite must be exactly 120 branches with maximum <=200; got total={total}, max={maximum}")
    if tuple(config.get("views", ())) != ("main", "secondary"):
        raise ValueError("Smoke contract requires views [main, secondary]")
    if len(set(config.get("background_styles", ()))) < 3:
        raise ValueError("Smoke contract requires at least three background styles")
    required_outcomes = set(config.get("validation", {}).get("require_objective_success_and_failure", ()))
    expected_outcome_families = {"falling_catch", "rolling_interception", "projectile_rebound", "cloth", "rope"}
    if not expected_outcome_families.issubset(required_outcomes):
        raise ValueError(
            "Smoke validation must require objective success/failure examples for rigid tasks, cloth, and rope"
        )
    if not bool(config.get("validation", {}).get("require_objective_cloth_metric", False)):
        raise ValueError("Smoke validation must require at least one objective cloth metric")
    if not bool(config.get("validation", {}).get("run_wan_roundtrip", False)):
        raise ValueError("Smoke validation must exercise the Wan export round trip")


def run_smoke_suite(
    config_path: str | Path,
    output: str | Path,
    *,
    seed: int = 0,
    resume: bool = False,
    deep_video_checks: bool = True,
) -> dict[str, Any]:
    """Generate, finalize, split, and validate the exact smoke dataset."""

    config_file = Path(config_path).resolve(strict=True)
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    _validate_spec(config)
    plans = plan_smoke(seed)
    results = [get_family(plan.family).simulate(plan) for plan in plans]
    counts = Counter(result.plan.family for result in results)
    if dict(counts) != EXPECTED_SMOKE_COUNTS:
        raise AssertionError(f"Simulated smoke counts drifted: {dict(counts)}")

    repo_root = Path(__file__).resolve().parents[2]
    commit = get_git_commit(repo_root)
    resolved_config = {
        "schema_version": config["schema_version"],
        "config_path": str(config_file),
        "config": config,
        "seed": seed,
        "generator_git_commit": commit,
        "renderer": SMOKE_RENDERER,
        "non_production": True,
    }
    writer = EpisodeWriter(
        output,
        resolved_config,
        resume=resume,
        video_spec=VideoSpec(width=832, height=480, fps_num=30, fps_den=1, preset="veryfast"),
    )
    task_rows = _task_rows(results)
    task_index_by_key = {
        (str(row["family"]), str(row["subfamily"])): int(row["task_index"])
        for row in task_rows
    }
    records = [_record(result, index, commit) for index, result in enumerate(results)]
    for record in records:
        record.task_index = task_index_by_key[(record.family, record.subfamily)]
    assignments = SplitAssigner(seed=seed).assign(records)
    assignment_by_uuid = {assignment.episode_uuid: assignment for assignment in assignments}
    for record in records:
        assignment = assignment_by_uuid[record.episode_uuid]
        record.split = Split(assignment.split)
        record.extras["planned_split_group_id"] = record.split_group_id
        record.split_group_id = assignment.split_group_id

    cameras = smoke_cameras()
    committed: list[EpisodeRecord] = []
    for result, record in zip(results, records):
        committed.append(
            writer.write_episode(
                record,
                frame_rows=_frame_rows(result, record.episode_index),
                videos={
                    name: DiagnosticRenderer(result, calibration).frames()
                    for name, calibration in cameras.items()
                },
                high_rate_rows=_high_rate_rows(result, record.episode_index),
                event_rows=_event_rows(result, record.episode_index),
                object_state_rows=(),
            )
        )

    info = DatasetInfo(
        name=str(config.get("name", "canonical_smoke_120")),
        description="Non-production exact 120-branch dynamic-robot pipeline smoke suite",
        time_base=TimeBase(sim_hz=240.0, control_hz=60.0, video_hz=30.0),
        state_features=[
            NamedFeature("primary_target_state", "family_specific_named_columns", "SI", description="See per-episode frame Parquet column names"),
        ],
        action_features=[
            NamedFeature("command", "family_specific_named_columns", "declared_per_column", description="Every action column is prefixed action.command"),
        ],
        camera_names=list(CAMERA_NAMES),
        generator_version="0.1.0",
        extras={
            "smoke_only": True,
            "default_training_manifest_episode_count": 0,
            "exclusion_reason": "diagnostic_state_renderer",
            "persistent_physics_transient_state_action_separate": True,
        },
    )
    provenance = GenerationProvenance(
        source_generator="dynamic_robot_dataset.smoke_runner",
        source_generator_version="0.1.0",
        generator_git_commit=commit,
        config_hash=writer.config_hash,
        simulator_name="family_specific; see episode metadata",
        simulator_version="family_specific",
        renderer=SMOKE_RENDERER,
        command=sys.argv,
        environment_hash=environment_hash(lockfiles=[repo_root / "uv.lock"]),
    )
    camera_rows = [
        {"camera_id": name, **calibration.to_dict()}
        for name, calibration in cameras.items()
    ]
    split_rows = [
        {
            "episode_uuid": assignment.episode_uuid,
            "episode_index": assignment.episode_index,
            "split_group_id": assignment.split_group_id,
            "split": assignment.split,
        }
        for assignment in assignments
    ]
    # The writer's marker-last metadata transaction is the only completion
    # authority. Presence of six linked files is not enough: a crash can occur
    # immediately before ``meta/.complete.json`` is published.
    writer.finalize(
        info,
        tasks=task_rows,
        cameras=camera_rows,
        provenance=[provenance.to_dict()],
        splits=split_rows,
    )
    qc_report_path = Path(output) / "qc" / "dataset_report.json"
    report = validate_dataset(
        output,
        deep_video_checks=deep_video_checks,
        write_reports=not qc_report_path.exists(),
    )
    wan_output = Path(output).resolve().parent / f"{Path(output).name}_wan"
    if not wan_output.exists():
        wan_summary = export_wan(output, wan_output, include_nonrelease=True).to_dict()
    else:
        wan_summary = json.loads((wan_output / "export_summary.json").read_text(encoding="utf-8"))
    if wan_summary["episode_count"] != 120 or wan_summary["video_count"] != 240:
        raise RuntimeError(f"Wan smoke round trip is incomplete: {wan_summary}")

    contact_sheet_dir = Path(output) / "qc" / "contact_sheets"
    if not (contact_sheet_dir / "samples.json").exists():
        subprocess.run(
            [
                sys.executable,
                str(repo_root / "tools" / "make_contact_sheets.py"),
                "--dataset-root",
                str(Path(output).resolve()),
                "--output",
                str(contact_sheet_dir.resolve()),
            ],
            check=True,
        )

    summary = {
        "schema_version": "dynamic-robot-smoke-result/v1",
        "output": str(Path(output).resolve()),
        "episode_count": len(committed),
        "family_counts": dict(sorted(counts.items())),
        "background_style_counts": dict(sorted(Counter(result.plan.scene_style for result in results).items())),
        "objective_outcomes": {
            family: dict(sorted(Counter(result.actual_outcome for result in results if result.plan.family == family).items()))
            for family in EXPECTED_SMOKE_COUNTS
        },
        "physics_qc_pass_counts": {
            family: sum(
                bool(result.physics_qc.get("physics_qc_pass", False))
                for result in results
                if result.plan.family == family
            )
            for family in EXPECTED_SMOKE_COUNTS
        },
        "release_eligible_count": sum(record.release_eligible for record in committed),
        "qc_passed": report.passed and all(episode.passed for episode in report.episodes),
        "qc_release_policy_passed": report.passed,
        "qc_failed_episode_count": sum(not episode.passed for episode in report.episodes),
        "qc_global_failures": report.global_failures,
        "wan_export": wan_summary,
        "contact_sheet_ledger": "qc/contact_sheets/samples.json",
        "note": "All episodes are excluded from default training because the smoke renderer is diagnostic.",
    }
    result_path = Path(output) / "qc" / "smoke_summary.json"
    if not result_path.exists():
        atomic_write_json(result_path, summary)
    return summary


__all__ = ["DiagnosticRenderer", "render_result", "run_smoke_suite", "smoke_cameras"]
