#!/usr/bin/env python3
"""Render honest lateral projectile-catch previews with visible arm reaches.

This is a diagnostic renderer. It uses the canonical source_mujoco backend,
actuator-only controller, RoboCasa admission path, and sampled_preview initial
state contract, but it does not write a canonical or training-eligible dataset.
Runtime physics failures are recorded and never retried for a desired label.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import math
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import numpy as np
from PIL import Image

from dynamic_robot_dataset.backends.source_mujoco import SourceMujocoBackend
from dynamic_robot_dataset.common.hashing import sha256_json
from dynamic_robot_dataset.common.paths import atomic_write_json
from dynamic_robot_dataset.common.review_suite import build_review_suite_plan
from dynamic_robot_dataset.common.video_writer import VideoSpec, encode_video


SCHEMA_VERSION = "dynamic-robot-projectile-preview-pack/v1"
BASE_CASE_IDS = ("F1d-review-01", "F1d-review-02", "F1d-review-04")


def _appearance(asset_seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(int(asset_seed))
    external_variant_seed = int(rng.integers(0, 2**31, dtype=np.int64))
    color = tuple(float(value) for value in rng.uniform(0.15, 0.95, 3))
    return {
        "external_variant_seed": external_variant_seed,
        "ball_rgb": color,
        "lighting_intensity": float(rng.uniform(0.85, 1.18)),
        "floor_material_jitter": float(rng.uniform(-0.04, 0.04)),
    }


def _preview_case(base: Any, preview_index: int) -> dict[str, Any]:
    case = base.to_dict()
    case_id = f"F1d-lateral-projectile-{preview_index:02d}"
    case.update(
        {
            "branch_role": "nominal_success",
            "case_id": case_id,
            "counterfactual_branch_id": "nominal_success",
            "counterfactual_bundle_id": f"projectile-preview-{preview_index:02d}",
            "embodiment": "franka_hand",
            "episode_index": preview_index,
            "episode_uuid": str(uuid5(NAMESPACE_URL, case_id)),
            "initial_state_mode": "sampled_preview",
            "intended_outcome": "success",
            "task_variant": "mild_projectile_catch",
        }
    )
    case.pop("case_sha256", None)
    return case


def _strip(frames: list[np.ndarray], event_time_s: float, output: Path) -> None:
    last = len(frames) - 1
    indices = (
        max(0, min(last, round((event_time_s - 0.1) * 30))),
        max(0, min(last, round(event_time_s * 30))),
        max(0, min(last, round((event_time_s + 0.1) * 30))),
        max(0, min(last, round((event_time_s + 0.3) * 30))),
        last,
    )
    images = [Image.fromarray(frames[index], mode="RGB") for index in indices]
    strip = Image.new("RGB", (sum(image.width for image in images), images[0].height))
    x = 0
    for image in images:
        strip.paste(image, (x, 0))
        x += image.width
    strip.save(output)


def render(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    plan = build_review_suite_plan()
    by_id = {case.case_id: case for case in plan.cases}
    backend = SourceMujocoBackend()
    rows: list[dict[str, Any]] = []
    for index, base_id in enumerate(BASE_CASE_IDS):
        value = _preview_case(by_id[base_id], index)
        case_root = output / value["case_id"]
        case_root.mkdir()
        try:
            result = backend.run(value, render=True)
        except Exception as error:
            rows.append(
                {
                    "case_id": value["case_id"],
                    "status": "construction_error",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "retried": False,
                }
            )
            continue
        probes: dict[str, Any] = {}
        for camera in ("main", "secondary"):
            frames = list(result.frames_by_camera[camera])
            video_path = case_root / f"{camera}.mp4"
            probes[camera] = asdict(encode_video(frames, video_path, VideoSpec()))
            _strip(
                frames,
                float(result.scenario.ballistic_event_time_s),
                case_root / f"{camera}_events.png",
            )
        velocity = result.scenario.object_initial_linear_velocity_m_s
        failed_common_checks = sorted(
            name
            for name, passed in dict(result.physics_qc.get("checks") or {}).items()
            if passed is False
        )
        failed_task_checks = sorted(
            name
            for name, passed in dict(
                result.physics_qc.get("task_evidence_checks") or {}
            ).items()
            if passed is False
        )
        rows.append(
            {
                "case_id": value["case_id"],
                "status": "rendered",
                "scene_profile": result.scenario.scene_profile,
                "embodiment": result.scenario.embodiment,
                "initial_state_mode": result.scenario.initial_state_mode,
                "initial_position_m": result.scenario.object_initial_position_m,
                "initial_velocity_m_s": velocity,
                "initial_speed_m_s": math.sqrt(sum(item * item for item in velocity)),
                "horizontal_distance_m": result.scenario.initial_state_sampling_contract[
                    "sampled_horizontal_distance_m"
                ],
                "flight_time_s": result.scenario.ballistic_event_time_s,
                "physical_target_position_m": result.scenario.physical_target_position_m,
                "appearance": _appearance(result.scenario.rng_subseeds["assets"]),
                "task_success": result.outcome.get("task_success"),
                "actual_outcome": result.outcome.get("actual_outcome"),
                "physics_qc_pass": result.physics_qc.get("physics_qc_pass"),
                "physics_qc_failures": list(result.physics_qc.get("failures") or ()),
                "failed_common_checks": failed_common_checks,
                "failed_task_checks": failed_task_checks,
                "quality_flags": list(result.quality_flags),
                "videos": probes,
                "retried": False,
                "training_eligible": False,
            }
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "diagnostic_only": True,
        "training_eligible": False,
        "outcome_conditioned_retry": False,
        "source_review_plan_sha256": plan.plan_sha256,
        "cases": rows,
    }
    payload["manifest_sha256"] = sha256_json(payload)
    atomic_write_json(output / "manifest.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = render(args.output.resolve())
    print(payload["manifest_sha256"])
    return 0 if any(row["status"] == "rendered" for row in payload["cases"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
