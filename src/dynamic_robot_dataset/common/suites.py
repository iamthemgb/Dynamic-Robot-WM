"""Deterministic expansion and validation of acceptance and pilot suites."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from .hashing import stable_uint64


@dataclass(slots=True, frozen=True)
class SuiteCase:
    suite_name: str
    category_id: str
    case_index: int
    request_index: int
    repeat_index: int
    family: str
    subfamily: str
    branch: str
    physics_variant: str
    scene_style: str
    seed: int
    split_group_id: str
    counterfactual_bundle_id: str
    physics_counterfactual_family_id: str | None
    physics_intervention_fields: tuple[str, ...]
    backend: str
    robot_model: str
    views: tuple[str, ...]
    options: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["views"] = list(self.views)
        value["physics_intervention_fields"] = list(self.physics_intervention_fields)
        return value


def _stable_identifier(namespace: str, payload: Mapping[str, Any]) -> str:
    """Return a readable, deterministic identifier without process hash state."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"{namespace}-{hashlib.sha256(encoded).hexdigest()[:24]}"


def _physics_intervention_fields(variant: str) -> tuple[str, ...]:
    """Map a sweep member to every persisted field that may change.

    The semantic parameter and its MuJoCo implementation proxy are both
    declared. Otherwise a restitution or friction family would appear to
    change an undeclared field even though the solver/contact value is the
    mechanism that implements the named intervention.
    """

    prefix = variant.split("_", 1)[0]
    return {
        "gravity": ("gravity",),
        "friction": (
            "surface_dynamic_friction",
            "mujoco_surface_friction",
            "mujoco_object_friction",
        ),
        "restitution": (
            "effective_restitution_target",
            "mujoco_object_solref",
        ),
        "mass": ("mass",),
        "size": ("radius", "shape_half_extents"),
    }.get(prefix, ())


def _load_yaml(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"Configuration must contain a mapping: {path}")
    return dict(value)


def expand_suite(path: str | Path) -> list[SuiteCase]:
    """Expand a suite into stable, explicit branch/physics cases."""

    config = _load_yaml(path)
    if config.get("schema_version") != "dynamic-robot-suite/v1":
        raise ValueError("Unsupported suite schema_version")
    suite_name = str(config["name"])
    backend = str(config.get("backend", "diagnostic"))
    robot_model = str(config.get("robot_model", "abstract"))
    views = tuple(str(value) for value in config.get("views", ("main", "secondary")))
    result: list[SuiteCase] = []
    for category in config.get("categories", ()):
        category_id = str(category["id"])
        before = len(result)
        for request_index, request in enumerate(category.get("requests", ())):
            branches = tuple(str(value) for value in request.get("branches", ()))
            if not branches:
                raise ValueError(f"{category_id} request {request_index} has no branches")
            physics_variants = tuple(
                str(value) for value in request.get("physics_variants", ("nominal",))
            )
            repeats = int(request.get("repeats", 1))
            if repeats < 1 or not physics_variants:
                raise ValueError(f"{category_id} request {request_index} has invalid repeats/physics variants")
            known = {
                "family",
                "subfamily",
                "branches",
                "physics_variants",
                "repeats",
                "scene_style",
            }
            options = {key: value for key, value in request.items() if key not in known}
            for repeat in range(repeats):
                # One request/repeat is one physical scene. Branch labels and
                # physics sweep members must never perturb its seed; otherwise
                # counterfactual siblings would silently change appearance or
                # initial state.
                group_payload = {
                    "suite": suite_name,
                    "category": category_id,
                    "request": request_index,
                    "repeat": repeat,
                }
                seed = stable_uint64(
                    group_payload,
                    namespace="dynamic-robot-suite-group-seed-v1",
                ) & 0x7FFF_FFFF
                split_group_id = _stable_identifier("split", group_payload)
                for physics_variant in physics_variants:
                    for branch in branches:
                        case_index = len(result)
                        bundle_id = _stable_identifier(
                            "action-bundle",
                            {**group_payload, "physics_variant": physics_variant},
                        )
                        physics_family_id = (
                            _stable_identifier(
                                "physics-family",
                                {**group_payload, "branch": branch},
                            )
                            if len(physics_variants) > 1
                            else None
                        )
                        result.append(
                            SuiteCase(
                                suite_name=suite_name,
                                category_id=category_id,
                                case_index=case_index,
                                request_index=request_index,
                                repeat_index=repeat,
                                family=str(request["family"]),
                                subfamily=str(request.get("subfamily", "default")),
                                branch=branch,
                                physics_variant=physics_variant,
                                scene_style=str(request.get("scene_style", "clean_franka_lab")),
                                seed=seed,
                                split_group_id=split_group_id,
                                counterfactual_bundle_id=bundle_id,
                                physics_counterfactual_family_id=physics_family_id,
                                physics_intervention_fields=_physics_intervention_fields(
                                    physics_variant
                                ),
                                backend=backend,
                                robot_model=robot_model,
                                views=views,
                                options=options,
                            )
                        )
        measured = len(result) - before
        expected = int(category["expected_branches"])
        if measured != expected:
            raise ValueError(
                f"suite category {category_id} expands to {measured} cases, expected {expected}"
            )
    expected_total = int(config["expected_branch_count"])
    if len(result) != expected_total:
        raise ValueError(f"suite expands to {len(result)} cases, expected {expected_total}")
    styles = {case.scene_style for case in result}
    minimum_styles = int((config.get("requirements") or {}).get("minimum_background_styles", 0))
    if len(styles) < minimum_styles:
        raise ValueError(f"suite uses {len(styles)} styles; at least {minimum_styles} required")
    return result


def load_pilot_plan(path: str | Path) -> dict[str, Any]:
    """Load and validate a non-submitting unique-hour pilot contract."""

    config = _load_yaml(path)
    if config.get("schema_version") != "dynamic-robot-pilot/v1":
        raise ValueError("Unsupported pilot schema_version")
    if config.get("submit") is not False:
        raise ValueError("Pilot contracts in this repository must default to submit=false")
    allocations = list(config.get("allocations", ()))
    if not allocations:
        raise ValueError("Pilot plan has no allocations")
    identifiers = [str(value["id"]) for value in allocations]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Pilot allocation IDs must be unique")
    allocated = sum(float(value["unique_hours"]) for value in allocations)
    target = float(config["target_unique_release_hours"])
    if abs(allocated - target) > 1e-9:
        raise ValueError(f"pilot allocations sum to {allocated} h, target is {target} h")
    return config
