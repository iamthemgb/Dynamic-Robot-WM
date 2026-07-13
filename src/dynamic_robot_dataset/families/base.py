"""Stable family-adapter contract used by all cleaned generators.

The adapters in this package intentionally have no dependency on the dataset
writer.  A simulator returns plain, JSON-serialisable records which the common
writer can convert to Arrow rows and videos.  Keeping this boundary small also
makes the physics and label logic testable without a renderer or GPU.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import uuid

from ..common.schema import validate_failure_mode
from ..common.randomization import RandomizationCatalog, RandomizationPlanner


SCHEMA_VERSION = "dynamic-robot-family/v1"
_UUID_NAMESPACE = uuid.UUID("53c06143-b31e-46f9-9d69-42fa1fe4e901")

STANDARD_BRANCHES = (
    "success_seeking",
    "near_miss",
    "contact_failure",
    "no_op",
)

_BRANCH_ALIASES = {
    "success": "success_seeking",
    "success-seeking": "success_seeking",
    "nominal": "success_seeking",
    "near-miss": "near_miss",
    "contact-failure": "contact_failure",
    "bad-action": "bad_action",
    "noop": "no_op",
    "no-op": "no_op",
    "no_op": "no_op",
}


def _jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__dataclass_fields__"):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite values are not valid generator metadata")
        # Stable hashes should not depend on insignificant platform formatting.
        return float(f"{value:.12g}")
    return value


def stable_hash(value: Any) -> str:
    payload = json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def deterministic_seed(*parts: Any) -> int:
    return int(stable_hash(parts)[:16], 16) & 0x7FFF_FFFF


def deterministic_uuid(*parts: Any) -> str:
    return str(uuid.uuid5(_UUID_NAMESPACE, stable_hash(parts)))


def normalize_branch(branch: str) -> str:
    normalized = branch.strip().lower().replace(" ", "_")
    return _BRANCH_ALIASES.get(normalized, normalized)


def physics_field(
    value: float | Sequence[float] | None,
    unit: str,
    *,
    valid: bool = True,
    implemented: bool = True,
    interpretation: str = "physical",
) -> dict[str, Any]:
    """Create a named, masked physics field.

    Unknown values are represented explicitly, never as invented numeric
    defaults.  ``interpretation`` distinguishes physical quantities from
    calibrated-effective or proxy values.
    """

    if interpretation == "proxy":
        interpretation = "simulator_proxy"
    if value is None:
        valid = False
    return {
        "value": _jsonable(value),
        "unit": unit,
        "valid": bool(valid),
        "implemented": bool(implemented),
        "interpretation": interpretation,
    }


def physics_value(fields: Mapping[str, Any], name: str) -> Any:
    field_value = fields[name]
    if isinstance(field_value, Mapping) and "value" in field_value:
        if not field_value.get("valid", True):
            raise ValueError(f"physics field {name!r} is unknown")
        return field_value["value"]
    return field_value


@dataclass(frozen=True)
class GenerationRequest:
    family: str
    subfamily: str = "default"
    variant: str = "clean_v1"
    robot_model: str = "abstract"
    tool_type: str = "task_default"
    num_bundles: int = 1
    branches: tuple[str, ...] = STANDARD_BRANCHES
    views: tuple[str, ...] = ("main", "secondary")
    seed: int = 0
    randomization_level: str = "R0"
    scene_style: str = "neutral"
    sim_hz: int = 240
    control_hz: int = 60
    video_hz: int = 30
    duration_s: float | None = None
    physics_sweep: str | None = None
    options: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any] | "GenerationRequest", family: str | None = None
    ) -> "GenerationRequest":
        if isinstance(value, cls):
            request = value
        else:
            known = set(cls.__dataclass_fields__)
            payload = dict(value)
            extras = dict(payload.pop("options", {}) or {})
            for key in list(payload):
                if key not in known:
                    extras[key] = payload.pop(key)
            if isinstance(payload.get("branches"), str):
                payload["branches"] = tuple(
                    part for part in payload["branches"].split(",") if part
                )
            if isinstance(payload.get("views"), str):
                payload["views"] = tuple(
                    part for part in payload["views"].split(",") if part
                )
            payload["options"] = extras
            request = cls(**payload)
        if family is not None and request.family != family:
            raise ValueError(
                f"request family {request.family!r} does not match adapter {family!r}"
            )
        request.validate()
        return request

    def validate(self) -> None:
        if self.num_bundles < 1:
            raise ValueError("num_bundles must be positive")
        if not self.branches:
            raise ValueError("at least one branch is required")
        normalized_branches = [normalize_branch(branch) for branch in self.branches]
        if len(normalized_branches) != len(set(normalized_branches)):
            raise ValueError("branches must be unique after alias normalization")
        if self.sim_hz <= 0 or self.control_hz <= 0 or self.video_hz <= 0:
            raise ValueError("simulation, control, and video rates must be positive")
        if self.sim_hz % self.control_hz or self.sim_hz % self.video_hz:
            raise ValueError("sim_hz must be divisible by control_hz and video_hz")
        if not self.views or len(set(self.views)) != len(self.views):
            raise ValueError("view names must be non-empty and unique")
        if self.duration_s is not None and self.duration_s <= 0:
            raise ValueError("duration_s must be positive")


@dataclass(frozen=True)
class EpisodePlan:
    schema_version: str
    family: str
    subfamily: str
    variant: str
    robot_model: str
    tool_type: str
    episode_uuid: str
    counterfactual_bundle_id: str
    physics_counterfactual_family_id: str | None
    split_group_id: str
    scene_seed: int
    branch_seed: int
    intended_branch: str
    branch_parameters: Mapping[str, Any]
    scene_parameters: Mapping[str, Any]
    physics: Mapping[str, Any]
    physics_variant: str
    action_hash: str
    invariant_hash: str
    physics_hash: str
    views: tuple[str, ...]
    rates_hz: Mapping[str, int]
    duration_s: float
    randomization_level: str
    scene_style: str
    source_generator: str
    source_generator_version: str
    config_hash: str
    options: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class OutcomeResult:
    task_success: bool
    partial_success_score: float | None
    failure_mode: str
    metrics: Mapping[str, Any]
    label_confidence: float
    label_status: str = "verified_objective"

    def __post_init__(self) -> None:
        validate_failure_mode(self.task_success, self.failure_mode)
        if not 0.0 <= self.label_confidence <= 1.0:
            raise ValueError("label_confidence must be in [0, 1]")
        if self.partial_success_score is not None and not (
            0.0 <= self.partial_success_score <= 1.0
        ):
            raise ValueError("partial_success_score must be in [0, 1]")

    @classmethod
    def unverified(
        cls, metrics: Mapping[str, Any], reason: str = "objective_metric_unavailable"
    ) -> "OutcomeResult":
        return cls(
            False,
            None,
            "label_unverified",
            {"unverified_reason": reason, **dict(metrics)},
            0.0,
            "unverified",
        )

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class SimulationResult:
    plan: EpisodePlan
    frame_times_s: Sequence[float]
    states: Sequence[Mapping[str, Any]]
    actions: Sequence[Mapping[str, Any]]
    high_rate_states: Sequence[Mapping[str, Any]]
    contacts: Sequence[Mapping[str, Any]]
    outcome: OutcomeResult
    actual_outcome: str
    dynamics_mode: str
    release_tier: str
    assistance: Mapping[str, Any]
    physics_qc: Mapping[str, Any]
    simulator: Mapping[str, Any]
    notes: Sequence[str] = ()

    def __post_init__(self) -> None:
        if len(self.frame_times_s) != len(self.states):
            raise ValueError("frame timestamps and frame states must have equal length")
        if any(
            current <= previous
            for previous, current in zip(self.frame_times_s, self.frame_times_s[1:])
        ):
            raise ValueError("frame timestamps must be strictly increasing")
        if self.dynamics_mode not in {
            "free_contact",
            "assisted_contact",
            "scripted_motion",
        }:
            raise ValueError(f"invalid dynamics_mode {self.dynamics_mode!r}")
        if not self.outcome.task_success and self.outcome.failure_mode == "none":
            raise ValueError("actual failure cannot have failure_mode='none'")

    def episode_metadata(self) -> dict[str, Any]:
        metadata = self.plan.to_dict()
        metadata.update(
            {
                "actual_outcome": self.actual_outcome,
                **self.outcome.to_dict(),
                "dynamics_mode": self.dynamics_mode,
                "release_tier": self.release_tier,
                "assistance": _jsonable(self.assistance),
                "physics_qc": _jsonable(self.physics_qc),
                "simulator": _jsonable(self.simulator),
                "creation_timestamp": datetime.now(timezone.utc).isoformat(),
                "notes": list(self.notes),
            }
        )
        return metadata

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode": self.episode_metadata(),
            "frame_times_s": _jsonable(self.frame_times_s),
            "states": _jsonable(self.states),
            "actions": _jsonable(self.actions),
            "high_rate_states": _jsonable(self.high_rate_states),
            "contacts": _jsonable(self.contacts),
        }


def classify_actual_outcome(
    *,
    success: bool,
    contacted: bool,
    near_distance_m: float,
    near_threshold_m: float,
    bad_action: bool,
    no_op: bool = False,
) -> str:
    if success:
        return "success"
    if no_op:
        return "no_op"
    if bad_action:
        return "bad_action"
    if contacted:
        return "contact_failure"
    if near_distance_m <= near_threshold_m:
        return "near_miss"
    return "miss"


class FamilyAdapter(ABC):
    """Deterministic planning plus lightweight simulation family interface."""

    family: str
    version = "1.0.0"
    default_duration_s = 4.0
    supported_subfamilies: tuple[str, ...] = ("default",)

    def plan(
        self, config: Mapping[str, Any] | GenerationRequest
    ) -> list[EpisodePlan]:
        request = GenerationRequest.from_mapping(config, self.family)
        if request.subfamily not in self.supported_subfamilies:
            raise ValueError(
                f"unsupported {self.family} subfamily {request.subfamily!r}; "
                f"expected one of {self.supported_subfamilies}"
            )
        physics_variants = list(self.physics_variants(request))
        if not physics_variants:
            raise ValueError("physics_variants returned no configurations")
        plans: list[EpisodePlan] = []
        for scene_index in range(request.num_bundles):
            scene_seed = deterministic_seed(request.seed, self.family, scene_index)
            split_group_id = deterministic_uuid(
                SCHEMA_VERSION,
                self.family,
                request.subfamily,
                request.variant,
                request.seed,
                scene_index,
            )
            raw_scene = dict(self.scene_parameters(request, scene_seed, scene_index))
            catalog_value = request.options.get("randomization_catalog")
            catalog_mapping = catalog_value if isinstance(catalog_value, Mapping) else {}
            catalog = RandomizationCatalog(
                scene_asset_ids=tuple(str(value) for value in catalog_mapping.get("scene_asset_ids", ())),
                lighting_ids=tuple(str(value) for value in catalog_mapping.get("lighting_ids", ("neutral",))),
                object_asset_ids=tuple(str(value) for value in catalog_mapping.get("object_asset_ids", ())),
                object_color_ids=tuple(str(value) for value in catalog_mapping.get("object_color_ids", ("default",))),
                tool_asset_ids=tuple(str(value) for value in catalog_mapping.get("tool_asset_ids", ())),
                camera_preset_ids=tuple(str(value) for value in catalog_mapping.get("camera_preset_ids", ("main_secondary_v1",))),
            )
            randomization = RandomizationPlanner(catalog=catalog, seed=request.seed).plan(
                split_group_id,
                randomization_level=request.randomization_level,
            ).to_dict()
            if request.scene_style not in {"auto", "mixed", "catalog"}:
                randomization["background_style"] = request.scene_style
            raw_scene["background_style"] = randomization["background_style"]
            raw_scene["randomization"] = randomization
            scene: Mapping[str, Any] = raw_scene
            for physics_variant, physics in physics_variants:
                bundle_id = deterministic_uuid(split_group_id, physics_variant, "actions")
                for raw_branch in request.branches:
                    branch = normalize_branch(raw_branch)
                    branch_seed = deterministic_seed(scene_seed, branch)
                    branch_parameters = self.branch_parameters(
                        request, branch, branch_seed, scene
                    )
                    action_hash = stable_hash(branch_parameters)
                    invariant_payload = {
                        "family": self.family,
                        "subfamily": request.subfamily,
                        "variant": request.variant,
                        "scene": scene,
                        "scene_style": request.scene_style,
                        "views": request.views,
                        "branch": branch,
                        "action_hash": action_hash,
                    }
                    invariant_hash = stable_hash(invariant_payload)
                    physics_hash = stable_hash(physics)
                    physics_cf_id = (
                        deterministic_uuid(
                            split_group_id, branch, action_hash, "physics"
                        )
                        if len(physics_variants) > 1
                        else None
                    )
                    episode_uuid = deterministic_uuid(
                        bundle_id, physics_cf_id, physics_hash, branch_seed
                    )
                    config_hash = stable_hash(
                        {
                            "request": request,
                            "scene": scene,
                            "branch_parameters": branch_parameters,
                            "physics": physics,
                        }
                    )
                    plans.append(
                        EpisodePlan(
                            schema_version=SCHEMA_VERSION,
                            family=self.family,
                            subfamily=request.subfamily,
                            variant=request.variant,
                            robot_model=request.robot_model,
                            tool_type=request.tool_type,
                            episode_uuid=episode_uuid,
                            counterfactual_bundle_id=bundle_id,
                            physics_counterfactual_family_id=physics_cf_id,
                            split_group_id=split_group_id,
                            scene_seed=scene_seed,
                            branch_seed=branch_seed,
                            intended_branch=branch,
                            branch_parameters=branch_parameters,
                            scene_parameters=scene,
                            physics=physics,
                            physics_variant=physics_variant,
                            action_hash=action_hash,
                            invariant_hash=invariant_hash,
                            physics_hash=physics_hash,
                            views=request.views,
                            rates_hz={
                                "simulation": request.sim_hz,
                                "control": request.control_hz,
                                "video": request.video_hz,
                            },
                            duration_s=request.duration_s or self.default_duration_s,
                            randomization_level=request.randomization_level,
                            scene_style=request.scene_style,
                            source_generator=f"{self.__class__.__module__}.{self.__class__.__name__}",
                            source_generator_version=self.version,
                            config_hash=config_hash,
                            options=dict(request.options),
                        )
                    )
        return plans

    def generate(
        self, config: Mapping[str, Any] | GenerationRequest
    ) -> list[SimulationResult]:
        return [self.simulate(episode_plan) for episode_plan in self.plan(config)]

    @abstractmethod
    def scene_parameters(
        self, request: GenerationRequest, scene_seed: int, scene_index: int
    ) -> Mapping[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def branch_parameters(
        self,
        request: GenerationRequest,
        branch: str,
        branch_seed: int,
        scene: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def physics_variants(
        self, request: GenerationRequest
    ) -> Iterable[tuple[str, Mapping[str, Any]]]:
        raise NotImplementedError

    @abstractmethod
    def simulate(self, plan: EpisodePlan) -> SimulationResult:
        raise NotImplementedError


def assistance_record(
    *,
    assisted_grasp: bool = False,
    assisted_retention: bool = False,
    equality_constraint_active: bool = False,
    latch_active: bool = False,
    activation_time_s: float | None = None,
    deactivation_time_s: float | None = None,
    mechanism_id: str | None = None,
    mechanism_type: str | None = None,
    constraint_ids: Sequence[str] = (),
    target_body_ids: Sequence[str] = (),
    target_element_ids: Sequence[str] = (),
) -> dict[str, Any]:
    active = any(
        (assisted_grasp, assisted_retention, equality_constraint_active, latch_active)
    )
    mechanisms: list[dict[str, Any]] = []
    if active:
        if not mechanism_id or not mechanism_type:
            raise ValueError("Active assistance requires a mechanism ID and type")
        if activation_time_s is None:
            raise ValueError("Active assistance requires an activation time")
        if not target_body_ids and not target_element_ids:
            raise ValueError("Active assistance requires a named target body or element")
        mechanisms.append(
            {
                "mechanism_id": mechanism_id,
                "mechanism_type": mechanism_type,
                "source": "simulator_observed",
                "constraint_ids": list(constraint_ids),
                "target_body_ids": list(target_body_ids),
                "target_element_ids": list(target_element_ids),
                "activation_intervals": [
                    {
                        "start_time_s": activation_time_s,
                        "end_time_s": deactivation_time_s,
                    }
                ],
            }
        )
    return {
        "assisted_grasp": assisted_grasp,
        "assisted_retention": assisted_retention,
        "equality_constraint_active": equality_constraint_active,
        "latch_active": latch_active,
        "constraint_activation_time_s": activation_time_s,
        "constraint_deactivation_time_s": deactivation_time_s,
        "mechanisms": mechanisms,
    }


def default_physics_qc(**checks: bool | float | str) -> dict[str, Any]:
    hard_flags = [value for value in checks.values() if isinstance(value, bool)]
    return {"physics_qc_pass": all(hard_flags), "checks": _jsonable(checks)}
