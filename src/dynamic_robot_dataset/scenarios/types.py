"""Canonical per-subfamily scenario module contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Any, Callable, Mapping, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from ..common.corpus_registry import CorpusLeaf


class ScenarioBlockedError(RuntimeError):
    """Raised when a declared corpus leaf has no honest executable recipe."""


@dataclass(frozen=True, slots=True)
class ScenarioBuildContext:
    """Backend-neutral inputs supplied to one canonical recipe builder."""

    leaf_id: str
    task_variant: str
    embodiment: str
    branch_role: str
    seed: int
    physics_seed: int
    tabletop_height_m: float
    rolling_island_scene: Any | None = None
    initial_state_mode: str = "fixed_review"


class RecipeBuilder(Protocol):
    def __call__(self, context: ScenarioBuildContext) -> Mapping[str, Any]: ...


@dataclass(frozen=True, slots=True)
class ControllerPlan:
    """Declarative controller selection; low-level actuation remains shared."""

    kind: str
    trajectory: str
    actuator_only: bool = True
    retention_required: bool = False
    hand_orientation: str = "auto"
    settled_aim_correction: bool = False
    compact_pickup_ready: bool = False
    robotiq_reach_arrival_lead_s: float | None = None
    robotiq_tendon_profile: str = "default"
    robotiq_tendon_target: float | None = None
    robotiq_actuator_force_limit_n: float | None = None
    robotiq_pad_half_depth_m: float | None = None
    robotiq_pad_contact_margin_m: float | None = None
    robotiq_controller_target_bias_m: tuple[float, float, float] | None = None
    negative_controller_offset_m: tuple[float, float, float] | None = None
    interior_joint_margin_rad: float = 0.0
    sampled_projectile_ready_offset_m: tuple[float, float, float] = (
        0.0,
        0.0,
        0.0,
    )

    def validate(self) -> None:
        if not self.kind or not self.trajectory:
            raise ValueError("scenario controller plan is incomplete")
        if not self.actuator_only:
            raise ValueError("canonical scenarios must remain actuator-only")
        if self.hand_orientation not in {"auto", "none", "catch_up", "pick_down"}:
            raise ValueError("scenario hand orientation is invalid")
        if self.robotiq_tendon_profile not in {"default", "pickup", "f2c"}:
            raise ValueError("scenario Robotiq tendon profile is invalid")
        if (
            self.robotiq_tendon_target is not None
            and not 0.0 < self.robotiq_tendon_target <= 200.0
        ):
            raise ValueError("scenario Robotiq tendon target is invalid")
        if (
            self.robotiq_actuator_force_limit_n is not None
            and not 0.0 < self.robotiq_actuator_force_limit_n <= 0.16
        ):
            raise ValueError("scenario Robotiq actuator force limit is invalid")
        if (
            self.robotiq_pad_half_depth_m is not None
            and not 0.010 <= self.robotiq_pad_half_depth_m <= 0.022
        ):
            raise ValueError(
                "scenario Robotiq collision pad must stay inside the visible pad"
            )
        if (
            self.robotiq_pad_contact_margin_m is not None
            and not 0.0 <= self.robotiq_pad_contact_margin_m <= 0.004
        ):
            raise ValueError("scenario Robotiq pad contact margin is invalid")
        if self.robotiq_controller_target_bias_m is not None and (
            len(self.robotiq_controller_target_bias_m) != 3
            or not all(
                math.isfinite(float(value))
                for value in self.robotiq_controller_target_bias_m
            )
            or math.sqrt(
                sum(
                    float(value) ** 2
                    for value in self.robotiq_controller_target_bias_m
                )
            )
            > 0.05
        ):
            raise ValueError(
                "scenario Robotiq controller-target bias must be a bounded "
                "finite 3-vector"
            )
        if self.negative_controller_offset_m is not None and (
            len(self.negative_controller_offset_m) != 3
            or not all(
                math.isfinite(float(value))
                for value in self.negative_controller_offset_m
            )
        ):
            raise ValueError(
                "scenario negative-controller offset must be a finite 3-vector"
            )
        if (
            self.robotiq_reach_arrival_lead_s is not None
            and self.robotiq_reach_arrival_lead_s <= 0.0
        ):
            raise ValueError("scenario reach-arrival lead must be positive")
        if self.interior_joint_margin_rad < 0.0:
            raise ValueError("scenario joint-limit margin cannot be negative")
        if (
            len(self.sampled_projectile_ready_offset_m) != 3
            or not all(
                math.isfinite(float(value))
                for value in self.sampled_projectile_ready_offset_m
            )
        ):
            raise ValueError("scenario projectile ready offset must be a finite 3-vector")

    def to_dict(self) -> dict[str, Any]:
        """Serialize without changing leaves that do not declare new options."""

        payload = asdict(self)
        for name in (
            "robotiq_tendon_target",
            "robotiq_actuator_force_limit_n",
            "robotiq_pad_half_depth_m",
            "robotiq_pad_contact_margin_m",
            "robotiq_controller_target_bias_m",
            "negative_controller_offset_m",
        ):
            if payload[name] is None:
                payload.pop(name)
        return payload


@dataclass(frozen=True, slots=True)
class ScenarioModuleSpec:
    """Static identity exported as ``SCENARIO`` by every leaf module."""

    leaf_id: str
    family: str
    subfamily: str
    backend: str
    fixture_policy: str
    controller_plan: ControllerPlan
    build_recipe: RecipeBuilder | None
    implementation_note: str
    randomization_contract: Mapping[str, Any] | None = None

    @property
    def implemented(self) -> bool:
        return self.build_recipe is not None

    def validate(self) -> None:
        if not all(
            value.strip()
            for value in (
                self.leaf_id,
                self.family,
                self.subfamily,
                self.backend,
                self.fixture_policy,
                self.implementation_note,
            )
        ):
            raise ValueError("scenario module identity is incomplete")
        self.controller_plan.validate()
        if self.randomization_contract is not None and not self.randomization_contract:
            raise ValueError("scenario randomization contract cannot be empty")

    def build(self, context: ScenarioBuildContext) -> dict[str, Any]:
        self.validate()
        if context.leaf_id != self.leaf_id:
            raise ValueError(
                f"scenario module {self.leaf_id} cannot build {context.leaf_id}"
            )
        if self.build_recipe is None:
            raise ScenarioBlockedError(
                f"{self.leaf_id}/{self.subfamily} is contract-only: "
                f"{self.implementation_note}"
            )
        return dict(self.build_recipe(context))


@dataclass(frozen=True, slots=True)
class ScenarioDefinition:
    """Registry-bound scenario metadata returned to compiler and operators."""

    module: ScenarioModuleSpec
    module_path: str
    variants: tuple[str, ...]
    embodiments: tuple[str, ...]
    evaluator_id: str
    release_state: str
    blockers: tuple[str, ...]

    @property
    def leaf_id(self) -> str:
        return self.module.leaf_id

    @property
    def family(self) -> str:
        return self.module.family

    @property
    def subfamily(self) -> str:
        return self.module.subfamily

    @property
    def backend(self) -> str:
        return self.module.backend

    @property
    def implemented(self) -> bool:
        return self.module.implemented

    def validate_against_leaf(self, leaf: "CorpusLeaf") -> None:
        self.module.validate()
        expected = (
            leaf.corpus_id,
            leaf.family,
            leaf.subfamily,
            leaf.backend,
            leaf.task_variants,
            leaf.supported_embodiments,
            leaf.evaluator,
            leaf.release_state.value,
            leaf.blockers,
        )
        actual = (
            self.leaf_id,
            self.family,
            self.subfamily,
            self.backend,
            self.variants,
            self.embodiments,
            self.evaluator_id,
            self.release_state,
            self.blockers,
        )
        if actual != expected:
            raise ValueError(
                f"scenario module {self.module_path} disagrees with corpus leaf "
                f"{leaf.corpus_id}"
            )

    def build(self, context: ScenarioBuildContext) -> dict[str, Any]:
        if context.task_variant not in self.variants:
            raise ValueError(
                f"{self.leaf_id} does not declare variant {context.task_variant!r}"
            )
        if context.embodiment not in self.embodiments:
            raise ValueError(
                f"{self.leaf_id} does not declare embodiment {context.embodiment!r}"
            )
        return self.module.build(context)

    def to_dict(self) -> dict[str, Any]:
        return {
            "leaf_id": self.leaf_id,
            "family": self.family,
            "subfamily": self.subfamily,
            "backend": self.backend,
            "module_path": self.module_path,
            "variants": list(self.variants),
            "embodiments": list(self.embodiments),
            "evaluator_id": self.evaluator_id,
            "fixture_policy": self.module.fixture_policy,
            "controller_plan": self.module.controller_plan.to_dict(),
            "implemented": self.implemented,
            "implementation_note": self.module.implementation_note,
            "randomization_contract": (
                None
                if self.module.randomization_contract is None
                else dict(self.module.randomization_contract)
            ),
            "release_state": self.release_state,
            "blockers": list(self.blockers),
        }


def bind_module(spec: ScenarioModuleSpec, leaf: "CorpusLeaf") -> ScenarioDefinition:
    result = ScenarioDefinition(
        module=spec,
        module_path=leaf.scenario_module,
        variants=leaf.task_variants,
        embodiments=leaf.supported_embodiments,
        evaluator_id=leaf.evaluator,
        release_state=leaf.release_state.value,
        blockers=leaf.blockers,
    )
    result.validate_against_leaf(leaf)
    return result


__all__ = [
    "ControllerPlan",
    "RecipeBuilder",
    "ScenarioBlockedError",
    "ScenarioBuildContext",
    "ScenarioDefinition",
    "ScenarioModuleSpec",
    "bind_module",
]
