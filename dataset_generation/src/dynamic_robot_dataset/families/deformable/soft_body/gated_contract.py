"""Fail-closed contracts for soft-object scenarios that are not yet admitted."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import json
from typing import Any, Mapping


class SoftObjectScenario(str, Enum):
    FOAM_BALL_CATCH = "foam_ball_catch"
    FOAM_BALL_PADDLE_DEFLECTION = "foam_ball_paddle_deflection"
    BEANBAG_DROP_CATCH = "beanbag_drop_catch"
    SOFT_POUCH_SLIDING_SCOOP = "soft_pouch_sliding_scoop"


@dataclass(slots=True, frozen=True)
class SoftObjectGateDecision:
    scenario: SoftObjectScenario
    unlocked: bool
    blockers: tuple[str, ...]
    evidence: dict[str, Any]


def evaluate_soft_object_gate(
    scenario: SoftObjectScenario | str,
    *,
    rigid_gate_report: str | Path | Mapping[str, Any] | None = None,
    deformable_model_report: str | Path | Mapping[str, Any] | None = None,
) -> SoftObjectGateDecision:
    """Require persisted validation evidence; config flags alone cannot unlock generation."""

    scenario = SoftObjectScenario(scenario)

    def load(value: str | Path | Mapping[str, Any] | None) -> dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, Mapping):
            return dict(value)
        return dict(json.loads(Path(value).resolve(strict=True).read_text(encoding="utf-8")))

    rigid = load(rigid_gate_report)
    deformable = load(deformable_model_report)
    blockers: list[str] = []
    if rigid.get("gate_id") != "native_rigid_contact_acceptance_v1" or rigid.get("passed") is not True:
        blockers.append("rigid_contact_validation_not_passed")
    if scenario in {
        SoftObjectScenario.BEANBAG_DROP_CATCH,
        SoftObjectScenario.SOFT_POUCH_SLIDING_SCOOP,
    } and not (
        deformable.get("shell_model_validated") is True
        and deformable.get("self_contact_validated") is True
        and deformable.get("passed") is True
    ):
        blockers.append("shell_and_self_contact_model_not_validated")
    return SoftObjectGateDecision(
        scenario=scenario,
        unlocked=not blockers,
        blockers=tuple(blockers),
        evidence={
            "rigid_gate_id": rigid.get("gate_id"),
            "rigid_gate_passed": rigid.get("passed"),
            "deformable_model_id": deformable.get("model_id"),
            "deformable_model_passed": deformable.get("passed"),
        },
    )
