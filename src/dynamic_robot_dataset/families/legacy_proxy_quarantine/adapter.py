"""Explicit quarantine records for four known legacy assistance/proxy modes."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from ..base import (
    EpisodePlan,
    FamilyAdapter,
    GenerationRequest,
    OutcomeResult,
    SimulationResult,
    assistance_record,
    default_physics_qc,
    physics_field,
)


LEGACY_PROXY_TYPES = (
    "assisted_latch",
    "equality_grasp",
    "scripted_bounce",
    "legacy_motion_proxy",
)


class LegacyProxyQuarantineAdapter(FamilyAdapter):
    family = "legacy_proxy_quarantine"
    supported_subfamilies = ("default",) + LEGACY_PROXY_TYPES
    default_duration_s = 2.5

    def scene_parameters(self, request: GenerationRequest, scene_seed: int, scene_index: int) -> Mapping[str, Any]:
        return {
            "proxy_type": request.subfamily if request.subfamily != "default" else "legacy_motion_proxy",
            "visual_seed": scene_seed,
            "background_style": request.scene_style,
        }

    def branch_parameters(self, request: GenerationRequest, branch: str, branch_seed: int, scene: Mapping[str, Any]) -> Mapping[str, Any]:
        return {"legacy_branch": branch, "quarantine_only": True}

    def physics_variants(self, request: GenerationRequest) -> Iterable[tuple[str, Mapping[str, Any]]]:
        yield "unknown_legacy", {
            "gravity": physics_field(None, "m/s^2", implemented=False, interpretation="unknown"),
            "source_physics_complete": physics_field(None, "bool", valid=False, implemented=False, interpretation="unknown"),
        }

    def simulate(self, plan: EpisodePlan) -> SimulationResult:
        proxy_type = str(plan.scene_parameters["proxy_type"])
        assisted = proxy_type in {"assisted_latch", "equality_grasp"}
        scripted = not assisted
        count = int(round(plan.duration_s * plan.rates_hz["video"])) + 1
        times = [index / plan.rates_hz["video"] for index in range(count)]
        states = [
            {
                "timestamp": time_s,
                "legacy.proxy_position": [0.05 * time_s, 0.0, 0.2],
                "assistance.active": assisted,
                "legacy.quarantine": True,
            }
            for time_s in times
        ]
        outcome = OutcomeResult(
            False,
            None,
            "legacy_proxy_quarantined",
            {"proxy_type": proxy_type, "objective_evaluator_available": False},
            0.0,
            "unverified",
        )
        return SimulationResult(
            plan=plan,
            frame_times_s=times,
            states=states,
            actions=[{"timestamp": time_s, "command.legacy_proxy": proxy_type} for time_s in times],
            high_rate_states=states,
            contacts=(),
            outcome=outcome,
            actual_outcome="quarantined",
            dynamics_mode="assisted_contact" if assisted else "scripted_motion",
            release_tier="assisted_contact" if assisted else "scripted_motion",
            assistance=assistance_record(
                assisted_grasp=assisted,
                assisted_retention=proxy_type == "assisted_latch",
                equality_constraint_active=proxy_type == "equality_grasp",
                latch_active=proxy_type == "assisted_latch",
                activation_time_s=0.0 if assisted else None,
                deactivation_time_s=plan.duration_s if assisted else None,
            ),
            physics_qc=default_physics_qc(source_physics_complete=False, native_contact_dynamics=False),
            simulator={"name": f"legacy_quarantine.{proxy_type}", "version": "unknown", "production_eligible": False},
            notes=("Quarantine fixture only; never eligible for default release or training.",),
        )
