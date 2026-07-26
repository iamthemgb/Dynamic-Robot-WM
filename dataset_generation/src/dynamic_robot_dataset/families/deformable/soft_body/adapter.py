"""Soft-body material schema and quarantined evaluator smoke model."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

from ...base import (
    EpisodePlan,
    FamilyAdapter,
    GenerationRequest,
    OutcomeResult,
    SimulationResult,
    assistance_record,
    classify_actual_outcome,
    default_physics_qc,
    physics_field,
    physics_value,
)


SOFT_BODY_TASKS = ("compression", "indentation", "stretch", "drop_rebound")
_MATERIALS = {
    "soft": {"youngs_modulus_pa": 2.0e4, "poisson_ratio": 0.42, "damping": 0.16},
    "medium": {"youngs_modulus_pa": 8.0e4, "poisson_ratio": 0.38, "damping": 0.10},
    "stiff": {"youngs_modulus_pa": 3.2e5, "poisson_ratio": 0.34, "damping": 0.06},
}


def material_sweep() -> tuple[tuple[str, Mapping[str, float]], ...]:
    return tuple((name, dict(values)) for name, values in _MATERIALS.items())


class SoftBodyAdapter(FamilyAdapter):
    family = "soft_body"
    supported_subfamilies = ("default",) + SOFT_BODY_TASKS
    default_duration_s = 4.5

    def scene_parameters(self, request: GenerationRequest, scene_seed: int, scene_index: int) -> Mapping[str, Any]:
        return {
            "task": request.subfamily if request.subfamily != "default" else "compression",
            "rest_dimensions_m": [0.22, 0.18, 0.12],
            "initial_position_m": [0.0, 0.0, 0.10],
            "visual_seed": scene_seed,
            "background_style": request.scene_style,
        }

    def branch_parameters(self, request: GenerationRequest, branch: str, branch_seed: int, scene: Mapping[str, Any]) -> Mapping[str, Any]:
        magnitude = {"success_seeking": 1.0, "near_miss": 0.55, "contact_failure": 0.28, "bad_action": 0.0}.get(branch, 0.0)
        return {"command_magnitude": magnitude, "tool_contact": branch != "bad_action"}

    @staticmethod
    def _physics(request: GenerationRequest, material: str) -> Mapping[str, Any]:
        values = _MATERIALS[material]
        return {
            "youngs_modulus": physics_field(values["youngs_modulus_pa"], "Pa", interpretation="proxy"),
            "poisson_ratio": physics_field(values["poisson_ratio"], "1", interpretation="proxy"),
            "density": physics_field(620.0, "kg/m^3", interpretation="proxy"),
            "damping": physics_field(values["damping"], "1", interpretation="proxy"),
            "friction": physics_field(0.42, "1", interpretation="proxy"),
            "restitution": physics_field(0.18, "1", interpretation="proxy"),
            "plasticity": physics_field(0.0, "1", interpretation="proxy"),
            "mesh_resolution": physics_field([8, 7, 5], "vertices", interpretation="proxy"),
        }

    def physics_variants(self, request: GenerationRequest) -> Iterable[tuple[str, Mapping[str, Any]]]:
        if request.physics_sweep == "material":
            for material, _ in material_sweep():
                yield f"material_{material}", self._physics(request, material)
        elif request.physics_sweep:
            raise ValueError("soft_body supports only physics_sweep='material'")
        else:
            yield "material_medium", self._physics(request, "medium")

    def simulate(self, plan: EpisodePlan) -> SimulationResult:
        task = str(plan.scene_parameters["task"])
        magnitude = float(plan.branch_parameters["command_magnitude"])
        modulus = float(physics_value(plan.physics, "youngs_modulus"))
        compliance = (8.0e4 / modulus) ** 0.35
        peak_strain = min(0.55, 0.24 * magnitude * compliance)
        if task == "compression":
            metric = peak_strain
            success = metric >= 0.12
            failure = "none" if success else "insufficient_compression"
            metrics = {"peak_compressive_strain": metric, "threshold": 0.12}
        elif task == "indentation":
            metric = 0.045 * magnitude * compliance
            success = metric >= 0.018
            failure = "none" if success else "insufficient_indentation_depth"
            metrics = {"peak_indentation_depth_m": metric, "threshold_m": 0.018}
        elif task == "stretch":
            metric = peak_strain
            success = metric >= 0.10
            failure = "none" if success else "insufficient_tensile_strain"
            metrics = {"peak_tensile_strain": metric, "threshold": 0.10}
        elif task == "drop_rebound":
            restitution = float(physics_value(plan.physics, "restitution"))
            metric = 0.45 * magnitude * restitution
            success = metric >= 0.055
            failure = "none" if success else "insufficient_rebound_height"
            metrics = {"rebound_height_m": metric, "threshold_m": 0.055}
        else:
            outcome = OutcomeResult.unverified({"task": task}, "soft_body_metric_not_implemented")
            metric, success = 0.0, False
            failure, metrics = outcome.failure_mode, outcome.metrics
        if task in SOFT_BODY_TASKS:
            outcome = OutcomeResult(success, 1.0 if success else min(1.0, metric / max(next(iter(metrics.values())) if metrics else 1.0, 1e-9)), failure, metrics, 0.5, "unverified")
        frames = int(round(plan.duration_s * plan.rates_hz["video"])) + 1
        times = [index / plan.rates_hz["video"] for index in range(frames)]
        states = []
        actions = []
        for time_s in times:
            phase = math.sin(math.pi * min(1.0, time_s / (0.65 * plan.duration_s))) if time_s <= 0.65 * plan.duration_s else math.exp(-4.0 * (time_s / plan.duration_s - 0.65))
            deformation = metric * max(0.0, phase)
            states.append({
                "timestamp": time_s,
                "soft_body.centroid": list(plan.scene_parameters["initial_position_m"]),
                "soft_body.deformation_metric": deformation,
                "soft_body.task": task,
                "assistance.active": bool(plan.branch_parameters["tool_contact"]),
            })
            actions.append({
                "timestamp": time_s,
                "command.task": task,
                "command.magnitude": magnitude,
                "command.tool_contact": bool(plan.branch_parameters["tool_contact"]),
            })
        return SimulationResult(
            plan=plan, frame_times_s=times, states=states, actions=actions,
            high_rate_states=[{"timestamp": row["timestamp"], "soft_body.deformation_metric": row["soft_body.deformation_metric"]} for row in states],
            contacts=(), outcome=outcome,
            actual_outcome=classify_actual_outcome(
                success=outcome.task_success, contacted=bool(plan.branch_parameters["tool_contact"]),
                near_distance_m=0.0 if plan.branch_parameters["tool_contact"] else 1.0,
                near_threshold_m=0.1,
                bad_action=magnitude == 0.0 and plan.intended_branch != "no_op",
                no_op=plan.intended_branch == "no_op",
            ),
            dynamics_mode="scripted_motion", release_tier="scripted_motion",
            assistance=assistance_record(
                assisted_grasp=bool(plan.branch_parameters["tool_contact"]),
                equality_constraint_active=False,
                activation_time_s=0.0 if plan.branch_parameters["tool_contact"] else None,
                deactivation_time_s=plan.duration_s if plan.branch_parameters["tool_contact"] else None,
                mechanism_id=("diagnostic-soft-body-tool-contact" if plan.branch_parameters["tool_contact"] else None),
                mechanism_type=("assisted_grasp" if plan.branch_parameters["tool_contact"] else None),
                target_body_ids=("soft_body_proxy",) if plan.branch_parameters["tool_contact"] else (),
            ),
            physics_qc=default_physics_qc(finite_state=True, native_deformable_physics=False),
            simulator={"name": "dynamic_robot_dataset.quarantined_soft_body_response_proxy", "version": "1", "native_flex": False, "production_eligible": False},
            notes=("Scalar response curves test material conditioning and schemas only; they are not soft-body training data.",),
        )
