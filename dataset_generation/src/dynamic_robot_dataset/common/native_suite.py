"""Pure planning for the immutable native acceptance suite.

This module deliberately stops at planning.  It converts :class:`SuiteCase`
records into native rigid scenario specifications or explicit diagnostic
quarantine plans, and declares complete counterfactual membership before any
simulator is started.  No filesystem writes or MuJoCo imports occur here.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Any, Iterable, Literal, Mapping, Sequence, cast

from ..backends import IntendedBranch, PhysicsRangeProvenance, ScenarioSpec
from ..backends.mujoco_native import (
    make_scenario_spec_for_subfamily,
    scenario_to_episode_plan,
)
from ..families import get_family
from ..families.base import (
    EpisodePlan,
    deterministic_seed,
    deterministic_uuid,
    normalize_branch,
    stable_hash,
)
from .contract_v2 import CounterfactualFamilyRecord, CounterfactualRelation
from .hashing import sha256_json
from .suites import SuiteCase


ExecutionBackend = Literal["native_mujoco", "diagnostic_quarantine"]


# These are the five-point intervention values used by the existing physics
# oracle tests.  They are explicit here so a suite member named ``gravity_3``
# cannot silently mean a random sample or a nominal placeholder.
PHYSICS_SWEEP_VALUES: Mapping[str, tuple[float, ...]] = {
    "gravity": (4.905, 7.3575, 9.81, 12.2625, 14.715),
    "friction": (0.05, 0.15, 0.30, 0.60, 1.00),
    "restitution": (0.05, 0.25, 0.50, 0.75, 0.95),
}


@dataclass(frozen=True)
class PlannedSuiteEpisode:
    """One execution-ready plan paired with its immutable suite case."""

    case: SuiteCase
    episode_plan: EpisodePlan
    execution_backend: ExecutionBackend
    scenario_spec: ScenarioSpec | None

    def __post_init__(self) -> None:
        if self.execution_backend == "native_mujoco" and self.scenario_spec is None:
            raise ValueError("native suite plans require a typed ScenarioSpec")
        if self.execution_backend == "diagnostic_quarantine" and self.scenario_spec is not None:
            raise ValueError("diagnostic quarantine plans cannot carry a native ScenarioSpec")
        plan = self.episode_plan
        if plan.counterfactual_bundle_id != self.case.counterfactual_bundle_id:
            raise ValueError("suite action-bundle identity was not preserved")
        if plan.physics_counterfactual_family_id != self.case.physics_counterfactual_family_id:
            raise ValueError("suite physics-family identity was not preserved")
        if plan.split_group_id != self.case.split_group_id:
            raise ValueError("suite split-group identity was not preserved")
        if plan.scene_seed != self.case.seed:
            raise ValueError("suite bundle seed was not preserved")

    @property
    def episode_uuid(self) -> str:
        return self.episode_plan.episode_uuid


def _suite_episode_uuid(case: SuiteCase) -> str:
    return deterministic_uuid(
        "native-acceptance-suite-episode-v1",
        case.suite_name,
        case.category_id,
        case.request_index,
        case.repeat_index,
        case.branch,
        case.physics_variant,
    )


def _branch_for_case(case: SuiteCase) -> IntendedBranch:
    normalized = normalize_branch(case.branch)
    aliases = {
        "success_seeking": IntendedBranch.SUCCESS,
        "near_miss": IntendedBranch.NEAR_MISS,
        "contact_failure": IntendedBranch.CONTACT_FAILURE,
        "no_op": IntendedBranch.NO_OP,
        "bad_action": IntendedBranch.WRONG_ACTION,
        "wrong_action": IntendedBranch.WRONG_ACTION,
    }
    try:
        return aliases[normalized]
    except KeyError as error:
        raise ValueError(f"unsupported native intended branch {case.branch!r}") from error


def _sweep_intervention(case: SuiteCase) -> tuple[str | None, float | None]:
    if case.physics_variant == "nominal":
        if case.physics_intervention_fields:
            raise ValueError("nominal suite case cannot declare a physics intervention")
        return None, None
    try:
        sweep_name, raw_index = case.physics_variant.rsplit("_", 1)
        values = PHYSICS_SWEEP_VALUES[sweep_name]
        index = int(raw_index)
        value = values[index]
    except (KeyError, ValueError, IndexError) as error:
        raise ValueError(f"unsupported suite physics variant {case.physics_variant!r}") from error
    expected_fields = {
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
    }[sweep_name]
    if case.physics_intervention_fields != expected_fields:
        raise ValueError(
            f"{case.physics_variant} must declare intervention fields {expected_fields}, "
            f"not {case.physics_intervention_fields}"
        )
    return sweep_name, value


def _partition_for_sweep(variant: str) -> str:
    if variant == "nominal":
        return "nominal"
    index = int(variant.rsplit("_", 1)[1])
    # The catalog remains explicitly uncalibrated.  Outer sweep values are
    # tagged as candidate OOD and central values as candidate ID support.
    return "test_ood" if index in {0, 4} else "train_id"


def _validate_and_tag_sweep(spec: ScenarioSpec, case: SuiteCase) -> ScenarioSpec:
    """Validate backend-applied values and attach provisional range provenance."""

    sweep, value = _sweep_intervention(case)
    provenance = PhysicsRangeProvenance(
        range_version="franka_rigid_provisional_v1",
        partition=_partition_for_sweep(case.physics_variant),
        calibrated=False,
        calibration_artifact=None,
    )
    if sweep is None:
        return replace(spec, physics_provenance=provenance)
    assert value is not None
    if sweep == "gravity":
        if spec.gravity_m_s2 != (0.0, 0.0, -value):
            raise ValueError(f"native backend did not apply {case.physics_variant}")
    if sweep == "friction":
        if spec.surface_friction[0] != value:
            raise ValueError(f"native backend did not apply {case.physics_variant}")
    if sweep == "restitution":
        if spec.object.effective_restitution_target != value:
            raise ValueError(f"native backend did not apply {case.physics_variant}")
    return replace(spec, physics_provenance=provenance)


def _native_plan(case: SuiteCase) -> PlannedSuiteEpisode:
    extras = {
        **case.options,
        "scene_group_id": case.split_group_id,
        "suite_name": case.suite_name,
        "suite_category_id": case.category_id,
        "suite_request_index": case.request_index,
        "suite_repeat_index": case.repeat_index,
        "suite_subfamily": case.subfamily,
        "physics_variant": case.physics_variant,
        "counterfactual_intervention_fields": list(case.physics_intervention_fields),
        "bundle_seed": case.seed,
    }
    spec = make_scenario_spec_for_subfamily(
        case.family,
        case.subfamily,
        seed=case.seed,
        branch=_branch_for_case(case),
        scene_style=case.scene_style,
        physics_variant=case.physics_variant,
        split_group_id=case.split_group_id,
        counterfactual_bundle_id=case.counterfactual_bundle_id,
        physics_counterfactual_family_id=case.physics_counterfactual_family_id,
        options=extras,
    )
    spec = _validate_and_tag_sweep(spec, case)
    spec = replace(
        spec,
        robot_model=case.robot_model,
        counterfactual_bundle_id=case.counterfactual_bundle_id,
        physics_counterfactual_family_id=case.physics_counterfactual_family_id,
        split_group_id=case.split_group_id,
    )
    spec.validate()
    raw_plan = scenario_to_episode_plan(spec)
    # Preserve the suite's genuinely optional family ID: nominal, non-sweep
    # cases have no physics-family relationship to declare.
    options = {
        **raw_plan.options,
        "execution_backend": "native_mujoco",
        "suite_case": case.to_dict(),
        "diagnostic_quarantine": False,
        "production_eligible_by_planning": False,
    }
    plan = replace(
        raw_plan,
        episode_uuid=_suite_episode_uuid(case),
        subfamily=case.subfamily,
        counterfactual_bundle_id=case.counterfactual_bundle_id,
        physics_counterfactual_family_id=cast(str, case.physics_counterfactual_family_id),
        split_group_id=case.split_group_id,
        scene_seed=case.seed,
        branch_seed=deterministic_seed(case.seed, normalize_branch(case.branch)),
        intended_branch=normalize_branch(case.branch),
        views=case.views,
        physics_variant=case.physics_variant,
        options=options,
        config_hash=stable_hash({"suite_case": case.to_dict(), "scenario": spec.to_dict()}),
    )
    return PlannedSuiteEpisode(case, plan, "native_mujoco", spec)


def _diagnostic_plan(case: SuiteCase) -> PlannedSuiteEpisode:
    adapter = get_family(case.family)
    requested_branch = normalize_branch(case.branch)
    generated = adapter.plan(
        {
            "family": case.family,
            "subfamily": case.subfamily,
            "variant": "native_acceptance_diagnostic_quarantine_v1",
            "robot_model": case.robot_model,
            "tool_type": str(case.options.get("tool_type", "task_default")),
            "num_bundles": 1,
            "branches": [requested_branch],
            "views": list(case.views),
            "seed": case.seed,
            "scene_style": case.scene_style,
            "sim_hz": 240,
            "control_hz": 60,
            "video_hz": 30,
            "options": {
                **case.options,
                "execution_backend": "diagnostic_quarantine",
                "diagnostic_quarantine": True,
                "production_eligible": False,
            },
        }
    )
    if len(generated) != 1:
        raise ValueError(
            f"diagnostic adapter {case.family}/{case.subfamily} produced "
            f"{len(generated)} plans for one suite case"
        )
    raw = generated[0]
    scene = {
        **raw.scene_parameters,
        "visual_seed": case.seed,
        "background_style": case.scene_style,
    }
    branch_parameters = dict(raw.branch_parameters)
    if requested_branch == "no_op":
        # Diagnostic proxies predate first-class no-op.  Preserve their schema
        # path while explicitly disabling every known synthetic attachment.
        for key in (
            "controller_enabled",
            "attachment_enabled",
            "endpoint_attachment",
            "assisted_grasp",
            "assisted_retention",
            "latch_active",
        ):
            if key in branch_parameters:
                branch_parameters[key] = False
        if "control_magnitude" in branch_parameters:
            branch_parameters["control_magnitude"] = 0.0
    action_hash = stable_hash(branch_parameters)
    options = {
        **raw.options,
        "execution_backend": "diagnostic_quarantine",
        "diagnostic_quarantine": True,
        "production_eligible": False,
        "suite_case": case.to_dict(),
    }
    plan = replace(
        raw,
        episode_uuid=_suite_episode_uuid(case),
        counterfactual_bundle_id=case.counterfactual_bundle_id,
        physics_counterfactual_family_id=cast(str, case.physics_counterfactual_family_id),
        split_group_id=case.split_group_id,
        scene_seed=case.seed,
        branch_seed=deterministic_seed(case.seed, requested_branch),
        intended_branch=requested_branch,
        branch_parameters=branch_parameters,
        scene_parameters=scene,
        physics_variant=case.physics_variant,
        action_hash=action_hash,
        invariant_hash=stable_hash(
            {
                "scene": scene,
                "action": branch_parameters,
                "views": case.views,
                "robot_model": case.robot_model,
            }
        ),
        views=case.views,
        scene_style=case.scene_style,
        config_hash=stable_hash({"suite_case": case.to_dict(), "adapter_plan": raw.to_dict()}),
        options=options,
    )
    return PlannedSuiteEpisode(case, plan, "diagnostic_quarantine", None)


def _initial_state_payload(item: PlannedSuiteEpisode) -> Mapping[str, Any]:
    if item.scenario_spec is not None:
        state = item.scenario_spec.initial_state
        return {
            "position_m": state.position_m,
            "quaternion_wxyz": state.quaternion_wxyz,
            "linear_velocity_m_s": state.linear_velocity_m_s,
            "angular_velocity_rad_s": state.angular_velocity_rad_s,
        }
    scene = item.episode_plan.scene_parameters
    return {
        key: scene[key]
        for key in sorted(scene)
        if key.startswith("initial_") or key in {"mesh_shape", "length_m", "segment_count"}
    }


def _appearance_payload(item: PlannedSuiteEpisode) -> Mapping[str, Any]:
    if item.scenario_spec is not None:
        spec = item.scenario_spec
        return {
            "scene_style": spec.scene_style,
            "rgba": spec.object.rgba,
            "cameras": [camera.__dict__ for camera in spec.cameras],
            "seed": spec.seed,
        }
    plan = item.episode_plan
    return {
        "scene_style": plan.scene_style,
        "visual_seed": plan.scene_parameters.get("visual_seed", plan.scene_seed),
        "randomization": plan.scene_parameters.get("randomization", {}),
        "views": plan.views,
    }


def _nonintervened_physics(
    physics: Mapping[str, Any], intervention_fields: Sequence[str]
) -> dict[str, Any]:
    excluded = set(intervention_fields)
    return {name: value for name, value in physics.items() if name not in excluded}


def build_planned_counterfactual_family_records(
    planned: Iterable[PlannedSuiteEpisode],
) -> list[CounterfactualFamilyRecord]:
    """Declare exact suite membership and invariant hashes pre-simulation."""

    values = list(planned)
    action_groups: dict[str, list[PlannedSuiteEpisode]] = defaultdict(list)
    physics_groups: dict[str, list[PlannedSuiteEpisode]] = defaultdict(list)
    for item in values:
        action_groups[item.case.counterfactual_bundle_id].append(item)
        if item.case.physics_counterfactual_family_id is not None:
            physics_groups[item.case.physics_counterfactual_family_id].append(item)

    declarations: list[CounterfactualFamilyRecord] = []
    for family_id, siblings in sorted(action_groups.items()):
        if len(siblings) < 2:
            continue
        split_ids = {item.case.split_group_id for item in siblings}
        physics_hashes = {item.episode_plan.physics_hash for item in siblings}
        action_hashes = {item.episode_plan.action_hash for item in siblings}
        scene_hashes = {stable_hash(item.episode_plan.scene_parameters) for item in siblings}
        initial_hashes = {sha256_json(_initial_state_payload(item)) for item in siblings}
        appearance_hashes = {sha256_json(_appearance_payload(item)) for item in siblings}
        if len(split_ids) != 1 or len(physics_hashes) != 1:
            raise ValueError(f"action family {family_id} changes split group or physics")
        if len(action_hashes) < 2:
            raise ValueError(
                f"action family {family_id} has multiple labels but one action trajectory"
            )
        if len(scene_hashes) != 1 or len(initial_hashes) != 1 or len(appearance_hashes) != 1:
            raise ValueError(f"action family {family_id} changes initial state or appearance")
        record = CounterfactualFamilyRecord(
            family_id=family_id,
            relation=CounterfactualRelation.ACTION,
            split_group_id=next(iter(split_ids)),
            expected_member_count=len(siblings),
            expected_episode_uuids=sorted(item.episode_uuid for item in siblings),
            intervention_fields=["action"],
            fixed_field_hashes={
                "scene_seed": sha256_json(siblings[0].episode_plan.scene_seed),
                "appearance": next(iter(appearance_hashes)),
                "split_group_id": sha256_json(siblings[0].episode_plan.split_group_id),
                "initial_state": next(iter(initial_hashes)),
                "physics": next(iter(physics_hashes)),
            },
            expected_member_plan_hashes={
                item.episode_uuid: item.episode_plan.config_hash for item in siblings
            },
        )
        record.validate()
        declarations.append(record)

    for family_id, siblings in sorted(physics_groups.items()):
        if len(siblings) < 2:
            continue
        split_ids = {item.case.split_group_id for item in siblings}
        action_hashes = {item.episode_plan.action_hash for item in siblings}
        scene_hashes = {stable_hash(item.episode_plan.scene_parameters) for item in siblings}
        initial_hashes = {sha256_json(_initial_state_payload(item)) for item in siblings}
        appearance_hashes = {sha256_json(_appearance_payload(item)) for item in siblings}
        interventions = {item.case.physics_intervention_fields for item in siblings}
        if len(split_ids) != 1 or len(action_hashes) != 1:
            raise ValueError(f"physics family {family_id} changes split group or replayed action")
        if len(scene_hashes) != 1 or len(initial_hashes) != 1 or len(appearance_hashes) != 1:
            raise ValueError(f"physics family {family_id} changes initial state or appearance")
        if len(interventions) != 1 or not next(iter(interventions)):
            raise ValueError(f"physics family {family_id} has no unique named intervention")
        intervention_fields = next(iter(interventions))
        for field_name in intervention_fields:
            missing_members = [
                item.episode_uuid
                for item in siblings
                if field_name not in item.episode_plan.physics
            ]
            if missing_members:
                raise ValueError(
                    f"physics family {family_id} is missing declared intervention "
                    f"{field_name} in members {sorted(missing_members)}"
                )
            intervention_values = {
                sha256_json(item.episode_plan.physics[field_name])
                for item in siblings
            }
            if len(intervention_values) < 2:
                raise ValueError(
                    f"physics family {family_id} does not vary declared intervention "
                    f"{field_name}"
                )
        nonintervened_hashes = {
            sha256_json(
                _nonintervened_physics(item.episode_plan.physics, intervention_fields)
            )
            for item in siblings
        }
        if len(nonintervened_hashes) != 1:
            raise ValueError(f"physics family {family_id} changes undeclared physics")
        record = CounterfactualFamilyRecord(
            family_id=family_id,
            relation=CounterfactualRelation.PHYSICS,
            split_group_id=next(iter(split_ids)),
            expected_member_count=len(siblings),
            expected_episode_uuids=sorted(item.episode_uuid for item in siblings),
            intervention_fields=list(intervention_fields),
            fixed_field_hashes={
                "scene_seed": sha256_json(siblings[0].episode_plan.scene_seed),
                "appearance": next(iter(appearance_hashes)),
                "split_group_id": sha256_json(siblings[0].episode_plan.split_group_id),
                "initial_state": next(iter(initial_hashes)),
                "action": next(iter(action_hashes)),
                "nonintervened_physics": next(iter(nonintervened_hashes)),
            },
            expected_member_plan_hashes={
                item.episode_uuid: item.episode_plan.config_hash for item in siblings
            },
        )
        record.validate()
        declarations.append(record)
    return sorted(declarations, key=lambda value: (value.relation.value, value.family_id))


def plan_suite_cases(cases: Iterable[SuiteCase]) -> list[PlannedSuiteEpisode]:
    """Plan rigid native and deformable/legacy diagnostic suite cases.

    The returned list preserves input ordering.  Planning is deterministic and
    performs symmetric counterfactual validation before returning.
    """

    case_values = list(cases)
    if len({(case.suite_name, case.case_index) for case in case_values}) != len(case_values):
        raise ValueError("suite case identities must be unique")
    planned: list[PlannedSuiteEpisode] = []
    for case in case_values:
        if case.backend != "native_mujoco":
            raise ValueError(
                f"acceptance case {case.case_index} requests unsupported backend {case.backend!r}"
            )
        if case.family in {
            "falling_catch",
            "rolling_interception",
            "projectile_rebound",
        }:
            planned.append(_native_plan(case))
        elif case.family in {"cloth", "rope", "legacy_proxy_quarantine"}:
            planned.append(_diagnostic_plan(case))
        else:
            raise ValueError(
                f"acceptance case {case.case_index} has unsupported family {case.family!r}"
            )
    if len({item.episode_uuid for item in planned}) != len(planned):
        raise ValueError("planned suite episode UUIDs are not unique")
    # Building declarations is also the preflight invariant check.  Persist
    # the exact pre-simulation hashes into each plan so finalized records can
    # be compared to the immutable ledger without hashing a different runtime
    # representation of the same concept.
    declarations = build_planned_counterfactual_family_records(planned)
    declarations_by_key = {
        (declaration.relation, declaration.family_id): declaration
        for declaration in declarations
    }
    annotated: list[PlannedSuiteEpisode] = []
    for item in planned:
        hashes: dict[str, Mapping[str, str]] = {}
        action = declarations_by_key.get(
            (CounterfactualRelation.ACTION, item.case.counterfactual_bundle_id)
        )
        if action is not None:
            hashes[CounterfactualRelation.ACTION.value] = action.fixed_field_hashes
        if item.case.physics_counterfactual_family_id is not None:
            physics = declarations_by_key.get(
                (
                    CounterfactualRelation.PHYSICS,
                    item.case.physics_counterfactual_family_id,
                )
            )
            if physics is not None:
                hashes[CounterfactualRelation.PHYSICS.value] = (
                    physics.fixed_field_hashes
                )
        plan = replace(
            item.episode_plan,
            options={
                **item.episode_plan.options,
                "planned_counterfactual_fixed_hashes": {
                    relation: dict(values) for relation, values in hashes.items()
                },
            },
        )
        annotated.append(replace(item, episode_plan=plan))
    return annotated


__all__ = [
    "ExecutionBackend",
    "PHYSICS_SWEEP_VALUES",
    "PlannedSuiteEpisode",
    "build_planned_counterfactual_family_records",
    "plan_suite_cases",
]
