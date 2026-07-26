"""Orchestration bridge for diagnostic ``ScaleSuiteCase`` declarations.

Parallels :mod:`dynamic_robot_dataset.common.source_execution` for the
sampled-scale corpus cases.  Declarations carry their own schema version so
run-shard dispatch selects exactly one bridge per run plan.  All shared
verification, normalization, evaluation, and labeling logic is reused from
the canonical bridge unchanged: every scale episode remains diagnostic,
review-only, and training-ineligible.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from ..backends import get_backend
from ..backends.source_mujoco import SourceMujocoBackend, prepare_review_case
from .hashing import canonical_json_bytes, sha256_json
from .run_orchestration import EpisodeMaterialization, RunPlanEpisode
from .scale_suite import ScaleSuiteCase
from .source_execution import (
    SourceExecutionBindingError,
    _materialize_verified,
    _metadata_seed,
    _task_index,
    _verify_prepared_identity,
)
from .source_scenario import SourceScenarioSpec


SCALE_EXECUTION_BRIDGE_SCHEMA = "dynamic-robot-scale-execution-bridge/v1"


def _canonical_copy(value: Any) -> Any:
    return json.loads(canonical_json_bytes(value).decode("utf-8"))


def _require_executable_scale_case(case: ScaleSuiteCase) -> None:
    if not isinstance(case, ScaleSuiteCase):
        raise TypeError("scale planning requires an immutable ScaleSuiteCase")
    if case.backend != "source_mujoco":
        raise ValueError(
            f"scale case {case.case_id} uses {case.backend}, not source_mujoco"
        )
    if case.initial_state_mode != "sampled_scale":
        raise ValueError(
            f"scale case {case.case_id} does not declare sampled_scale initial states"
        )
    if not case.execution_eligible or case.executable_quota != 1:
        raise ValueError(
            f"scale case {case.case_id} is not admitted for execution: "
            + ", ".join(case.execution_blockers)
        )


def _scale_backend(case: ScaleSuiteCase) -> SourceMujocoBackend:
    """Resolve the owned backend through the capability gate.

    Scale generation uses the registry's ``review`` purpose, the only
    non-release purpose admitted for these leaves; the run configuration
    separately records ``purpose="scale_generation"`` for provenance.
    Pilot and production resolution remain fail-closed and untouched.
    """

    backend = get_backend(
        "source_mujoco",
        purpose="review",
        corpus_leaf_id=case.corpus_leaf_id,
        embodiment=case.embodiment,
        task_variant=case.task_variant,
    )
    if not isinstance(backend, SourceMujocoBackend):
        raise TypeError("capability factory returned a non-owned source_mujoco backend")
    return backend


def prepare_source_scale_declaration(
    case: ScaleSuiteCase,
    *,
    episode_index: int,
    generator_git_commit: str = "unknown",
) -> dict[str, Any]:
    """Prepare one immutable scale declaration for :func:`plan_run`."""

    _require_executable_scale_case(case)
    if isinstance(episode_index, bool) or episode_index < 0:
        raise ValueError("episode_index must be a non-negative contiguous plan index")
    backend = _scale_backend(case)
    scenario_spec = prepare_review_case(case, backend=backend)
    compiled = backend.compile_case(case)
    scenario_spec.validate()
    compiled.validate()
    if scenario_spec.scenario_id != case.case_id:
        raise SourceExecutionBindingError(
            "prepared SourceScenarioSpec does not preserve the logical scale case ID"
        )
    if compiled.episode_uuid != case.episode_uuid:
        raise SourceExecutionBindingError(
            "compiled source scenario does not preserve the scale episode UUID"
        )
    runtime_source_hashes = {
        **dict(scenario_spec.source_hashes),
        "robocasa_license": backend.robocasa_dependency.license_sha256,
    }
    declaration = {
        "schema_version": SCALE_EXECUTION_BRIDGE_SCHEMA,
        "episode_uuid": case.episode_uuid,
        "episode_index": int(episode_index),
        "scale_suite_episode_index": case.episode_index,
        "scale_suite_id": case.scale_suite_id,
        "scale_index": case.scale_index,
        "scale_case": case.to_dict(),
        "scale_case_sha256": case.case_sha256,
        "backend": "source_mujoco",
        "backend_version": backend.version,
        "corpus_leaf_id": case.corpus_leaf_id,
        "family": case.family,
        "subfamily": case.subfamily,
        "task_variant": case.task_variant,
        "variant": case.task_variant,
        "embodiment": case.embodiment,
        "robot_model": scenario_spec.embodiment.robot_model,
        "tool_type": scenario_spec.embodiment.end_effector,
        "duration_s": scenario_spec.duration_s,
        "task_index": _task_index(case.corpus_leaf_id),
        "counterfactual_bundle_id": case.counterfactual_bundle_id,
        "counterfactual_branch_id": case.counterfactual_branch_id,
        "counterfactual_sibling_index": case.counterfactual_sibling_index,
        "physics_counterfactual_family_id": (
            scenario_spec.counterfactual.physics_family_id
        ),
        "split_group_id": scenario_spec.counterfactual.split_group_id,
        "scene_seed": _metadata_seed(
            scenario_spec.rng_subseeds.scene_construction
        ),
        "branch_seed": _metadata_seed(scenario_spec.rng_subseeds.controller),
        "intended_branch": case.branch_role,
        "intended_outcome": case.intended_outcome,
        "generator_git_commit": str(generator_git_commit),
        "source_scenario_spec": scenario_spec.to_dict(),
        "source_scenario_spec_sha256": scenario_spec.spec_hash,
        "source_compiled_scenario": compiled.to_dict(),
        "source_compiled_scenario_sha256": compiled.scenario_sha256,
        "runtime_source_hashes": dict(sorted(runtime_source_hashes.items())),
        "runtime_source_hashes_sha256": sha256_json(runtime_source_hashes),
    }
    return _canonical_copy(declaration)


def _planned_scale_inputs(
    entry: RunPlanEpisode,
) -> tuple[ScaleSuiteCase, SourceScenarioSpec]:
    declaration = entry.declaration
    if declaration.get("schema_version") != SCALE_EXECUTION_BRIDGE_SCHEMA:
        raise SourceExecutionBindingError("run-plan entry is not a scale bridge request")
    if int(declaration.get("episode_index", -1)) != entry.episode_index:
        raise SourceExecutionBindingError("declaration episode index differs from its run plan")
    if str(declaration.get("episode_uuid") or "") != entry.episode_uuid:
        raise SourceExecutionBindingError("declaration episode UUID differs from its run plan")
    raw_case = declaration.get("scale_case")
    raw_spec = declaration.get("source_scenario_spec")
    raw_compiled = declaration.get("source_compiled_scenario")
    if not isinstance(raw_case, Mapping):
        raise SourceExecutionBindingError("scale declaration lacks its immutable case")
    if not isinstance(raw_spec, Mapping):
        raise SourceExecutionBindingError("scale declaration lacks SourceScenarioSpec")
    if not isinstance(raw_compiled, Mapping):
        raise SourceExecutionBindingError("scale declaration lacks compiled scenario identity")
    case = ScaleSuiteCase.from_dict(raw_case)
    _require_executable_scale_case(case)
    spec = SourceScenarioSpec.from_dict(raw_spec)
    if case.episode_uuid != entry.episode_uuid:
        raise SourceExecutionBindingError("scale case episode UUID differs from the run plan")
    if int(declaration.get("scale_suite_episode_index", -1)) != case.episode_index:
        raise SourceExecutionBindingError("scale-suite episode index binding changed")
    if declaration.get("scale_case_sha256") != case.case_sha256:
        raise SourceExecutionBindingError("scale-case hash binding changed")
    if entry.source_scenario_spec_sha256 != spec.spec_hash or declaration.get(
        "source_scenario_spec_sha256"
    ) != spec.spec_hash:
        raise SourceExecutionBindingError("SourceScenarioSpec hash binding changed")
    if spec.scenario_id != case.case_id:
        raise SourceExecutionBindingError("SourceScenarioSpec logical case identity changed")
    expected_outer = {
        "backend": spec.backend,
        "corpus_leaf_id": spec.corpus_leaf_id,
        "task_variant": spec.task_variant,
        "embodiment": spec.embodiment.end_effector,
        "duration_s": spec.duration_s,
        "family": case.family,
        "subfamily": case.subfamily,
        "variant": case.task_variant,
        "task_index": _task_index(case.corpus_leaf_id),
        "counterfactual_bundle_id": case.counterfactual_bundle_id,
        "counterfactual_branch_id": case.counterfactual_branch_id,
        "counterfactual_sibling_index": case.counterfactual_sibling_index,
        "physics_counterfactual_family_id": spec.counterfactual.physics_family_id,
        "split_group_id": spec.counterfactual.split_group_id,
        "scale_suite_id": case.scale_suite_id,
        "scale_index": case.scale_index,
    }
    for name, expected in expected_outer.items():
        if declaration.get(name) != expected:
            raise SourceExecutionBindingError(
                f"scale declaration {name} differs from its immutable inputs"
            )
    return case, spec


def execute_source_mujoco_scale_episode(
    entry: RunPlanEpisode,
    *,
    render: bool = True,
) -> EpisodeMaterialization:
    """Execute an immutable scale run-plan entry through the gated backend."""

    case, spec = _planned_scale_inputs(entry)
    backend = _scale_backend(case)
    compiled = _verify_prepared_identity(entry, backend, case, spec)
    result = backend.run(compiled, render=render)
    return _materialize_verified(
        entry, result, case, spec, case_payload_key="scale_case"
    )


__all__ = [
    "SCALE_EXECUTION_BRIDGE_SCHEMA",
    "execute_source_mujoco_scale_episode",
    "prepare_source_scale_declaration",
]
