"""Deterministic minting of diagnostic ``sampled_scale`` episode cases.

This module parallels :mod:`dynamic_robot_dataset.common.review_suite` for
scale generation.  It never mutates the fixed 20-by-6 acceptance matrix, its
seed, or its UUID namespace: scale cases live in an independent, explicitly
diagnostic namespace, are stamped ``initial_state_mode="sampled_scale"``, and
inherit the honest per-case execution gate from the live registries.  Nothing
minted here is release or training evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Any, Mapping
import uuid

from ..scenarios._rigid_shared import (
    SCALE_ACCEPTED_SAMPLER_VERSIONS,
    SCALE_INITIAL_STATE_SAMPLER_VERSION,
    SCALE_SAMPLED_LEAVES,
    scale_sampler_version,
)
from .assets import RoboCasaCatalogPolicy
from .corpus_registry import (
    BackendCapabilityRegistry,
    CorpusRegistry,
    load_backend_capability_registry,
    load_corpus_registry,
)
from .hashing import canonical_json_bytes, sha256_json, stable_uint64
from .review import REVIEW_SCENE_SEQUENCE
from .review_suite import (
    ReviewSuiteValidationError,
    _case_execution_gate,
    _intended_outcome,
    _validate_uuid,
    load_review_robocasa_catalog,
)
from .source_scenario import RNGSubseeds, SOURCE_SCENARIO_SCHEMA_VERSION


SCALE_SUITE_SCHEMA = "dynamic-robot-scale-suite/v1"
SCALE_SUITE_MANIFEST_SCHEMA = "dynamic-robot-scale-suite-manifest/v1"

# Independent diagnostic identity stream.  The fixed-review master seed and
# UUID namespace remain untouched; this is a new stream, not a replacement.
SCALE_MASTER_SEED = 20260721
_SCALE_UUID_NAMESPACE = uuid.UUID("c9a1c9c3-52b8-5e0f-9d38-6f4d1d1e2a71")

_RNG_STREAMS = (
    "physics",
    "initial_state",
    "camera",
    "assets",
    "controller",
    "scene_construction",
)

# Deterministic 20-case rotation: 13 nominal successes, 4 initial-state
# negatives, 3 controller-timing negatives (65/20/15).  P0 leaves are always
# passive observations.
_BRANCH_CYCLE = 20
_BRANCH_NOMINAL_COUNT = 13
_BRANCH_INITIAL_STATE_COUNT = 4

# One clean-lab case per 20; the rest rotate the admitted RoboCasa profiles.
_CLEAN_R0_CYCLE = 20
_R1_SCENE_PROFILES = tuple(REVIEW_SCENE_SEQUENCE[1:])
# Calibration blocks 9001/9002 measured the storage scene occluding the F1
# catch zone at the key event for ~6% of sampled episodes (both views), so
# the F1 leaves rotate without it.  Other leaves keep the full sequence.
_R1_SCENE_PROFILES_BY_LEAF: dict[str, tuple[str, ...]] = {
    leaf_id: tuple(
        profile for profile in _R1_SCENE_PROFILES if profile != "robocasa_storage"
    )
    for leaf_id in ("F1a", "F1b", "F1c", "F1d")
}


def _canonical_copy(value: Any) -> Any:
    return json.loads(canonical_json_bytes(value).decode("utf-8"))


@dataclass(frozen=True, slots=True)
class ScaleSuiteCase:
    """One deterministic diagnostic scale episode request.

    Mirrors :class:`ReviewSuiteCase` field-for-field and adds the sampled
    initial-state identity.  It is intentionally a distinct type: the fixed
    review executor requires a genuine ``ReviewSuiteCase`` and fails closed on
    scale declarations, and extending the review dataclass would silently
    change every fixed case's ``case_sha256``.
    """

    case_id: str
    episode_uuid: str
    episode_index: int
    corpus_leaf_id: str
    family: str
    subfamily: str
    backend: str
    backend_release_state: str
    backend_blockers: tuple[str, ...]
    support_execution_state: str
    support_implemented_task_variants: tuple[str, ...]
    support_blockers: tuple[str, ...]
    leaf_release_state: str
    leaf_blockers: tuple[str, ...]
    evaluator: str
    scale_index: int
    scene_profile: str
    randomization_level: str
    requires_real_robocasa: bool
    embodiment: str
    task_variant: str
    branch_role: str
    intended_outcome: str
    passive_variation_profile: str | None
    rng_subseeds: RNGSubseeds
    counterfactual_bundle_id: str
    counterfactual_branch_id: str
    counterfactual_sibling_index: int
    execution_eligible: bool
    executable_quota: int
    execution_blockers: tuple[str, ...]
    scale_suite_id: str
    initial_state_mode: str = "sampled_scale"
    sampler_version: str = SCALE_INITIAL_STATE_SAMPLER_VERSION
    schema_version: str = SCALE_SUITE_SCHEMA
    source_scenario_schema_version: str = SOURCE_SCENARIO_SCHEMA_VERSION

    def _identity_dict(self) -> dict[str, Any]:
        return _canonical_copy(asdict(self))

    @property
    def case_sha256(self) -> str:
        return sha256_json(self._identity_dict())

    def validate(self) -> None:
        if self.schema_version != SCALE_SUITE_SCHEMA:
            raise ReviewSuiteValidationError(
                f"scale case must use {SCALE_SUITE_SCHEMA}"
            )
        if self.initial_state_mode != "sampled_scale":
            raise ReviewSuiteValidationError(
                "scale cases must declare the sampled_scale initial-state mode"
            )
        if self.sampler_version not in SCALE_ACCEPTED_SAMPLER_VERSIONS:
            raise ReviewSuiteValidationError(
                "scale case names an unsupported initial-state sampler version"
            )
        if self.corpus_leaf_id not in SCALE_SAMPLED_LEAVES:
            raise ReviewSuiteValidationError(
                f"{self.corpus_leaf_id} has no scale initial-state sampler"
            )
        if self.backend != "source_mujoco":
            raise ReviewSuiteValidationError("scale cases require source_mujoco")
        if self.episode_index < 0 or self.scale_index < 0:
            raise ReviewSuiteValidationError("scale case indices must be non-negative")
        if self.executable_quota not in {0, 1} or (
            self.execution_eligible != bool(self.executable_quota)
        ):
            raise ReviewSuiteValidationError("scale case quota must be zero or one")
        if not self.scale_suite_id:
            raise ReviewSuiteValidationError("scale case lacks its suite identity")
        _validate_uuid(self.episode_uuid)
        self.rng_subseeds.validate()

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {**self._identity_dict(), "case_sha256": self.case_sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ScaleSuiteCase":
        subseeds = value.get("rng_subseeds")
        if not isinstance(subseeds, Mapping):
            raise ReviewSuiteValidationError("scale case RNG sub-seeds must be a mapping")
        result = cls(
            case_id=str(value.get("case_id", "")),
            episode_uuid=str(value.get("episode_uuid", "")),
            episode_index=int(value.get("episode_index", -1)),
            corpus_leaf_id=str(value.get("corpus_leaf_id", "")),
            family=str(value.get("family", "")),
            subfamily=str(value.get("subfamily", "")),
            backend=str(value.get("backend", "")),
            backend_release_state=str(value.get("backend_release_state", "")),
            backend_blockers=tuple(str(item) for item in value.get("backend_blockers", ())),
            support_execution_state=str(value.get("support_execution_state", "")),
            support_implemented_task_variants=tuple(
                str(item) for item in value.get("support_implemented_task_variants", ())
            ),
            support_blockers=tuple(str(item) for item in value.get("support_blockers", ())),
            leaf_release_state=str(value.get("leaf_release_state", "")),
            leaf_blockers=tuple(str(item) for item in value.get("leaf_blockers", ())),
            evaluator=str(value.get("evaluator", "")),
            scale_index=int(value.get("scale_index", -1)),
            scene_profile=str(value.get("scene_profile", "")),
            randomization_level=str(value.get("randomization_level", "")),
            requires_real_robocasa=bool(value.get("requires_real_robocasa")),
            embodiment=str(value.get("embodiment", "")),
            task_variant=str(value.get("task_variant", "")),
            branch_role=str(value.get("branch_role", "")),
            intended_outcome=str(value.get("intended_outcome", "")),
            passive_variation_profile=(
                None
                if value.get("passive_variation_profile") is None
                else str(value.get("passive_variation_profile"))
            ),
            rng_subseeds=RNGSubseeds.from_dict(subseeds),
            counterfactual_bundle_id=str(value.get("counterfactual_bundle_id", "")),
            counterfactual_branch_id=str(value.get("counterfactual_branch_id", "")),
            counterfactual_sibling_index=int(value.get("counterfactual_sibling_index", -1)),
            execution_eligible=bool(value.get("execution_eligible")),
            executable_quota=int(value.get("executable_quota", -1)),
            execution_blockers=tuple(str(item) for item in value.get("execution_blockers", ())),
            scale_suite_id=str(value.get("scale_suite_id", "")),
            initial_state_mode=str(value.get("initial_state_mode", "")),
            sampler_version=str(value.get("sampler_version", "")),
            schema_version=str(value.get("schema_version", "")),
            source_scenario_schema_version=str(
                value.get("source_scenario_schema_version", "")
            ),
        )
        if value.get("case_sha256") != result.case_sha256:
            raise ReviewSuiteValidationError(
                f"scale case hash mismatch for {result.case_id or '<unknown>'}"
            )
        result.validate()
        return result


def _scale_rng_subseeds(leaf_id: str, scale_index: int) -> RNGSubseeds:
    identity = {
        "master_seed": SCALE_MASTER_SEED,
        "leaf_id": leaf_id,
        "scale_index": scale_index,
    }
    values = {
        stream: stable_uint64(
            {**identity, "stream": stream}, namespace=SCALE_SUITE_SCHEMA
        )
        for stream in _RNG_STREAMS
    }
    result = RNGSubseeds(**values)
    result.validate()
    return result


def _scale_scene_profile(leaf_id: str, scale_index: int) -> str:
    if scale_index % _CLEAN_R0_CYCLE == 0:
        return "clean_R0"
    profiles = _R1_SCENE_PROFILES_BY_LEAF.get(leaf_id, _R1_SCENE_PROFILES)
    # Deterministic hash-based pick.  A modular rotation aliases with the
    # 20-case branch/embodiment cycle (calibration block-9003 measured one
    # branch/embodiment/scene combination recurring at every 20th case), so
    # the scene stream is decorrelated from the other rotations instead.
    selector = stable_uint64(
        {"leaf_id": leaf_id, "scale_index": scale_index, "stream": "scene"},
        namespace=SCALE_SUITE_SCHEMA,
    )
    return profiles[selector % len(profiles)]


def _scale_branch_role(leaf_id: str, scale_index: int, embodiment: str) -> str:
    if leaf_id.startswith("P0"):
        return "passive_observation"
    position = scale_index % _BRANCH_CYCLE
    if position < _BRANCH_NOMINAL_COUNT:
        return "nominal_success"
    if position < _BRANCH_NOMINAL_COUNT + _BRANCH_INITIAL_STATE_COUNT:
        return "deterministic_negative_initial_state"
    # Calibration blocks 9003/9004 measured the Panda controller-timing
    # negative on F1 leaves deflecting the ball out of every scene's framing
    # (final-checkpoint visibility fails deterministically).  Those slots use
    # the consistently in-frame initial-state negative instead; the Robotiq
    # keeps its controller-timing coverage.
    if leaf_id.startswith("F1") and embodiment == "franka_hand":
        return "deterministic_negative_initial_state"
    return "deterministic_negative_controller_timing"


def _scale_embodiment(leaf, scale_index: int) -> str:
    if leaf.corpus_id.startswith("P0"):
        return "no_robot"
    if leaf.supported_embodiments == ("franka_hand", "robotiq_2f85_thick_pad"):
        # F1b calibration block 9000 measured the Panda off-center catch
        # wedging the ball (16/50 nominals at 6.6-8.9 mm gripper
        # penetration) while its Robotiq near-miss lane was clean, and F1a
        # proves the Robotiq catch at scale.  F1b therefore flips the
        # embodiment phase so the catch variant lands on the Robotiq and
        # the near-miss lane on the Panda.
        if leaf.corpus_id == "F1b":
            return leaf.supported_embodiments[(scale_index + 1) % 2]
        return leaf.supported_embodiments[scale_index % 2]
    if leaf.supported_embodiments == ("franka_hand",):
        return "franka_hand"
    raise ReviewSuiteValidationError(
        f"{leaf.corpus_id} has no scale embodiment rotation for "
        f"{leaf.supported_embodiments}"
    )


# Scale-only task-variant restrictions.  The fixed review keeps every
# implemented variant; scale minting samples only lanes whose sampled physics
# passed calibration.  F2c block 9000: every sampled floor_bounce nominal rode
# the 2 mm gripper-penetration threshold (1.74-2.15 mm, 11/30 over) and none
# retained the catch to the final frame, while table_bounce passed 35/35, so
# scale samples the proven table lane until the floor lane is repaired.
_SCALE_TASK_VARIANT_OVERRIDES: dict[str, tuple[str, ...]] = {
    "F2c": ("table_bounce",),
}


def _scale_task_variants(leaf) -> tuple[str, ...]:
    return _SCALE_TASK_VARIANT_OVERRIDES.get(
        leaf.corpus_id, tuple(leaf.task_variants)
    )


def mint_scale_cases(
    leaf_id: str,
    *,
    episode_start: int,
    count: int,
    scale_suite_id: str,
    corpus_registry: CorpusRegistry | None = None,
    backend_registry: BackendCapabilityRegistry | None = None,
    robocasa_catalog: RoboCasaCatalogPolicy | None = None,
) -> tuple[ScaleSuiteCase, ...]:
    """Mint one contiguous block of deterministic diagnostic scale cases.

    ``episode_start`` is the leaf-global index of the first case; the minted
    ``episode_index`` values are block-local and contiguous from zero so they
    can be handed directly to ``plan_run``.  Minting fails closed if any case
    would be execution-ineligible: a scale run must never silently burn plan
    membership on blocked cases.
    """

    if count <= 0 or episode_start < 0:
        raise ReviewSuiteValidationError("scale minting requires a positive block")
    if leaf_id not in SCALE_SAMPLED_LEAVES:
        raise ReviewSuiteValidationError(
            f"{leaf_id} has no scale initial-state sampler"
        )
    corpus = corpus_registry or load_corpus_registry()
    backends = backend_registry or load_backend_capability_registry(corpus=corpus)
    catalog = robocasa_catalog or load_review_robocasa_catalog()
    leaf = corpus.resolve(leaf_id)
    cases: list[ScaleSuiteCase] = []
    for block_index in range(count):
        scale_index = episode_start + block_index
        embodiment = _scale_embodiment(leaf, scale_index)
        scale_variants = _scale_task_variants(leaf)
        task_variant = scale_variants[scale_index % len(scale_variants)]
        branch_role = _scale_branch_role(leaf_id, scale_index, embodiment)
        scene_profile = _scale_scene_profile(leaf_id, scale_index)
        requires_real_robocasa = scene_profile != "clean_R0"
        backend = backends.resolve(leaf.backend, leaf.corpus_id, embodiment)
        support = backend.support_by_leaf[leaf.corpus_id]
        eligible, hard_blockers = _case_execution_gate(
            backend,
            support,
            task_variant=task_variant,
            requires_real_robocasa=requires_real_robocasa,
            robocasa_catalog=catalog,
        )
        if not eligible:
            raise ReviewSuiteValidationError(
                f"scale case {leaf_id}#{scale_index} is not executable: "
                + ", ".join(hard_blockers)
            )
        case_id = f"{leaf_id}-scale-{scale_index:06d}"
        case = ScaleSuiteCase(
            case_id=case_id,
            episode_uuid=str(
                uuid.uuid5(
                    _SCALE_UUID_NAMESPACE,
                    f"{corpus.registry_id}/{SCALE_MASTER_SEED}/{case_id}",
                )
            ),
            episode_index=block_index,
            corpus_leaf_id=leaf.corpus_id,
            family=leaf.family,
            subfamily=leaf.subfamily,
            backend=leaf.backend,
            backend_release_state=backend.release_state.value,
            backend_blockers=backend.blockers,
            support_execution_state=support.execution_state.value,
            support_implemented_task_variants=support.implemented_task_variants,
            support_blockers=support.blockers,
            leaf_release_state=leaf.release_state.value,
            leaf_blockers=leaf.blockers,
            evaluator=leaf.evaluator,
            scale_index=scale_index,
            scene_profile=scene_profile,
            randomization_level="R0" if scene_profile == "clean_R0" else "R1",
            requires_real_robocasa=requires_real_robocasa,
            embodiment=embodiment,
            task_variant=task_variant,
            branch_role=branch_role,
            intended_outcome=_intended_outcome(branch_role),
            passive_variation_profile=(
                "nominal" if leaf_id.startswith("P0") else None
            ),
            rng_subseeds=_scale_rng_subseeds(leaf_id, scale_index),
            counterfactual_bundle_id=f"scale-{leaf_id}-{scale_index:06d}",
            counterfactual_branch_id=branch_role,
            counterfactual_sibling_index=0,
            execution_eligible=eligible,
            executable_quota=int(eligible),
            execution_blockers=hard_blockers,
            scale_suite_id=scale_suite_id,
            sampler_version=scale_sampler_version(leaf_id),
        )
        case.validate()
        cases.append(case)
    return tuple(cases)


def scale_block_manifest(cases: tuple[ScaleSuiteCase, ...]) -> dict[str, Any]:
    """Content-bound identity of one minted block for run-config provenance."""

    if not cases:
        raise ReviewSuiteValidationError("a scale block requires at least one case")
    leaf_ids = {case.corpus_leaf_id for case in cases}
    suite_ids = {case.scale_suite_id for case in cases}
    if len(leaf_ids) != 1 or len(suite_ids) != 1:
        raise ReviewSuiteValidationError("a scale block covers exactly one leaf/suite")
    if [case.episode_index for case in cases] != list(range(len(cases))):
        raise ReviewSuiteValidationError("scale block episode indices must be contiguous")
    manifest = {
        "schema_version": SCALE_SUITE_MANIFEST_SCHEMA,
        "scale_suite_id": cases[0].scale_suite_id,
        "corpus_leaf_id": cases[0].corpus_leaf_id,
        "master_seed": SCALE_MASTER_SEED,
        "sampler_version": scale_sampler_version(cases[0].corpus_leaf_id),
        "episode_start": cases[0].scale_index,
        "episode_count": len(cases),
        "case_sha256": [case.case_sha256 for case in cases],
        "training_eligible": False,
        "diagnostic_only": True,
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    return _canonical_copy(manifest)


__all__ = [
    "SCALE_MASTER_SEED",
    "SCALE_SUITE_MANIFEST_SCHEMA",
    "SCALE_SUITE_SCHEMA",
    "ScaleSuiteCase",
    "mint_scale_cases",
    "scale_block_manifest",
]
