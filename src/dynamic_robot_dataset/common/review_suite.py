"""Deterministic planning for the fixed 20-by-6 corpus review suite.

This module plans acceptance work; it does not simulate episodes, synthesize
media, or change release state.  A plan always contains the complete 120-case
matrix so missing implementations stay visible.  Review execution is admitted
per leaf by backend support entries; backend and corpus release states remain
independent pilot/production gates.

The companion request ledger deliberately contains no placeholder artifact
hashes or human decisions.  It binds each future review to the immutable case
request and names the hashes that must be supplied after real artifacts exist.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
import uuid

import yaml

from ..backends.actuator_only import action_spec
from .assets import RoboCasaCatalogPolicy, load_robocasa_catalog_policy
from .corpus_registry import (
    BackendCapability,
    BackendCapabilityRegistry,
    BackendSupport,
    CorpusLeaf,
    CorpusRegistry,
    EXPECTED_CORPUS_LEAVES,
    load_backend_capability_registry,
    load_corpus_registry,
)
from .hashing import canonical_json_bytes, sha256_file, sha256_json, stable_uint64
from .paths import ExistingOutputError, atomic_write_json, ensure_not_source_path
from .review import (
    REVIEW_ARTIFACT_SCHEMA,
    REVIEW_CHECKS,
    REVIEW_LEDGER_SCHEMA,
    REVIEW_SCENE_SEQUENCE,
    EventStripSpec,
)
from .source_scenario import RNGSubseeds, SOURCE_SCENARIO_SCHEMA_VERSION


REVIEW_CONFIG_SCHEMA = "dynamic-robot-review-suite/v1"
REVIEW_SUITE_PLAN_SCHEMA = "dynamic-robot-review-suite-plan/v1"
REVIEW_REQUEST_LEDGER_SCHEMA = "dynamic-robot-review-request-ledger/v1"
REVIEW_REQUEST_SCHEMA = "dynamic-robot-review-artifact-request/v1"
REVIEW_SUITE_BUNDLE_SCHEMA = "dynamic-robot-review-suite-bundle/v1"

REVIEW_PLAN_FILE = "review_suite_plan.json"
REVIEW_REQUEST_LEDGER_FILE = "review_requests.json"
REVIEW_BUNDLE_FILE = "review_suite_manifest.json"

FIXED_REVIEW_MASTER_SEED = 20260717
_REVIEW_UUID_NAMESPACE = uuid.UUID("d4de4476-e86b-5e4a-bdb1-3bbfcce932af")

_DUAL_EMBODIMENT_PATTERN = (
    "franka_hand",
    "robotiq_2f85_thick_pad",
    "franka_hand",
    "robotiq_2f85_thick_pad",
    "franka_hand",
    "robotiq_2f85_thick_pad",
)
_DUAL_BRANCH_PATTERN = (
    "nominal_success",
    "nominal_success",
    "deterministic_negative_initial_state",
    "deterministic_negative_initial_state",
    "deterministic_negative_controller_timing",
    "deterministic_negative_controller_timing",
)
_SINGLE_BRANCH_PATTERN = (
    "nominal_success",
    "deterministic_negative_initial_state",
    "nominal_success",
    "deterministic_negative_controller_timing",
    "nominal_success",
    "deterministic_negative_task_geometry",
)
_PASSIVE_VARIATION_PATTERN = (
    "nominal",
    "lower_initial_speed",
    "higher_initial_speed",
    "initial_spin",
    "lower_admitted_contact_parameter",
    "higher_admitted_contact_parameter",
)
_RNG_STREAMS = (
    "physics",
    "initial_state",
    "camera",
    "assets",
    "controller",
    "scene_construction",
)
_REQUIRED_ARTIFACT_HASH_FIELDS = (
    "review_plan_sha256",
    "review_case_sha256",
    "review_request_ledger_sha256",
    "review_request_sha256",
    "scenario_spec_sha256",
    "source_manifest_sha256",
    "qc_report_sha256",
    "qc_episode_result_sha256",
    "frame_timestamps_sha256",
    "video_sha256.main",
    "video_sha256.secondary",
    "event_strip_sha256.main",
    "event_strip_sha256.secondary",
)


class ReviewSuiteValidationError(ValueError):
    """A review plan/configuration is incomplete or no longer reproducible."""


def _canonical_copy(value: Any) -> Any:
    return json.loads(canonical_json_bytes(value).decode("utf-8"))


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_uuid(value: str) -> None:
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, TypeError, ValueError) as error:
        raise ReviewSuiteValidationError(f"invalid review episode UUID: {value!r}") from error
    if str(parsed) != value:
        raise ReviewSuiteValidationError("review episode UUID must use canonical lowercase text")


@dataclass(frozen=True, slots=True)
class ReviewSuiteConfig:
    """Strictly versioned acceptance-matrix policy."""

    scene_sequence: tuple[str, ...]
    views: tuple[str, ...]
    event_strip: EventStripSpec
    human_checks: tuple[str, ...]
    required_artifact_hash_fields: tuple[str, ...]
    replacement_seed_forbidden: bool
    increment_profile_version_after_repair: bool
    schema_version: str = REVIEW_CONFIG_SCHEMA
    rollouts_per_leaf: int = 6

    def validate(self) -> None:
        if self.schema_version != REVIEW_CONFIG_SCHEMA:
            raise ReviewSuiteValidationError(
                f"review config must use {REVIEW_CONFIG_SCHEMA}"
            )
        if self.rollouts_per_leaf != 6:
            raise ReviewSuiteValidationError("review suite requires exactly six cases per leaf")
        if self.scene_sequence != REVIEW_SCENE_SEQUENCE:
            raise ReviewSuiteValidationError("review config changed the fixed scene sequence")
        if self.views != ("main", "secondary"):
            raise ReviewSuiteValidationError("review suite requires main and secondary views")
        self.event_strip.validate()
        if self.human_checks != REVIEW_CHECKS:
            raise ReviewSuiteValidationError("review config changed canonical human checks")
        if self.required_artifact_hash_fields != _REQUIRED_ARTIFACT_HASH_FIELDS:
            raise ReviewSuiteValidationError("review config does not require every artifact hash")
        if not self.replacement_seed_forbidden:
            raise ReviewSuiteValidationError("replacement review seeds must remain forbidden")
        if not self.increment_profile_version_after_repair:
            raise ReviewSuiteValidationError("repairs must increment the physics/profile version")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return _canonical_copy(asdict(self))

    @property
    def config_sha256(self) -> str:
        return sha256_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ReviewSuiteConfig":
        expected_top_level = {
            "schema_version",
            "rollouts_per_leaf",
            "scene_sequence",
            "views",
            "event_strip",
            "human_checks",
            "binding",
            "activation",
        }
        if set(value) != expected_top_level:
            raise ReviewSuiteValidationError(
                "review config keys differ from the fixed acceptance contract"
            )
        strip = value.get("event_strip")
        binding = value.get("binding")
        activation = value.get("activation")
        if not isinstance(strip, Mapping) or not isinstance(binding, Mapping) or not isinstance(
            activation, Mapping
        ):
            raise ReviewSuiteValidationError(
                "event_strip, binding, and activation must be mappings"
            )
        if float(strip.get("event_offset_s", float("nan"))) != 0.0:
            raise ReviewSuiteValidationError("event strip must include the exact key-event frame")
        if strip.get("include_final") is not True:
            raise ReviewSuiteValidationError("event strip must include the final frame")
        post = tuple(float(item) for item in strip.get("post_event_offsets_s", ()))
        required_binding = {
            "require_review_plan_sha256": "review_plan_sha256",
            "require_review_case_sha256": "review_case_sha256",
            "require_review_request_ledger_sha256": "review_request_ledger_sha256",
            "require_review_request_sha256": "review_request_sha256",
            "require_scenario_spec_sha256": "scenario_spec_sha256",
            "require_source_manifest_sha256": "source_manifest_sha256",
            "require_qc_report_sha256": "qc_report_sha256",
            "require_qc_episode_result_sha256": "qc_episode_result_sha256",
            "require_frame_timestamps_sha256": "frame_timestamps_sha256",
            "require_video_sha256": ("video_sha256.main", "video_sha256.secondary"),
            "require_event_strip_sha256": (
                "event_strip_sha256.main",
                "event_strip_sha256.secondary",
            ),
        }
        if set(binding) != set(required_binding) or any(
            binding.get(name) is not True for name in required_binding
        ):
            raise ReviewSuiteValidationError("every review artifact binding must be required")
        required_hashes: list[str] = []
        for name in required_binding:
            bound = required_binding[name]
            required_hashes.extend((bound,) if isinstance(bound, str) else bound)
        required_activation = {
            "require_all_automated_qc_passed",
            "require_all_human_checks_passed",
            "replacement_seed_forbidden",
            "increment_profile_version_after_repair",
        }
        if set(activation) != required_activation or any(
            activation.get(name) is not True for name in required_activation
        ):
            raise ReviewSuiteValidationError("review activation policy must remain fail-closed")
        result = cls(
            schema_version=str(value.get("schema_version", "")),
            rollouts_per_leaf=int(value.get("rollouts_per_leaf", 0)),
            scene_sequence=tuple(str(item) for item in value.get("scene_sequence", ())),
            views=tuple(str(item) for item in value.get("views", ())),
            event_strip=EventStripSpec(
                pre_event_offset_s=float(strip.get("pre_event_offset_s", float("nan"))),
                post_event_offsets_s=post,  # type: ignore[arg-type]
                include_event=True,
                include_final=True,
            ),
            human_checks=tuple(str(item) for item in value.get("human_checks", ())),
            required_artifact_hash_fields=tuple(required_hashes),
            replacement_seed_forbidden=bool(activation.get("replacement_seed_forbidden")),
            increment_profile_version_after_repair=bool(
                activation.get("increment_profile_version_after_repair")
            ),
        )
        result.validate()
        return result


def load_review_suite_config(path: str | Path | None = None) -> ReviewSuiteConfig:
    """Load the strict review policy; no alternate scene matrix is admitted."""

    source = (
        _repository_root() / "configs/review/acceptance_v1.yaml"
        if path is None
        else Path(path)
    )
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ReviewSuiteValidationError(f"review config is unavailable: {source}") from error
    if not isinstance(raw, Mapping):
        raise ReviewSuiteValidationError("review config must be a mapping")
    return ReviewSuiteConfig.from_dict(raw)


def load_review_robocasa_catalog(
    path: str | Path | None = None,
) -> RoboCasaCatalogPolicy:
    """Load and content-bind the read-only catalog used by fixed review scenes."""

    source = (
        _repository_root() / "configs/assets/robocasa_catalog_v1.yaml"
        if path is None
        else Path(path)
    )
    return load_robocasa_catalog_policy(source)


def _case_execution_gate(
    backend: BackendCapability,
    support: BackendSupport,
    *,
    task_variant: str,
    requires_real_robocasa: bool,
    robocasa_catalog: RoboCasaCatalogPolicy,
) -> tuple[bool, tuple[str, ...]]:
    blockers: list[str] = []
    if task_variant not in support.implemented_task_variants:
        blockers.append(f"support_task_variant_not_implemented:{task_variant}")
    if not backend.source_hashes:
        blockers.append("backend_source_hashes_unpinned")
    if requires_real_robocasa and not robocasa_catalog.review_ready:
        blockers.append("robocasa_catalog_not_review_ready")
    return not blockers, tuple(blockers)


def _rng_subseeds(leaf_id: str, rollout_index: int) -> RNGSubseeds:
    identity = {
        "master_seed": FIXED_REVIEW_MASTER_SEED,
        "leaf_id": leaf_id,
        "rollout_index": rollout_index,
    }
    values = {
        stream: stable_uint64(
            {**identity, "stream": stream}, namespace="dynamic-robot-review-suite/v1"
        )
        for stream in _RNG_STREAMS
    }
    result = RNGSubseeds(**values)
    result.validate()
    return result


def _assignment(
    leaf: CorpusLeaf,
    rollout_index: int,
) -> tuple[str, str, str, str | None]:
    """Return embodiment, task variant, branch role, passive profile."""

    if rollout_index < 0 or rollout_index >= 6:
        raise ReviewSuiteValidationError("review rollout index must be in [0, 6)")
    if leaf.corpus_id.startswith("P0"):
        return (
            "no_robot",
            leaf.task_variants[rollout_index % len(leaf.task_variants)],
            "passive_observation",
            _PASSIVE_VARIATION_PATTERN[rollout_index],
        )
    if leaf.supported_embodiments == (
        "franka_hand",
        "robotiq_2f85_thick_pad",
    ):
        variant_group = rollout_index // 2
        return (
            _DUAL_EMBODIMENT_PATTERN[rollout_index],
            leaf.task_variants[variant_group % len(leaf.task_variants)],
            _DUAL_BRANCH_PATTERN[rollout_index],
            None,
        )
    if leaf.supported_embodiments == ("franka_hand",):
        return (
            "franka_hand",
            leaf.task_variants[rollout_index % len(leaf.task_variants)],
            _SINGLE_BRANCH_PATTERN[rollout_index],
            None,
        )
    raise ReviewSuiteValidationError(
        f"{leaf.corpus_id} has no fixed review assignment for "
        f"{leaf.supported_embodiments}"
    )


def _intended_outcome(branch_role: str) -> str:
    if branch_role == "passive_observation":
        return "passive_observation"
    if branch_role == "nominal_success":
        return "success"
    return "failure"


def _embodiment_scenario_identity(embodiment: str) -> dict[str, Any]:
    """Return an ``EmbodimentSpec``-compatible applied-action declaration."""

    if embodiment == "no_robot":
        return {
            "end_effector": "no_robot",
            "robot_model": "none",
            "action_names": [],
            "action_semantics": "no_actuators/v1",
        }
    specification = action_spec(embodiment)
    robot_models = {
        "franka_hand": "franka_panda",
        "robotiq_2f85_thick_pad": "franka_panda_nohand_plus_robotiq_2f85",
    }
    return {
        "end_effector": embodiment,
        "robot_model": robot_models[embodiment],
        "action_names": list(specification.actuator_names),
        "action_semantics": specification.semantics,
    }


@dataclass(frozen=True, slots=True)
class ReviewSuiteCase:
    """One deterministic logical episode request in the acceptance matrix."""

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
    rollout_index: int
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
    source_scenario_schema_version: str = SOURCE_SCENARIO_SCHEMA_VERSION

    def _identity_dict(self) -> dict[str, Any]:
        return _canonical_copy(asdict(self))

    @property
    def case_sha256(self) -> str:
        return sha256_json(self._identity_dict())

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_dict(), "case_sha256": self.case_sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ReviewSuiteCase":
        subseeds = value.get("rng_subseeds")
        if not isinstance(subseeds, Mapping):
            raise ReviewSuiteValidationError("review case RNG sub-seeds must be a mapping")
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
            support_blockers=tuple(
                str(item) for item in value.get("support_blockers", ())
            ),
            leaf_release_state=str(value.get("leaf_release_state", "")),
            leaf_blockers=tuple(str(item) for item in value.get("leaf_blockers", ())),
            evaluator=str(value.get("evaluator", "")),
            rollout_index=int(value.get("rollout_index", -1)),
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
            execution_blockers=tuple(
                str(item) for item in value.get("execution_blockers", ())
            ),
            source_scenario_schema_version=str(
                value.get("source_scenario_schema_version", "")
            ),
        )
        if value.get("case_sha256") != result.case_sha256:
            raise ReviewSuiteValidationError(
                f"case hash mismatch for {result.case_id or '<unknown>'}"
            )
        return result


def _registry_sha256(value: CorpusRegistry | BackendCapabilityRegistry) -> str:
    return sha256_json(asdict(value))


@dataclass(frozen=True, slots=True)
class ReviewSuitePlan:
    """Content-addressed, complete acceptance matrix."""

    suite_id: str
    plan_sha256: str
    corpus_registry_id: str
    corpus_registry_sha256: str
    backend_registry_id: str
    backend_registry_sha256: str
    review_config_sha256: str
    robocasa_catalog_version: str
    robocasa_catalog_sha256: str
    robocasa_license_sha256: str
    robocasa_catalog_release_ready: bool
    fixed_master_seed: int
    cases: tuple[ReviewSuiteCase, ...]
    schema_version: str = REVIEW_SUITE_PLAN_SCHEMA

    def _identity_dict(self, *, include_suite_id: bool) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema_version": self.schema_version,
            "corpus_registry_id": self.corpus_registry_id,
            "corpus_registry_sha256": self.corpus_registry_sha256,
            "backend_registry_id": self.backend_registry_id,
            "backend_registry_sha256": self.backend_registry_sha256,
            "review_config_sha256": self.review_config_sha256,
            "robocasa_catalog_version": self.robocasa_catalog_version,
            "robocasa_catalog_sha256": self.robocasa_catalog_sha256,
            "robocasa_license_sha256": self.robocasa_license_sha256,
            "robocasa_catalog_release_ready": self.robocasa_catalog_release_ready,
            "fixed_master_seed": self.fixed_master_seed,
            "leaf_count": len({case.corpus_leaf_id for case in self.cases}),
            "cases_per_leaf": 6,
            "expected_case_count": len(self.cases),
            "expected_video_count": len(self.cases) * 2,
            "executable_case_count": sum(case.executable_quota for case in self.cases),
            "cases": [case.to_dict() for case in self.cases],
        }
        if include_suite_id:
            value["suite_id"] = self.suite_id
        return value

    def validate(
        self,
        *,
        corpus_registry: CorpusRegistry | None = None,
        backend_registry: BackendCapabilityRegistry | None = None,
        config: ReviewSuiteConfig | None = None,
        robocasa_catalog: RoboCasaCatalogPolicy | None = None,
    ) -> None:
        if self.schema_version != REVIEW_SUITE_PLAN_SCHEMA:
            raise ReviewSuiteValidationError(
                f"review plan must use {REVIEW_SUITE_PLAN_SCHEMA}"
            )
        corpus = corpus_registry or load_corpus_registry()
        backends = backend_registry or load_backend_capability_registry(corpus=corpus)
        policy = config or load_review_suite_config()
        catalog = robocasa_catalog or load_review_robocasa_catalog()
        policy.validate()
        corpus_binding_matches = (
            self.corpus_registry_id == corpus.registry_id
            and self.corpus_registry_sha256 == _registry_sha256(corpus)
        )
        if not corpus_binding_matches:
            raise ReviewSuiteValidationError("review plan is bound to a different corpus registry")
        backend_binding_matches = (
            self.backend_registry_id == backends.registry_id
            and self.backend_registry_sha256 == _registry_sha256(backends)
        )
        if not backend_binding_matches:
            raise ReviewSuiteValidationError("review plan is bound to a different backend registry")
        if self.review_config_sha256 != policy.config_sha256:
            raise ReviewSuiteValidationError("review plan is bound to a different review config")
        if (
            self.robocasa_catalog_version != catalog.catalog_version
            or self.robocasa_catalog_sha256 != catalog.catalog_sha256
            or self.robocasa_license_sha256 != catalog.license_notice_sha256
            or self.robocasa_catalog_release_ready != catalog.release_ready
        ):
            raise ReviewSuiteValidationError(
                "review plan is bound to a different RoboCasa catalog/license state"
            )
        if self.fixed_master_seed != FIXED_REVIEW_MASTER_SEED:
            raise ReviewSuiteValidationError("replacement review seeds are forbidden")
        if len(self.cases) != 120:
            raise ReviewSuiteValidationError("review plan must contain exactly 120 cases")

        expected_order = [
            (leaf.corpus_id, rollout_index)
            for leaf in corpus.leaves
            for rollout_index in range(6)
        ]
        actual_order = [(case.corpus_leaf_id, case.rollout_index) for case in self.cases]
        if actual_order != expected_order:
            raise ReviewSuiteValidationError(
                "review cases must follow registry order with indices zero through five"
            )
        if {case.corpus_leaf_id for case in self.cases} != EXPECTED_CORPUS_LEAVES:
            raise ReviewSuiteValidationError("review plan does not cover all 20 corpus leaves")
        case_ids = [case.case_id for case in self.cases]
        episode_uuids = [case.episode_uuid for case in self.cases]
        if len(case_ids) != len(set(case_ids)) or len(episode_uuids) != len(set(episode_uuids)):
            raise ReviewSuiteValidationError("review case and episode identities must be unique")

        for expected_episode_index, case in enumerate(self.cases):
            leaf = corpus.resolve(case.corpus_leaf_id)
            backend = backends.resolve(case.backend, case.corpus_leaf_id, case.embodiment)
            support = backend.support_by_leaf[case.corpus_leaf_id]
            embodiment, task_variant, branch_role, passive_profile = _assignment(
                leaf, case.rollout_index
            )
            eligible, hard_blockers = _case_execution_gate(
                backend,
                support,
                task_variant=task_variant,
                requires_real_robocasa=case.rollout_index > 0,
                robocasa_catalog=catalog,
            )
            expected = {
                "case_id": f"{leaf.corpus_id}-review-{case.rollout_index:02d}",
                "episode_index": expected_episode_index,
                "family": leaf.family,
                "subfamily": leaf.subfamily,
                "backend": leaf.backend,
                "backend_release_state": backend.release_state.value,
                "backend_blockers": backend.blockers,
                "support_execution_state": support.execution_state.value,
                "support_implemented_task_variants": support.implemented_task_variants,
                "support_blockers": support.blockers,
                "leaf_release_state": leaf.release_state.value,
                "leaf_blockers": leaf.blockers,
                "evaluator": leaf.evaluator,
                "scene_profile": policy.scene_sequence[case.rollout_index],
                "randomization_level": "R0" if case.rollout_index == 0 else "R1",
                "requires_real_robocasa": case.rollout_index > 0,
                "embodiment": embodiment,
                "task_variant": task_variant,
                "branch_role": branch_role,
                "intended_outcome": _intended_outcome(branch_role),
                "passive_variation_profile": passive_profile,
                "rng_subseeds": _rng_subseeds(leaf.corpus_id, case.rollout_index),
                "counterfactual_bundle_id": f"review-{leaf.corpus_id}-{case.rollout_index:02d}",
                "counterfactual_branch_id": branch_role,
                "counterfactual_sibling_index": 0,
                "execution_eligible": eligible,
                "executable_quota": int(eligible),
                "execution_blockers": hard_blockers,
                "source_scenario_schema_version": SOURCE_SCENARIO_SCHEMA_VERSION,
            }
            for field_name, expected_value in expected.items():
                if getattr(case, field_name) != expected_value:
                    raise ReviewSuiteValidationError(
                        f"{case.case_id} has non-canonical {field_name}"
                    )
            _validate_uuid(case.episode_uuid)
            expected_uuid = str(
                uuid.uuid5(
                    _REVIEW_UUID_NAMESPACE,
                    f"{corpus.registry_id}/{FIXED_REVIEW_MASTER_SEED}/{case.case_id}",
                )
            )
            if case.episode_uuid != expected_uuid:
                raise ReviewSuiteValidationError(f"{case.case_id} UUID is not deterministic")
            if case.executable_quota not in {0, 1}:
                raise ReviewSuiteValidationError("each review case quota must be zero or one")

        identity_without_suite = self._identity_dict(include_suite_id=False)
        expected_suite_id = "review-suite-" + sha256_json(identity_without_suite)[:24]
        if self.suite_id != expected_suite_id:
            raise ReviewSuiteValidationError("suite_id does not match review-plan contents")
        expected_plan_hash = sha256_json(self._identity_dict(include_suite_id=True))
        if self.plan_sha256 != expected_plan_hash:
            raise ReviewSuiteValidationError("plan_sha256 does not match review-plan contents")

    @property
    def executable_case_count(self) -> int:
        return sum(case.executable_quota for case in self.cases)

    def leaf_execution_quota(self) -> dict[str, int]:
        return {
            leaf_id: sum(
                case.executable_quota
                for case in self.cases
                if case.corpus_leaf_id == leaf_id
            )
            for leaf_id in sorted(EXPECTED_CORPUS_LEAVES)
        }

    def to_dict(
        self,
        *,
        corpus_registry: CorpusRegistry | None = None,
        backend_registry: BackendCapabilityRegistry | None = None,
        config: ReviewSuiteConfig | None = None,
        robocasa_catalog: RoboCasaCatalogPolicy | None = None,
    ) -> dict[str, Any]:
        self.validate(
            corpus_registry=corpus_registry,
            backend_registry=backend_registry,
            config=config,
            robocasa_catalog=robocasa_catalog,
        )
        return {**self._identity_dict(include_suite_id=True), "plan_sha256": self.plan_sha256}

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
        *,
        corpus_registry: CorpusRegistry | None = None,
        backend_registry: BackendCapabilityRegistry | None = None,
        config: ReviewSuiteConfig | None = None,
        robocasa_catalog: RoboCasaCatalogPolicy | None = None,
    ) -> "ReviewSuitePlan":
        cases = tuple(ReviewSuiteCase.from_dict(item) for item in value.get("cases", ()))
        if int(value.get("leaf_count", -1)) != len({case.corpus_leaf_id for case in cases}):
            raise ReviewSuiteValidationError("review-plan leaf_count is inconsistent")
        if int(value.get("cases_per_leaf", -1)) != 6:
            raise ReviewSuiteValidationError("review-plan cases_per_leaf is inconsistent")
        if int(value.get("expected_case_count", -1)) != len(cases):
            raise ReviewSuiteValidationError("review-plan expected_case_count is inconsistent")
        if int(value.get("expected_video_count", -1)) != len(cases) * 2:
            raise ReviewSuiteValidationError("review-plan expected_video_count is inconsistent")
        if int(value.get("executable_case_count", -1)) != sum(
            case.executable_quota for case in cases
        ):
            raise ReviewSuiteValidationError("review-plan executable count is inconsistent")
        result = cls(
            suite_id=str(value.get("suite_id", "")),
            plan_sha256=str(value.get("plan_sha256", "")),
            corpus_registry_id=str(value.get("corpus_registry_id", "")),
            corpus_registry_sha256=str(value.get("corpus_registry_sha256", "")),
            backend_registry_id=str(value.get("backend_registry_id", "")),
            backend_registry_sha256=str(value.get("backend_registry_sha256", "")),
            review_config_sha256=str(value.get("review_config_sha256", "")),
            robocasa_catalog_version=str(value.get("robocasa_catalog_version", "")),
            robocasa_catalog_sha256=str(value.get("robocasa_catalog_sha256", "")),
            robocasa_license_sha256=str(value.get("robocasa_license_sha256", "")),
            robocasa_catalog_release_ready=bool(
                value.get("robocasa_catalog_release_ready")
            ),
            fixed_master_seed=int(value.get("fixed_master_seed", -1)),
            cases=cases,
            schema_version=str(value.get("schema_version", "")),
        )
        result.validate(
            corpus_registry=corpus_registry,
            backend_registry=backend_registry,
            config=config,
            robocasa_catalog=robocasa_catalog,
        )
        return result


def build_review_suite_plan(
    *,
    corpus_registry: CorpusRegistry | None = None,
    backend_registry: BackendCapabilityRegistry | None = None,
    config: ReviewSuiteConfig | None = None,
    robocasa_catalog: RoboCasaCatalogPolicy | None = None,
) -> ReviewSuitePlan:
    """Build the complete fixed matrix without executing or activating it."""

    corpus = corpus_registry or load_corpus_registry()
    backends = backend_registry or load_backend_capability_registry(corpus=corpus)
    policy = config or load_review_suite_config()
    catalog = robocasa_catalog or load_review_robocasa_catalog()
    policy.validate()
    cases: list[ReviewSuiteCase] = []
    for leaf in corpus.leaves:
        for rollout_index in range(policy.rollouts_per_leaf):
            embodiment, task_variant, branch_role, passive_profile = _assignment(
                leaf, rollout_index
            )
            backend = backends.resolve(leaf.backend, leaf.corpus_id, embodiment)
            support = backend.support_by_leaf[leaf.corpus_id]
            eligible, hard_blockers = _case_execution_gate(
                backend,
                support,
                task_variant=task_variant,
                requires_real_robocasa=rollout_index > 0,
                robocasa_catalog=catalog,
            )
            case_id = f"{leaf.corpus_id}-review-{rollout_index:02d}"
            cases.append(
                ReviewSuiteCase(
                    case_id=case_id,
                    episode_uuid=str(
                        uuid.uuid5(
                            _REVIEW_UUID_NAMESPACE,
                            f"{corpus.registry_id}/{FIXED_REVIEW_MASTER_SEED}/{case_id}",
                        )
                    ),
                    episode_index=len(cases),
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
                    rollout_index=rollout_index,
                    scene_profile=policy.scene_sequence[rollout_index],
                    randomization_level="R0" if rollout_index == 0 else "R1",
                    requires_real_robocasa=rollout_index > 0,
                    embodiment=embodiment,
                    task_variant=task_variant,
                    branch_role=branch_role,
                    intended_outcome=_intended_outcome(branch_role),
                    passive_variation_profile=passive_profile,
                    rng_subseeds=_rng_subseeds(leaf.corpus_id, rollout_index),
                    counterfactual_bundle_id=f"review-{leaf.corpus_id}-{rollout_index:02d}",
                    counterfactual_branch_id=branch_role,
                    counterfactual_sibling_index=0,
                    execution_eligible=eligible,
                    executable_quota=int(eligible),
                    execution_blockers=hard_blockers,
                )
            )
    provisional = ReviewSuitePlan(
        suite_id="",
        plan_sha256="",
        corpus_registry_id=corpus.registry_id,
        corpus_registry_sha256=_registry_sha256(corpus),
        backend_registry_id=backends.registry_id,
        backend_registry_sha256=_registry_sha256(backends),
        review_config_sha256=policy.config_sha256,
        robocasa_catalog_version=catalog.catalog_version,
        robocasa_catalog_sha256=catalog.catalog_sha256,
        robocasa_license_sha256=catalog.license_notice_sha256,
        robocasa_catalog_release_ready=catalog.release_ready,
        fixed_master_seed=FIXED_REVIEW_MASTER_SEED,
        cases=tuple(cases),
    )
    suite_id = "review-suite-" + sha256_json(
        provisional._identity_dict(include_suite_id=False)
    )[:24]
    with_id = replace(provisional, suite_id=suite_id)
    result = replace(
        with_id,
        plan_sha256=sha256_json(with_id._identity_dict(include_suite_id=True)),
    )
    result.validate(
        corpus_registry=corpus,
        backend_registry=backends,
        config=policy,
        robocasa_catalog=catalog,
    )
    return result


@dataclass(frozen=True, slots=True)
class ReviewArtifactRequest:
    """Pending request for real, content-bound review artifacts."""

    review_plan_sha256: str
    case_id: str
    case_sha256: str
    episode_uuid: str
    corpus_leaf_id: str
    rollout_index: int
    execution_eligible: bool
    source_scenario_identity: Mapping[str, Any]
    event_strip_request: Mapping[str, Any]
    required_artifact_hash_fields: tuple[str, ...]
    human_review_checks: tuple[str, ...]
    final_artifact_schema: str = REVIEW_ARTIFACT_SCHEMA
    final_human_ledger_schema: str = REVIEW_LEDGER_SCHEMA
    status: str = "pending_artifacts"
    schema_version: str = REVIEW_REQUEST_SCHEMA

    def _identity_dict(self) -> dict[str, Any]:
        return _canonical_copy(asdict(self))

    @property
    def request_sha256(self) -> str:
        return sha256_json(self._identity_dict())

    def validate(self, plan: ReviewSuitePlan) -> None:
        if self.schema_version != REVIEW_REQUEST_SCHEMA or self.status != "pending_artifacts":
            raise ReviewSuiteValidationError("review requests must remain pending artifacts")
        if self.review_plan_sha256 != plan.plan_sha256:
            raise ReviewSuiteValidationError("review request is bound to another plan")
        selected = [case for case in plan.cases if case.case_id == self.case_id]
        if len(selected) != 1:
            raise ReviewSuiteValidationError(f"review request has unknown case {self.case_id}")
        case = selected[0]
        if (
            self.case_sha256 != case.case_sha256
            or self.episode_uuid != case.episode_uuid
            or self.corpus_leaf_id != case.corpus_leaf_id
            or self.rollout_index != case.rollout_index
            or self.execution_eligible != case.execution_eligible
        ):
            raise ReviewSuiteValidationError(f"review request identity drifted for {self.case_id}")
        expected_source_identity = {
            "schema_version": SOURCE_SCENARIO_SCHEMA_VERSION,
            "scenario_id": case.case_id,
            "corpus_leaf_id": case.corpus_leaf_id,
            "backend": case.backend,
            "embodiment": _embodiment_scenario_identity(case.embodiment),
            "task_variant": case.task_variant,
            "scene_profile": case.scene_profile,
            "randomization_level": case.randomization_level,
            "rng_subseeds": _canonical_copy(asdict(case.rng_subseeds)),
            "counterfactual_bundle_id": case.counterfactual_bundle_id,
            "counterfactual_branch_id": case.counterfactual_branch_id,
            "counterfactual_sibling_index": case.counterfactual_sibling_index,
            "robocasa_catalog_sha256": plan.robocasa_catalog_sha256,
            "robocasa_license_sha256": plan.robocasa_license_sha256,
            "require_pinned_source_hashes": True,
            "require_admitted_robocasa_manifest": case.requires_real_robocasa,
            "preserve_runtime_outcome_mismatch": True,
            "replacement_seed_forbidden": True,
        }
        if dict(self.source_scenario_identity) != expected_source_identity:
            raise ReviewSuiteValidationError(
                f"source-scenario request drifted for {self.case_id}"
            )
        expected_strip = {
            "views": ["main", "secondary"],
            "event_time_source": f"evaluator:{case.evaluator}",
            "targets": [
                {"name": "pre_event", "offset_s": -0.1},
                {"name": "event", "offset_s": 0.0},
                {"name": "post_0p1_s", "offset_s": 0.1},
                {"name": "post_0p3_s", "offset_s": 0.3},
                {"name": "final", "frame": "final"},
            ],
            "output_paths": {
                view: f"reviews/event_strips/{case.corpus_leaf_id}/{case.case_id}/{view}.png"
                for view in ("main", "secondary")
            },
            "select_from_persisted_timestamps": True,
            "require_synchronized_views": True,
        }
        if dict(self.event_strip_request) != expected_strip:
            raise ReviewSuiteValidationError(f"event-strip request drifted for {self.case_id}")
        if self.required_artifact_hash_fields != _REQUIRED_ARTIFACT_HASH_FIELDS:
            raise ReviewSuiteValidationError("review request is missing artifact hash fields")
        if self.human_review_checks != REVIEW_CHECKS:
            raise ReviewSuiteValidationError("review request changed the human checklist")
        if (
            self.final_artifact_schema != REVIEW_ARTIFACT_SCHEMA
            or self.final_human_ledger_schema != REVIEW_LEDGER_SCHEMA
        ):
            raise ReviewSuiteValidationError("review request targets unsupported final schemas")

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_dict(), "request_sha256": self.request_sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ReviewArtifactRequest":
        result = cls(
            review_plan_sha256=str(value.get("review_plan_sha256", "")),
            case_id=str(value.get("case_id", "")),
            case_sha256=str(value.get("case_sha256", "")),
            episode_uuid=str(value.get("episode_uuid", "")),
            corpus_leaf_id=str(value.get("corpus_leaf_id", "")),
            rollout_index=int(value.get("rollout_index", -1)),
            execution_eligible=bool(value.get("execution_eligible")),
            source_scenario_identity=dict(value.get("source_scenario_identity") or {}),
            event_strip_request=dict(value.get("event_strip_request") or {}),
            required_artifact_hash_fields=tuple(
                str(item) for item in value.get("required_artifact_hash_fields", ())
            ),
            human_review_checks=tuple(
                str(item) for item in value.get("human_review_checks", ())
            ),
            final_artifact_schema=str(value.get("final_artifact_schema", "")),
            final_human_ledger_schema=str(value.get("final_human_ledger_schema", "")),
            status=str(value.get("status", "")),
            schema_version=str(value.get("schema_version", "")),
        )
        if value.get("request_sha256") != result.request_sha256:
            raise ReviewSuiteValidationError(
                f"review request hash mismatch for {result.case_id or '<unknown>'}"
            )
        return result


@dataclass(frozen=True, slots=True)
class ReviewRequestLedger:
    review_plan_sha256: str
    requests: tuple[ReviewArtifactRequest, ...]
    schema_version: str = REVIEW_REQUEST_LEDGER_SCHEMA

    def _identity_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "review_plan_sha256": self.review_plan_sha256,
            "expected_request_count": len(self.requests),
            "expected_video_count": len(self.requests) * 2,
            "expected_event_strip_count": len(self.requests) * 2,
            "requests": [request.to_dict() for request in self.requests],
        }

    @property
    def ledger_sha256(self) -> str:
        return sha256_json(self._identity_dict())

    def validate(self, plan: ReviewSuitePlan) -> None:
        if self.schema_version != REVIEW_REQUEST_LEDGER_SCHEMA:
            raise ReviewSuiteValidationError(
                f"review request ledger must use {REVIEW_REQUEST_LEDGER_SCHEMA}"
            )
        if self.review_plan_sha256 != plan.plan_sha256:
            raise ReviewSuiteValidationError("review request ledger is bound to another plan")
        if len(self.requests) != 120:
            raise ReviewSuiteValidationError("review request ledger must contain 120 requests")
        if [request.case_id for request in self.requests] != [case.case_id for case in plan.cases]:
            raise ReviewSuiteValidationError("review requests do not exactly match plan order")
        for request in self.requests:
            request.validate(plan)

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_dict(), "ledger_sha256": self.ledger_sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], plan: ReviewSuitePlan) -> "ReviewRequestLedger":
        requests = tuple(
            ReviewArtifactRequest.from_dict(item) for item in value.get("requests", ())
        )
        if int(value.get("expected_request_count", -1)) != len(requests):
            raise ReviewSuiteValidationError("review request count is inconsistent")
        if int(value.get("expected_video_count", -1)) != len(requests) * 2:
            raise ReviewSuiteValidationError("review video request count is inconsistent")
        if int(value.get("expected_event_strip_count", -1)) != len(requests) * 2:
            raise ReviewSuiteValidationError("event-strip request count is inconsistent")
        result = cls(
            review_plan_sha256=str(value.get("review_plan_sha256", "")),
            requests=requests,
            schema_version=str(value.get("schema_version", "")),
        )
        if value.get("ledger_sha256") != result.ledger_sha256:
            raise ReviewSuiteValidationError("review request ledger hash mismatch")
        result.validate(plan)
        return result


def build_review_request_ledger(
    plan: ReviewSuitePlan,
    *,
    corpus_registry: CorpusRegistry | None = None,
    backend_registry: BackendCapabilityRegistry | None = None,
    config: ReviewSuiteConfig | None = None,
    robocasa_catalog: RoboCasaCatalogPolicy | None = None,
) -> ReviewRequestLedger:
    """Create pending requests without claiming that any artifact exists."""

    plan.validate(
        corpus_registry=corpus_registry,
        backend_registry=backend_registry,
        config=config,
        robocasa_catalog=robocasa_catalog,
    )
    requests: list[ReviewArtifactRequest] = []
    for case in plan.cases:
        requests.append(
            ReviewArtifactRequest(
                review_plan_sha256=plan.plan_sha256,
                case_id=case.case_id,
                case_sha256=case.case_sha256,
                episode_uuid=case.episode_uuid,
                corpus_leaf_id=case.corpus_leaf_id,
                rollout_index=case.rollout_index,
                execution_eligible=case.execution_eligible,
                source_scenario_identity={
                    "schema_version": SOURCE_SCENARIO_SCHEMA_VERSION,
                    "scenario_id": case.case_id,
                    "corpus_leaf_id": case.corpus_leaf_id,
                    "backend": case.backend,
                    "embodiment": _embodiment_scenario_identity(case.embodiment),
                    "task_variant": case.task_variant,
                    "scene_profile": case.scene_profile,
                    "randomization_level": case.randomization_level,
                    "rng_subseeds": _canonical_copy(asdict(case.rng_subseeds)),
                    "counterfactual_bundle_id": case.counterfactual_bundle_id,
                    "counterfactual_branch_id": case.counterfactual_branch_id,
                    "counterfactual_sibling_index": case.counterfactual_sibling_index,
                    "robocasa_catalog_sha256": plan.robocasa_catalog_sha256,
                    "robocasa_license_sha256": plan.robocasa_license_sha256,
                    "require_pinned_source_hashes": True,
                    "require_admitted_robocasa_manifest": case.requires_real_robocasa,
                    "preserve_runtime_outcome_mismatch": True,
                    "replacement_seed_forbidden": True,
                },
                event_strip_request={
                    "views": ["main", "secondary"],
                    "event_time_source": f"evaluator:{case.evaluator}",
                    "targets": [
                        {"name": "pre_event", "offset_s": -0.1},
                        {"name": "event", "offset_s": 0.0},
                        {"name": "post_0p1_s", "offset_s": 0.1},
                        {"name": "post_0p3_s", "offset_s": 0.3},
                        {"name": "final", "frame": "final"},
                    ],
                    "output_paths": {
                        view: (
                            f"reviews/event_strips/{case.corpus_leaf_id}/"
                            f"{case.case_id}/{view}.png"
                        )
                        for view in ("main", "secondary")
                    },
                    "select_from_persisted_timestamps": True,
                    "require_synchronized_views": True,
                },
                required_artifact_hash_fields=_REQUIRED_ARTIFACT_HASH_FIELDS,
                human_review_checks=REVIEW_CHECKS,
            )
        )
    result = ReviewRequestLedger(
        review_plan_sha256=plan.plan_sha256,
        requests=tuple(requests),
    )
    result.validate(plan)
    return result


def _write_or_validate_json(path: Path, value: Mapping[str, Any]) -> None:
    normalized = _canonical_copy(value)
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ExistingOutputError(f"immutable review artifact is not JSON: {path}") from error
        if existing != normalized:
            raise ExistingOutputError(f"immutable review artifact differs: {path}")
        return
    atomic_write_json(path, normalized)


@dataclass(frozen=True, slots=True)
class ReviewSuiteBundle:
    root: Path
    plan: ReviewSuitePlan
    requests: ReviewRequestLedger
    file_sha256: Mapping[str, str] = field(default_factory=dict)


def write_review_suite_bundle(
    output_root: str | Path,
    *,
    corpus_registry: CorpusRegistry | None = None,
    backend_registry: BackendCapabilityRegistry | None = None,
    config: ReviewSuiteConfig | None = None,
    robocasa_catalog: RoboCasaCatalogPolicy | None = None,
) -> ReviewSuiteBundle:
    """Write or byte/content-validate the three immutable suite artifacts."""

    root = ensure_not_source_path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    corpus = corpus_registry or load_corpus_registry()
    backends = backend_registry or load_backend_capability_registry(corpus=corpus)
    policy = config or load_review_suite_config()
    catalog = robocasa_catalog or load_review_robocasa_catalog()
    plan = build_review_suite_plan(
        corpus_registry=corpus,
        backend_registry=backends,
        config=policy,
        robocasa_catalog=catalog,
    )
    requests = build_review_request_ledger(
        plan,
        corpus_registry=corpus,
        backend_registry=backends,
        config=policy,
        robocasa_catalog=catalog,
    )
    plan_path = root / REVIEW_PLAN_FILE
    requests_path = root / REVIEW_REQUEST_LEDGER_FILE
    _write_or_validate_json(
        plan_path,
        plan.to_dict(
            corpus_registry=corpus,
            backend_registry=backends,
            config=policy,
            robocasa_catalog=catalog,
        ),
    )
    _write_or_validate_json(requests_path, requests.to_dict())
    file_hashes = {
        REVIEW_PLAN_FILE: sha256_file(plan_path),
        REVIEW_REQUEST_LEDGER_FILE: sha256_file(requests_path),
    }
    manifest = {
        "schema_version": REVIEW_SUITE_BUNDLE_SCHEMA,
        "suite_id": plan.suite_id,
        "plan_sha256": plan.plan_sha256,
        "request_ledger_sha256": requests.ledger_sha256,
        "files": file_hashes,
    }
    _write_or_validate_json(root / REVIEW_BUNDLE_FILE, manifest)
    return ReviewSuiteBundle(root=root, plan=plan, requests=requests, file_sha256=file_hashes)


def load_review_suite_bundle(
    output_root: str | Path,
    *,
    corpus_registry: CorpusRegistry | None = None,
    backend_registry: BackendCapabilityRegistry | None = None,
    config: ReviewSuiteConfig | None = None,
    robocasa_catalog: RoboCasaCatalogPolicy | None = None,
) -> ReviewSuiteBundle:
    """Load a suite and validate internal, registry, and on-disk hash bindings."""

    root = Path(output_root).resolve(strict=True)
    plan_path = root / REVIEW_PLAN_FILE
    request_path = root / REVIEW_REQUEST_LEDGER_FILE
    manifest_path = root / REVIEW_BUNDLE_FILE
    try:
        plan_raw = json.loads(plan_path.read_text(encoding="utf-8"))
        request_raw = json.loads(request_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ReviewSuiteValidationError(
            f"review bundle is incomplete: {error.filename}"
        ) from error
    except json.JSONDecodeError as error:
        raise ReviewSuiteValidationError("review bundle contains malformed JSON") from error
    plan = ReviewSuitePlan.from_dict(
        plan_raw,
        corpus_registry=corpus_registry,
        backend_registry=backend_registry,
        config=config,
        robocasa_catalog=robocasa_catalog,
    )
    requests = ReviewRequestLedger.from_dict(request_raw, plan)
    if manifest.get("schema_version") != REVIEW_SUITE_BUNDLE_SCHEMA:
        raise ReviewSuiteValidationError("review bundle manifest has an unsupported schema")
    if (
        manifest.get("suite_id") != plan.suite_id
        or manifest.get("plan_sha256") != plan.plan_sha256
        or manifest.get("request_ledger_sha256") != requests.ledger_sha256
    ):
        raise ReviewSuiteValidationError("review bundle manifest identity mismatch")
    expected_hashes = {
        REVIEW_PLAN_FILE: sha256_file(plan_path),
        REVIEW_REQUEST_LEDGER_FILE: sha256_file(request_path),
    }
    if manifest.get("files") != expected_hashes:
        raise ReviewSuiteValidationError("review bundle file hash mismatch")
    return ReviewSuiteBundle(
        root=root,
        plan=plan,
        requests=requests,
        file_sha256=expected_hashes,
    )


__all__ = [
    "FIXED_REVIEW_MASTER_SEED",
    "REVIEW_BUNDLE_FILE",
    "REVIEW_CONFIG_SCHEMA",
    "REVIEW_PLAN_FILE",
    "REVIEW_REQUEST_LEDGER_FILE",
    "REVIEW_REQUEST_LEDGER_SCHEMA",
    "REVIEW_REQUEST_SCHEMA",
    "REVIEW_SUITE_BUNDLE_SCHEMA",
    "REVIEW_SUITE_PLAN_SCHEMA",
    "ReviewArtifactRequest",
    "ReviewRequestLedger",
    "ReviewSuiteBundle",
    "ReviewSuiteCase",
    "ReviewSuiteConfig",
    "ReviewSuitePlan",
    "ReviewSuiteValidationError",
    "build_review_request_ledger",
    "build_review_suite_plan",
    "load_review_suite_bundle",
    "load_review_suite_config",
    "load_review_robocasa_catalog",
    "write_review_suite_bundle",
]
