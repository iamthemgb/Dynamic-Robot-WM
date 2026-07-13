"""Version-2 counterfactual, objective, frame-semantic, and assistance checks.

The helpers in this module are deliberately pure. Generators can run them
before committing an episode, and dataset QC can rerun them from persisted
Parquet rows without importing a simulator.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

from .hashing import sha256_json
from .schema import (
    ActualOutcomeClass,
    EpisodeRecord,
    SchemaValidationError,
    validate_outcome_contract,
)

COUNTERFACTUAL_TABLE_VERSION = "dynamic-robot-counterfactual-families/v1"
OBJECTIVE_EVIDENCE_VERSION = "dynamic-robot-objective-evidence/v1"


class CounterfactualRelation(str, Enum):
    """The only two supported counterfactual sibling relationships."""

    ACTION = "action"
    PHYSICS = "physics"


class MotionMode(str, Enum):
    """Closed per-frame motion-regime labels."""

    STATIONARY = "stationary"
    FREE_FLIGHT = "free_flight"
    ROLLING = "rolling"
    SLIDING = "sliding"
    IMPACT = "impact"
    RETAINED = "retained"
    ROBOT_MANIPULATED = "robot_manipulated"
    DEFORMING = "deforming"
    SETTLING = "settling"
    UNKNOWN = "unknown"


class ContactRole(str, Enum):
    """Closed role of the active contact at a saved frame."""

    NONE = "none"
    SUPPORT = "support"
    FIXTURE = "fixture"
    ROBOT_TOOL = "robot_tool"
    TARGET = "target"
    ASSISTANCE = "assistance"
    SELF_CONTACT = "self_contact"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class CounterfactualFamilyRecord:
    """One declared family row written before simulation begins.

    Expected membership, rather than observed membership, is authoritative so
    a crashed or selectively missing sibling is detectable.
    """

    family_id: str
    relation: CounterfactualRelation
    split_group_id: str
    expected_member_count: int
    expected_episode_uuids: list[str]
    intervention_fields: list[str]
    fixed_field_hashes: dict[str, str] = field(default_factory=dict)
    expected_member_plan_hashes: dict[str, str] = field(default_factory=dict)
    table_version: str = COUNTERFACTUAL_TABLE_VERSION

    def validate(self) -> None:
        if self.table_version != COUNTERFACTUAL_TABLE_VERSION:
            raise SchemaValidationError(
                f"Unsupported counterfactual table version: {self.table_version}"
            )
        if not self.family_id or not self.split_group_id:
            raise SchemaValidationError("Counterfactual family/split IDs are required")
        if not isinstance(self.relation, CounterfactualRelation):
            self.relation = CounterfactualRelation(self.relation)
        if self.expected_member_count < 2:
            raise SchemaValidationError("Counterfactual families require at least two members")
        if len(self.expected_episode_uuids) != self.expected_member_count:
            raise SchemaValidationError(
                "expected_episode_uuids length differs from expected_member_count"
            )
        if len(set(self.expected_episode_uuids)) != len(self.expected_episode_uuids):
            raise SchemaValidationError("Counterfactual expected UUIDs must be unique")
        if not self.intervention_fields or len(self.intervention_fields) != len(
            set(self.intervention_fields)
        ):
            raise SchemaValidationError(
                "Counterfactual intervention_fields must be non-empty and unique"
            )
        for name, digest in self.fixed_field_hashes.items():
            if not name or len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise SchemaValidationError(
                    f"Invalid fixed-field SHA-256 for {name or '<unnamed>'}"
                )
        if self.expected_member_plan_hashes and set(
            self.expected_member_plan_hashes
        ) != set(self.expected_episode_uuids):
            raise SchemaValidationError(
                "expected_member_plan_hashes must bind every expected UUID exactly"
            )
        for episode_uuid, digest in self.expected_member_plan_hashes.items():
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise SchemaValidationError(
                    f"Invalid expected plan SHA-256 for {episode_uuid}"
                )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {**asdict(self), "relation": self.relation.value}

    def to_table_row(self) -> dict[str, Any]:
        value = self.to_dict()
        for name in (
            "expected_episode_uuids",
            "intervention_fields",
            "fixed_field_hashes",
            "expected_member_plan_hashes",
        ):
            value[f"{name}_json"] = json.dumps(
                value.pop(name), sort_keys=True, separators=(",", ":"), allow_nan=False
            )
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CounterfactualFamilyRecord":
        data = dict(value)
        for name in (
            "expected_episode_uuids",
            "intervention_fields",
            "fixed_field_hashes",
            "expected_member_plan_hashes",
        ):
            encoded = data.pop(f"{name}_json", None)
            if encoded is not None:
                data[name] = json.loads(str(encoded))
        data["relation"] = CounterfactualRelation(data["relation"])
        result = cls(**data)
        result.validate()
        return result


def _physics_payload(record: EpisodeRecord, intervention_fields: Sequence[str] = ()) -> dict[str, Any]:
    excluded = set(intervention_fields)
    value = record.physics.to_dict()
    value["parameters"] = {
        name: parameter
        for name, parameter in value["parameters"].items()
        if name not in excluded
    }
    # These are normalized convenience mirrors of named parameters, not
    # independent fields. Excluding a declared intervention must also exclude
    # its mirror or a valid physics family would appear to change an invariant.
    if "gravity" in excluded:
        value.pop("gravity_world_m_s2", None)
        value.pop("gravity_valid", None)
    if "simulation_timestep" in excluded:
        value.pop("simulation_timestep_s", None)
    if "substeps" in excluded:
        value.pop("substeps", None)
    return value


def _appearance_payload(record: EpisodeRecord) -> dict[str, Any]:
    return {
        "asset_ids": record.asset_ids,
        "asset_hashes": record.asset_hashes,
        "randomization": record.randomization,
        "camera_stream_calibration_ids": record.camera_stream_calibration_ids,
    }


_FRAME_IDENTITY_FIELDS = {
    "episode_index",
    "frame_index",
    "video_frame_index",
    "task_index",
    "timestamp",
}
_TRANSIENT_SEMANTIC_FIELDS = {
    "task_phase",
    "motion_mode",
    "active_surface",
    "contact_role",
    "free_fall",
    "object.motion_mode",
    "object.active_surface",
}
_TRANSIENT_SEMANTIC_PREFIXES = (
    "action.",
    "contact.",
    "event.",
    "assistance.",
    "task.",
)


def derived_action_hash_from_rows(rows: Sequence[Mapping[str, Any]]) -> str:
    """Hash the complete timestamped action replay from persisted frame rows."""

    action_fields = [
        "timestamp",
        *sorted(
            name
            for name in (rows[0] if rows else {})
            if name.startswith("action.")
        ),
    ]
    return sha256_json([[row.get(name) for name in action_fields] for row in rows])


def derived_initial_state_hash_from_rows(rows: Sequence[Mapping[str, Any]]) -> str:
    """Hash physical state at frame zero, excluding branch semantics.

    Phase, motion/contact classifications, and action/event masks describe how
    a rollout is interpreted or controlled. They are not part of the physical
    initial condition and may legitimately differ across action siblings,
    especially a first-class no-op branch.
    """

    initial_fields = sorted(
        name
        for name in (rows[0] if rows else {})
        if name not in _FRAME_IDENTITY_FIELDS
        and name not in _TRANSIENT_SEMANTIC_FIELDS
        and not name.startswith(_TRANSIENT_SEMANTIC_PREFIXES)
    )
    return sha256_json(
        {name: rows[0].get(name) for name in initial_fields} if rows else {}
    )


def counterfactual_fixed_hashes(
    record: EpisodeRecord,
    relation: CounterfactualRelation,
    *,
    intervention_fields: Sequence[str] = (),
    derived_initial_state_hash: str | None = None,
    derived_action_hash: str | None = None,
) -> dict[str, str]:
    """Compute invariant hashes from canonical metadata and persisted tables."""

    planned_by_relation = record.extras.get("planned_counterfactual_fixed_hashes")
    if isinstance(planned_by_relation, Mapping):
        planned = planned_by_relation.get(relation.value)
        if isinstance(planned, Mapping):
            return {
                str(name): str(digest)
                for name, digest in planned.items()
                if isinstance(digest, str) and digest
            }

    common = {
        "scene_seed": sha256_json(record.scene_seed),
        "appearance": sha256_json(_appearance_payload(record)),
        "split_group_id": sha256_json(record.split_group_id),
    }
    initial_state_hash = derived_initial_state_hash or str(
        record.extras.get("initial_state_hash") or ""
    )
    if initial_state_hash:
        common["initial_state"] = initial_state_hash
    if relation == CounterfactualRelation.ACTION:
        common["physics"] = sha256_json(_physics_payload(record))
    else:
        common["nonintervened_physics"] = sha256_json(
            _physics_payload(record, intervention_fields)
        )
        common["action"] = derived_action_hash or str(record.extras.get("action_hash") or "")
    return common


def build_counterfactual_family_records(
    records: Iterable[EpisodeRecord],
    *,
    action_intervention_fields: Sequence[str] = ("action",),
    physics_interventions: Mapping[str, Sequence[str]] | None = None,
) -> list[CounterfactualFamilyRecord]:
    """Build complete table rows from a planned (not filtered) episode list.

    The caller must invoke this on plans before simulation so failed branches
    remain in ``expected_episode_uuids``.
    """

    values = list(records)
    groups: dict[tuple[CounterfactualRelation, str], list[EpisodeRecord]] = defaultdict(list)
    for record in values:
        groups[(CounterfactualRelation.ACTION, record.counterfactual_bundle_id)].append(record)
        if record.physics_counterfactual_family_id:
            groups[(CounterfactualRelation.PHYSICS, record.physics_counterfactual_family_id)].append(record)
    result: list[CounterfactualFamilyRecord] = []
    for (relation, family_id), siblings in sorted(
        groups.items(), key=lambda item: (item[0][0].value, item[0][1])
    ):
        if len(siblings) < 2:
            continue
        interventions = (
            list(action_intervention_fields)
            if relation == CounterfactualRelation.ACTION
            else list((physics_interventions or {}).get(family_id, ()))
        )
        if not interventions:
            # A physics family without a named intervention is unsafe to publish.
            interventions = ["__undeclared_physics_intervention__"]
        hashes_by_name: dict[str, set[str]] = defaultdict(set)
        for sibling in siblings:
            for name, digest in counterfactual_fixed_hashes(
                sibling,
                relation,
                intervention_fields=interventions,
                derived_initial_state_hash=str(
                    sibling.extras.get("derived_initial_state_hash") or ""
                )
                or None,
                derived_action_hash=str(
                    sibling.extras.get("derived_action_hash") or ""
                )
                or None,
            ).items():
                if digest:
                    hashes_by_name[name].add(digest)
        fixed = {
            name: next(iter(digests))
            for name, digests in hashes_by_name.items()
            if len(digests) == 1
        }
        declaration = CounterfactualFamilyRecord(
            family_id=family_id,
            relation=relation,
            split_group_id=siblings[0].split_group_id or "",
            expected_member_count=len(siblings),
            expected_episode_uuids=sorted(record.episode_uuid for record in siblings),
            intervention_fields=interventions,
            fixed_field_hashes=fixed,
            expected_member_plan_hashes={
                record.episode_uuid: record.config_hash for record in siblings
            },
        )
        declaration.validate()
        result.append(declaration)
    return result


def validate_counterfactual_family_records(
    declarations: Iterable[CounterfactualFamilyRecord],
    records: Iterable[EpisodeRecord],
    *,
    derived_by_uuid: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[str]:
    """Return concrete missing-member, leakage, and invariant violations."""

    episodes = list(records)
    episode_by_uuid = {record.episode_uuid: record for record in episodes}
    derived = derived_by_uuid or {}
    problems: list[str] = []
    declared_keys: set[tuple[CounterfactualRelation, str]] = set()
    for declaration in declarations:
        try:
            declaration.validate()
        except SchemaValidationError as error:
            problems.append(f"counterfactual declaration {declaration.family_id}: {error}")
            continue
        key = (declaration.relation, declaration.family_id)
        if key in declared_keys:
            problems.append(
                f"duplicate counterfactual declaration: {declaration.relation.value}/{declaration.family_id}"
            )
            continue
        declared_keys.add(key)
        required_fixed = {
            "scene_seed",
            "appearance",
            "split_group_id",
            "initial_state",
            "physics" if declaration.relation == CounterfactualRelation.ACTION else "action",
        }
        if declaration.relation == CounterfactualRelation.PHYSICS:
            required_fixed.add("nonintervened_physics")
        missing_fixed = sorted(required_fixed - set(declaration.fixed_field_hashes))
        if missing_fixed:
            problems.append(
                f"{declaration.relation.value} family {declaration.family_id} "
                f"does not declare fixed hashes for {missing_fixed}"
            )
        if set(declaration.expected_member_plan_hashes) != set(
            declaration.expected_episode_uuids
        ):
            problems.append(
                f"{declaration.relation.value} family {declaration.family_id} "
                "does not content-bind every expected member plan"
            )
        missing = sorted(set(declaration.expected_episode_uuids) - set(episode_by_uuid))
        if missing:
            problems.append(
                f"{declaration.relation.value} family {declaration.family_id} missing expected members: {missing}"
            )
        observed_members = {
            episode.episode_uuid
            for episode in episodes
            if (
                episode.counterfactual_bundle_id
                if declaration.relation == CounterfactualRelation.ACTION
                else episode.physics_counterfactual_family_id
            )
            == declaration.family_id
        }
        unexpected = sorted(
            observed_members - set(declaration.expected_episode_uuids)
        )
        if unexpected:
            problems.append(
                f"{declaration.relation.value} family {declaration.family_id} "
                f"has undeclared extra members: {unexpected}"
            )
        siblings = [
            episode_by_uuid[episode_uuid]
            for episode_uuid in declaration.expected_episode_uuids
            if episode_uuid in episode_by_uuid
        ]
        for sibling in siblings:
            actual_family_id = (
                sibling.counterfactual_bundle_id
                if declaration.relation == CounterfactualRelation.ACTION
                else sibling.physics_counterfactual_family_id
            )
            if actual_family_id != declaration.family_id:
                problems.append(
                    f"episode {sibling.episode_uuid} has wrong {declaration.relation.value} family ID"
                )
            if sibling.split_group_id != declaration.split_group_id:
                problems.append(
                    f"{declaration.relation.value} family {declaration.family_id} crosses split groups"
                )
            expected_plan_hash = declaration.expected_member_plan_hashes.get(
                sibling.episode_uuid
            )
            actual_plan_hash = str(
                sibling.extras.get("family_plan_config_hash")
                or sibling.config_hash
            )
            if expected_plan_hash != actual_plan_hash:
                problems.append(
                    f"{declaration.relation.value} family {declaration.family_id} "
                    f"has a plan-hash mismatch in {sibling.episode_uuid}"
                )
            computed = counterfactual_fixed_hashes(
                sibling,
                declaration.relation,
                intervention_fields=declaration.intervention_fields,
                derived_initial_state_hash=derived.get(sibling.episode_uuid, {}).get(
                    "derived_initial_state_hash"
                ),
                derived_action_hash=derived.get(sibling.episode_uuid, {}).get(
                    "derived_action_hash"
                ),
            )
            for name, expected in declaration.fixed_field_hashes.items():
                actual = computed.get(name)
                if actual != expected:
                    problems.append(
                        f"{declaration.relation.value} family {declaration.family_id} "
                        f"changes fixed field {name} in {sibling.episode_uuid}"
                    )
        # Even when a table omitted fixed hashes, detect varying computed invariants.
        for name in (
            "scene_seed",
            "appearance",
            "split_group_id",
            "initial_state",
            "physics" if declaration.relation == CounterfactualRelation.ACTION else "action",
            "nonintervened_physics",
        ):
            values = {
                counterfactual_fixed_hashes(
                    sibling,
                    declaration.relation,
                    intervention_fields=declaration.intervention_fields,
                    derived_initial_state_hash=derived.get(sibling.episode_uuid, {}).get(
                        "derived_initial_state_hash"
                    ),
                    derived_action_hash=derived.get(sibling.episode_uuid, {}).get(
                        "derived_action_hash"
                    ),
                ).get(name)
                for sibling in siblings
            }
            values.discard(None)
            values.discard("")
            if len(values) > 1:
                problems.append(
                    f"{declaration.relation.value} family {declaration.family_id} changes invariant {name}"
                )
        measured_initial_hashes = {
            derived.get(sibling.episode_uuid, {}).get("derived_initial_state_hash")
            or sibling.extras.get("derived_initial_state_hash")
            or sibling.extras.get("initial_state_hash")
            for sibling in siblings
        }
        measured_initial_hashes.discard(None)
        measured_initial_hashes.discard("")
        if len(measured_initial_hashes) != 1:
            problems.append(
                f"{declaration.relation.value} family {declaration.family_id} "
                "changes or lacks the saved initial physical state"
            )
        if declaration.relation == CounterfactualRelation.ACTION and len(siblings) > 1:
            action_hashes = {
                derived.get(sibling.episode_uuid, {}).get("derived_action_hash")
                or sibling.extras.get("derived_action_hash")
                or sibling.extras.get("action_hash")
                or sibling.extras.get("action_hash")
                for sibling in siblings
            }
            action_hashes.discard(None)
            action_hashes.discard("")
            if len(action_hashes) < 2:
                problems.append(
                    f"action family {declaration.family_id} does not vary the saved action trajectory"
                )
        if declaration.relation == CounterfactualRelation.PHYSICS and len(siblings) > 1:
            measured_action_hashes = {
                derived.get(sibling.episode_uuid, {}).get("derived_action_hash")
                or sibling.extras.get("derived_action_hash")
                for sibling in siblings
            }
            measured_action_hashes.discard(None)
            measured_action_hashes.discard("")
            if len(measured_action_hashes) != 1:
                problems.append(
                    f"physics family {declaration.family_id} changes or lacks the "
                    "saved timestamped action replay"
                )
            for field_name in declaration.intervention_fields:
                values = {
                    sha256_json(
                        sibling.physics.to_dict().get("parameters", {}).get(field_name)
                    )
                    for sibling in siblings
                }
                if len(values) < 2:
                    problems.append(
                        f"physics family {declaration.family_id} does not vary declared "
                        f"intervention {field_name}"
                    )
    return sorted(set(problems))


@dataclass(slots=True, frozen=True)
class ObjectiveRecomputeInput:
    """Persisted evidence available to an objective evaluator."""

    record: EpisodeRecord
    frame_rows: Sequence[Mapping[str, Any]]
    event_rows: Sequence[Mapping[str, Any]]
    object_state_rows: Sequence[Mapping[str, Any]]


@dataclass(slots=True)
class ObjectiveRecomputeResult:
    """Independent outcome derived only from committed episode artifacts."""

    task_success: bool
    actual_outcome_class: ActualOutcomeClass
    primary_failure_code: str
    evidence: dict[str, Any]
    partial_success_score: float | None = None
    key_event_name: str | None = None
    key_event_time_s: float | None = None
    evidence_version: str = OBJECTIVE_EVIDENCE_VERSION

    def validate(self) -> None:
        if not isinstance(self.actual_outcome_class, ActualOutcomeClass):
            self.actual_outcome_class = ActualOutcomeClass(self.actual_outcome_class)
        validate_outcome_contract(
            task_success=self.task_success,
            outcome_class=self.actual_outcome_class,
            primary_failure_code=self.primary_failure_code,
            compatibility_failure_mode=self.primary_failure_code,
            partial_success_score=self.partial_success_score,
        )
        if self.evidence_version != OBJECTIVE_EVIDENCE_VERSION:
            raise SchemaValidationError("Unknown objective-evidence version")
        if not self.evidence:
            raise SchemaValidationError("Objective recomputation requires concrete evidence")
        if self.key_event_time_s is not None and (
            not math.isfinite(self.key_event_time_s) or self.key_event_time_s < 0
        ):
            raise SchemaValidationError("Objective key-event time must be finite and non-negative")
        if self.key_event_time_s is not None and not str(self.key_event_name or "").strip():
            raise SchemaValidationError("Timed objective key event requires a name")


class PersistedObjectiveEvaluator(Protocol):
    """A deterministic objective evaluator that cannot inspect branch intent."""

    def __call__(self, evidence: ObjectiveRecomputeInput) -> ObjectiveRecomputeResult: ...


class ObjectiveEvaluatorRegistry:
    """Explicit evaluator registry keyed by immutable ID and version."""

    def __init__(self) -> None:
        self._evaluators: dict[tuple[str, str], PersistedObjectiveEvaluator] = {}

    def register(
        self,
        evaluator_id: str,
        evaluator_version: str,
        evaluator: PersistedObjectiveEvaluator,
    ) -> None:
        key = (evaluator_id.strip(), evaluator_version.strip())
        if not all(key):
            raise ValueError("Objective evaluator ID/version must be non-empty")
        if key in self._evaluators:
            raise ValueError(f"Objective evaluator already registered: {key}")
        self._evaluators[key] = evaluator

    def get(self, evaluator_id: str, evaluator_version: str) -> PersistedObjectiveEvaluator | None:
        key = (evaluator_id, evaluator_version)
        evaluator = self._evaluators.get(key)
        if evaluator is None and key == ("native_rigid_state_event", "1.1.0"):
            # Keep schema-only commands free of simulator imports and avoid a
            # module-initialisation cycle by loading the built-in bridge only
            # when its immutable identifier is requested.
            from ..backends.mujoco_native.registration import (
                register_common_objective_evaluator,
            )

            register_common_objective_evaluator()
            evaluator = self._evaluators.get(key)
        return evaluator

    def decorator(
        self, evaluator_id: str, evaluator_version: str
    ) -> Callable[[PersistedObjectiveEvaluator], PersistedObjectiveEvaluator]:
        def register(evaluator: PersistedObjectiveEvaluator) -> PersistedObjectiveEvaluator:
            self.register(evaluator_id, evaluator_version, evaluator)
            return evaluator

        return register


DEFAULT_OBJECTIVE_EVALUATORS = ObjectiveEvaluatorRegistry()


def compare_recomputed_objective(
    record: EpisodeRecord, result: ObjectiveRecomputeResult
) -> list[str]:
    """Compare an independent recomputation with all stored label fields."""

    result.validate()
    problems: list[str] = []
    if result.task_success != record.task_success:
        problems.append("task_success disagrees with independent objective recomputation")
    if result.actual_outcome_class != record.actual_outcome_class:
        problems.append("actual_outcome_class disagrees with independent objective recomputation")
    if result.primary_failure_code != record.primary_failure_code:
        problems.append("primary_failure_code disagrees with independent objective recomputation")
    if result.partial_success_score != record.partial_success_score:
        if result.partial_success_score is None or record.partial_success_score is None or abs(
            result.partial_success_score - record.partial_success_score
        ) > 1e-9:
            problems.append("partial_success_score disagrees with independent objective recomputation")
    if result.key_event_time_s is not None and (
        record.key_event_time_s is None
        or abs(result.key_event_time_s - record.key_event_time_s) > 1e-6
    ):
        problems.append("key_event_time_s disagrees with independent objective recomputation")
    return problems


def validate_v2_frame_semantics(
    rows: Sequence[Mapping[str, Any]],
    *,
    strict: bool = True,
) -> list[str]:
    """Validate task phase, motion regime, surface, and contact role per frame."""

    problems: list[str] = []
    if not rows:
        return ["frame-semantic validation requires at least one frame row"]
    required = ("task_phase", "motion_mode", "active_surface", "contact_role")
    missing = [name for name in required if name not in rows[0]]
    if missing:
        return [f"missing v2 frame-semantic columns: {missing}"] if strict else []
    for index, row in enumerate(rows):
        if not str(row.get("task_phase", "")).strip():
            problems.append(f"frame {index} has empty task_phase")
        try:
            MotionMode(str(row.get("motion_mode")))
        except ValueError:
            problems.append(f"frame {index} has unknown motion_mode {row.get('motion_mode')!r}")
        surface = row.get("active_surface")
        if surface is not None and not str(surface).strip():
            problems.append(f"frame {index} has invalid active_surface")
        try:
            ContactRole(str(row.get("contact_role")))
        except ValueError:
            problems.append(f"frame {index} has unknown contact_role {row.get('contact_role')!r}")
    return sorted(set(problems))


def validate_assistance_observations(
    record: EpisodeRecord,
    frame_rows: Sequence[Mapping[str, Any]],
    *,
    tolerance_s: float,
) -> list[str]:
    """Match declared mechanism intervals to simulator-observed frame masks."""

    mechanisms = list(record.assistance.get("mechanisms") or [])
    if not mechanisms:
        return []
    if tolerance_s < 0 or not math.isfinite(tolerance_s):
        raise ValueError("tolerance_s must be finite and non-negative")
    problems: list[str] = []
    for index, row in enumerate(frame_rows):
        timestamp = float(row["timestamp"])
        measured_raw = row.get("assistance.mechanism_ids")
        if measured_raw is None:
            problems.append("frame rows lack assistance.mechanism_ids")
            break
        measured = {str(value) for value in measured_raw}
        expected: set[str] = set()
        for mechanism in mechanisms:
            for interval in mechanism["activation_intervals"]:
                start = float(interval["start_time_s"])
                end = interval.get("end_time_s")
                if timestamp + tolerance_s >= start and (
                    end is None or timestamp - tolerance_s <= float(end)
                ):
                    expected.add(str(mechanism["mechanism_id"]))
                    break
        if measured != expected:
            problems.append(
                f"frame {index} assistance mechanisms {sorted(measured)} != interval-derived {sorted(expected)}"
            )
    return sorted(set(problems))
