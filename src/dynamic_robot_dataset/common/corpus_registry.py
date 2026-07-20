"""Authoritative corpus taxonomy and physics-backend capability contracts.

The YAML files loaded here are the runtime source of truth.  The registry is
deliberately fail-closed: a typo in a leaf, backend, embodiment, or task variant
is an error rather than a request that can silently fall back to another
generator.  Release readiness is separate from capability so unfinished leaves
can be planned and tested without receiving production quota.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import yaml


CORPUS_REGISTRY_SCHEMA_VERSION = "dynamic-robot-corpus-registry/v1"
BACKEND_CAPABILITY_SCHEMA_VERSION = "dynamic-robot-backend-capabilities/v2"
CANONICAL_CONTRACT_VERSION = "dynamic-robot-dataset/v2"

EXPECTED_CORPUS_LEAVES = frozenset(
    {
        "P0a",
        "P0b",
        "P0c",
        "P0d",
        "F1a",
        "F1b",
        "F1c",
        "F1d",
        "F2a",
        "F2b",
        "F2c",
        "F2d",
        "F2e",
        "F2f",
        "F3a",
        "F3b",
        "F3c",
        "F3d",
        "D1",
        "D2",
    }
)

EXPECTED_BACKEND_BY_LEAF: Mapping[str, str] = {
    **{
        leaf: "source_mujoco"
        for leaf in EXPECTED_CORPUS_LEAVES
        if leaf.startswith(("P0", "F1", "F2"))
    },
    "F3a": "source_mujoco",
    "F3b": "source_mujoco",
    "F3c": "source_genesis_fluid",
    "F3d": "source_mujoco",
    "D1": "source_mujoco_deformable",
    "D2": "source_mujoco_deformable",
}

KNOWN_EMBODIMENTS = frozenset(
    {"no_robot", "franka_hand", "robotiq_2f85_thick_pad"}
)
PERMANENTLY_BLOCKED_BACKEND = "native_mujoco"
PILOT_ACTIVATION_REPORT_SCHEMA = "dynamic-robot-leaf-activation-report/v1"

_HASH_PATTERN = re.compile(r"^(?:[0-9a-f]{64}|sha256:[0-9a-f]{64}|git:[0-9a-f]{40})$")


class RegistryValidationError(ValueError):
    """A registry file is incomplete or internally inconsistent."""


class UnsupportedScenarioError(ValueError):
    """A leaf/backend/embodiment/task combination is not explicitly admitted."""


class BackendNotReleasedError(RuntimeError):
    """A supported capability was requested for production before activation."""


class ReleaseState(str, Enum):
    BLOCKED = "blocked"
    REVIEW = "review"
    PILOT = "pilot"
    RELEASED = "released"
    PERMANENTLY_BLOCKED = "permanently_blocked"

    @property
    def allows_production(self) -> bool:
        return self is ReleaseState.RELEASED

    @property
    def allows_pilot(self) -> bool:
        return self in {ReleaseState.PILOT, ReleaseState.RELEASED}


class ExecutionState(str, Enum):
    """Per-leaf backend readiness for fixed acceptance-review execution.

    This is intentionally independent of :class:`ReleaseState`.  A support
    entry may execute the immutable review suite while its owning backend and
    corpus leaf remain blocked from pilot or production use.
    """

    BLOCKED = "blocked"
    REVIEW = "review"

    @property
    def allows_review(self) -> bool:
        return self is ExecutionState.REVIEW


@dataclass(frozen=True, slots=True)
class RateSpec:
    simulation_hz: tuple[int, ...]
    control_hz: int
    video_hz: int

    def validate(self) -> None:
        if not self.simulation_hz or any(value <= 0 for value in self.simulation_hz):
            raise RegistryValidationError("simulation_hz requires positive candidates")
        if len(set(self.simulation_hz)) != len(self.simulation_hz):
            raise RegistryValidationError("simulation_hz candidates must be unique")
        if self.control_hz <= 0 or self.video_hz <= 0:
            raise RegistryValidationError("control_hz and video_hz must be positive")
        if any(value % self.control_hz for value in self.simulation_hz):
            raise RegistryValidationError(
                "each simulation rate must be an integer multiple of control_hz"
            )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RateSpec":
        result = cls(
            simulation_hz=tuple(int(item) for item in value.get("simulation_hz", ())),
            control_hz=int(value.get("control_hz", 0)),
            video_hz=int(value.get("video_hz", 0)),
        )
        result.validate()
        return result


@dataclass(frozen=True, slots=True)
class CorpusLeaf:
    corpus_id: str
    slug: str
    title: str
    family: str
    subfamily: str
    backend: str
    supported_embodiments: tuple[str, ...]
    task_variants: tuple[str, ...]
    rates: RateSpec
    evaluator: str
    required_metadata: tuple[str, ...]
    release_state: ReleaseState
    blockers: tuple[str, ...]

    def validate(self) -> None:
        if self.corpus_id not in EXPECTED_CORPUS_LEAVES:
            raise RegistryValidationError(f"unknown corpus leaf {self.corpus_id!r}")
        expected_backend = EXPECTED_BACKEND_BY_LEAF[self.corpus_id]
        if self.backend != expected_backend:
            raise RegistryValidationError(
                f"{self.corpus_id} must use {expected_backend}, not {self.backend}"
            )
        required_text = {
            "slug": self.slug,
            "title": self.title,
            "family": self.family,
            "subfamily": self.subfamily,
            "evaluator": self.evaluator,
        }
        missing = sorted(name for name, value in required_text.items() if not value.strip())
        if missing:
            raise RegistryValidationError(
                f"{self.corpus_id} has empty required fields: {', '.join(missing)}"
            )
        if not self.supported_embodiments or len(set(self.supported_embodiments)) != len(
            self.supported_embodiments
        ):
            raise RegistryValidationError(
                f"{self.corpus_id} needs unique supported embodiments"
            )
        unknown_embodiments = sorted(set(self.supported_embodiments) - KNOWN_EMBODIMENTS)
        if unknown_embodiments:
            raise RegistryValidationError(
                f"{self.corpus_id} has unknown embodiments: {', '.join(unknown_embodiments)}"
            )
        if self.corpus_id.startswith("P0"):
            if self.supported_embodiments != ("no_robot",):
                raise RegistryValidationError(f"{self.corpus_id} is passive and must use no_robot")
        elif "no_robot" in self.supported_embodiments:
            raise RegistryValidationError(
                f"{self.corpus_id} is an actuated leaf and cannot advertise no_robot"
            )
        if not self.task_variants or len(set(self.task_variants)) != len(self.task_variants):
            raise RegistryValidationError(f"{self.corpus_id} needs unique task variants")
        if not self.required_metadata or "source_hashes" not in self.required_metadata:
            raise RegistryValidationError(
                f"{self.corpus_id} must require source_hashes metadata"
            )
        if self.release_state is ReleaseState.RELEASED and self.blockers:
            raise RegistryValidationError(
                f"released leaf {self.corpus_id} cannot retain blockers"
            )
        if self.release_state is not ReleaseState.RELEASED and not self.blockers:
            raise RegistryValidationError(
                f"unreleased leaf {self.corpus_id} must declare at least one blocker"
            )
        if self.release_state is ReleaseState.PERMANENTLY_BLOCKED:
            raise RegistryValidationError(
                "corpus leaves may be repaired; permanent blocking is backend-only"
            )
        self.rates.validate()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CorpusLeaf":
        result = cls(
            corpus_id=str(value.get("id", "")),
            slug=str(value.get("slug", "")),
            title=str(value.get("title", "")),
            family=str(value.get("family", "")),
            subfamily=str(value.get("subfamily", "")),
            backend=str(value.get("backend", "")),
            supported_embodiments=tuple(
                str(item) for item in value.get("supported_embodiments", ())
            ),
            task_variants=tuple(str(item) for item in value.get("task_variants", ())),
            rates=RateSpec.from_dict(_mapping(value.get("rates"), "leaf rates")),
            evaluator=str(value.get("evaluator", "")),
            required_metadata=tuple(
                str(item) for item in value.get("required_metadata", ())
            ),
            release_state=_release_state(value.get("release_state")),
            blockers=tuple(str(item) for item in value.get("blockers", ())),
        )
        result.validate()
        return result


@dataclass(frozen=True, slots=True)
class CorpusRegistry:
    registry_id: str
    contract_version: str
    leaves: tuple[CorpusLeaf, ...]
    canonical_width: int
    canonical_height: int
    canonical_video_hz: int
    canonical_cameras: tuple[str, ...]
    fixed_review_rollouts: int

    @property
    def by_id(self) -> dict[str, CorpusLeaf]:
        return {leaf.corpus_id: leaf for leaf in self.leaves}

    def validate(self) -> None:
        if not self.registry_id:
            raise RegistryValidationError("corpus registry_id is required")
        if self.contract_version != CANONICAL_CONTRACT_VERSION:
            raise RegistryValidationError(
                f"corpus contract must be {CANONICAL_CONTRACT_VERSION}"
            )
        ids = [leaf.corpus_id for leaf in self.leaves]
        if len(ids) != len(set(ids)):
            raise RegistryValidationError("corpus leaf IDs must be unique")
        if set(ids) != EXPECTED_CORPUS_LEAVES:
            missing = sorted(EXPECTED_CORPUS_LEAVES - set(ids))
            extra = sorted(set(ids) - EXPECTED_CORPUS_LEAVES)
            raise RegistryValidationError(
                f"corpus registry must contain exactly 20 leaves; missing={missing}, extra={extra}"
            )
        slugs = [leaf.slug for leaf in self.leaves]
        if len(slugs) != len(set(slugs)):
            raise RegistryValidationError("corpus leaf slugs must be unique")
        if (self.canonical_width, self.canonical_height, self.canonical_video_hz) != (
            832,
            480,
            30,
        ):
            raise RegistryValidationError("canonical media must remain 832x480 at 30 Hz")
        if self.canonical_cameras != ("main", "secondary"):
            raise RegistryValidationError("canonical episodes require main and secondary cameras")
        if self.fixed_review_rollouts != 6:
            raise RegistryValidationError("each corpus leaf requires exactly six fixed reviews")
        for leaf in self.leaves:
            leaf.validate()
            if leaf.rates.video_hz != self.canonical_video_hz:
                raise RegistryValidationError(
                    f"{leaf.corpus_id} video rate differs from canonical media"
                )

    def resolve(
        self,
        corpus_id: str,
        *,
        embodiment: str | None = None,
        task_variant: str | None = None,
        require_released: bool = False,
        purpose: str | None = None,
    ) -> CorpusLeaf:
        try:
            leaf = self.by_id[str(corpus_id)]
        except KeyError as error:
            raise UnsupportedScenarioError(f"unknown corpus leaf {corpus_id!r}") from error
        if embodiment is not None and embodiment not in leaf.supported_embodiments:
            raise UnsupportedScenarioError(
                f"{leaf.corpus_id} does not support embodiment {embodiment!r}; "
                f"allowed={list(leaf.supported_embodiments)}"
            )
        if task_variant is not None and task_variant not in leaf.task_variants:
            raise UnsupportedScenarioError(
                f"{leaf.corpus_id} does not support task variant {task_variant!r}; "
                f"allowed={list(leaf.task_variants)}"
            )
        normalized_purpose = _generation_purpose(purpose)
        if require_released:
            if normalized_purpose not in {None, "production"}:
                raise ValueError(
                    "require_released cannot be combined with a non-production purpose"
                )
            normalized_purpose = "production"
        if normalized_purpose == "pilot" and not leaf.release_state.allows_pilot:
            raise BackendNotReleasedError(
                f"corpus leaf {leaf.corpus_id} is {leaf.release_state.value} and "
                f"cannot run a pilot: {', '.join(leaf.blockers)}"
            )
        if normalized_purpose == "production" and not leaf.release_state.allows_production:
            raise BackendNotReleasedError(
                f"corpus leaf {leaf.corpus_id} is {leaf.release_state.value}: "
                f"{', '.join(leaf.blockers)}"
            )
        return leaf


@dataclass(frozen=True, slots=True)
class BackendSupport:
    corpus_id: str
    family: str
    subfamily: str
    embodiments: tuple[str, ...]
    implemented_task_variants: tuple[str, ...]
    execution_state: ExecutionState
    blockers: tuple[str, ...]

    def validate(self) -> None:
        if self.corpus_id not in EXPECTED_CORPUS_LEAVES:
            raise RegistryValidationError(
                f"backend advertises unknown corpus leaf {self.corpus_id!r}"
            )
        if not self.family or not self.subfamily or not self.embodiments:
            raise RegistryValidationError(
                f"backend support for {self.corpus_id} is incomplete"
            )
        if len(set(self.embodiments)) != len(self.embodiments):
            raise RegistryValidationError(
                f"backend support for {self.corpus_id} repeats an embodiment"
            )
        unknown = sorted(set(self.embodiments) - KNOWN_EMBODIMENTS)
        if unknown:
            raise RegistryValidationError(
                f"backend support for {self.corpus_id} has unknown embodiments {unknown}"
            )
        if len(set(self.implemented_task_variants)) != len(
            self.implemented_task_variants
        ):
            raise RegistryValidationError(
                f"backend support for {self.corpus_id} repeats an implemented task variant"
            )
        if any(not value.strip() for value in self.implemented_task_variants):
            raise RegistryValidationError(
                f"backend support for {self.corpus_id} has an empty task variant"
            )
        if self.execution_state is ExecutionState.REVIEW and self.blockers:
            raise RegistryValidationError(
                f"review-executable support for {self.corpus_id} cannot retain blockers"
            )
        if self.execution_state is ExecutionState.BLOCKED and not self.blockers:
            raise RegistryValidationError(
                f"blocked support for {self.corpus_id} requires explicit blockers"
            )
        if any(not value.strip() for value in self.blockers):
            raise RegistryValidationError(
                f"backend support for {self.corpus_id} has an empty blocker"
            )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BackendSupport":
        expected_keys = {
            "leaf",
            "family",
            "subfamily",
            "embodiments",
            "implemented_task_variants",
            "execution_state",
            "blockers",
        }
        if set(value) != expected_keys:
            missing = sorted(expected_keys - set(value))
            extra = sorted(set(value) - expected_keys)
            raise RegistryValidationError(
                "backend support entry has non-canonical fields; "
                f"missing={missing}, extra={extra}"
            )
        result = cls(
            corpus_id=str(value.get("leaf", "")),
            family=str(value.get("family", "")),
            subfamily=str(value.get("subfamily", "")),
            embodiments=tuple(str(item) for item in value.get("embodiments", ())),
            implemented_task_variants=tuple(
                str(item) for item in value.get("implemented_task_variants", ())
            ),
            execution_state=_execution_state(value.get("execution_state")),
            blockers=tuple(str(item) for item in value.get("blockers", ())),
        )
        result.validate()
        return result


@dataclass(frozen=True, slots=True)
class BackendCapability:
    name: str
    simulator: str
    integrated_rendering: bool
    evaluator_interface: str
    release_state: ReleaseState
    source_hashes: Mapping[str, str]
    blockers: tuple[str, ...]
    support: tuple[BackendSupport, ...]

    @property
    def support_by_leaf(self) -> dict[str, BackendSupport]:
        return {value.corpus_id: value for value in self.support}

    def validate(self) -> None:
        if not self.name or not self.simulator or not self.evaluator_interface:
            raise RegistryValidationError("backend name, simulator, and evaluator are required")
        if not self.integrated_rendering:
            raise RegistryValidationError(
                f"backend {self.name} lacks required integrated rendering"
            )
        ids = [item.corpus_id for item in self.support]
        if len(ids) != len(set(ids)):
            raise RegistryValidationError(
                f"backend {self.name} repeats a corpus capability"
            )
        for item in self.support:
            item.validate()
        for source_id, digest in self.source_hashes.items():
            if not source_id or not _HASH_PATTERN.fullmatch(str(digest)):
                raise RegistryValidationError(
                    f"backend {self.name} has invalid source hash {source_id!r}={digest!r}"
                )
        if self.release_state is ReleaseState.RELEASED and self.blockers:
            raise RegistryValidationError(
                f"released backend {self.name} cannot retain blockers"
            )
        if self.release_state is not ReleaseState.RELEASED and not self.blockers:
            raise RegistryValidationError(
                f"unreleased backend {self.name} requires explicit blockers"
            )
        if self.release_state.allows_production and not self.source_hashes:
            raise RegistryValidationError(
                f"released backend {self.name} requires pinned source hashes"
            )
        if not self.source_hashes and not any("hash" in value for value in self.blockers):
            raise RegistryValidationError(
                f"unpinned backend {self.name} must expose a source-hash blocker"
            )
        if self.name == PERMANENTLY_BLOCKED_BACKEND:
            if self.release_state is not ReleaseState.PERMANENTLY_BLOCKED:
                raise RegistryValidationError("native_mujoco must remain permanently blocked")
            if self.support:
                raise RegistryValidationError("native_mujoco cannot advertise corpus support")
        elif self.release_state is ReleaseState.PERMANENTLY_BLOCKED:
            raise RegistryValidationError(
                "only the retired native_mujoco backend may be permanently blocked"
            )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BackendCapability":
        hashes = _mapping(value.get("source_hashes", {}), "backend source_hashes")
        result = cls(
            name=str(value.get("name", "")),
            simulator=str(value.get("simulator", "")),
            integrated_rendering=bool(value.get("integrated_rendering", False)),
            evaluator_interface=str(value.get("evaluator_interface", "")),
            release_state=_release_state(value.get("release_state")),
            source_hashes={str(key): str(item) for key, item in hashes.items()},
            blockers=tuple(str(item) for item in value.get("blockers", ())),
            support=tuple(
                BackendSupport.from_dict(_mapping(item, "backend support entry"))
                for item in value.get("support", ())
            ),
        )
        result.validate()
        return result


@dataclass(frozen=True, slots=True)
class BackendCapabilityRegistry:
    registry_id: str
    backends: tuple[BackendCapability, ...]

    @property
    def by_name(self) -> dict[str, BackendCapability]:
        return {backend.name: backend for backend in self.backends}

    def validate(self, corpus: CorpusRegistry) -> None:
        names = [backend.name for backend in self.backends]
        if len(names) != len(set(names)):
            raise RegistryValidationError("backend names must be unique")
        expected = {
            "source_mujoco",
            "source_genesis_fluid",
            "source_mujoco_deformable",
            PERMANENTLY_BLOCKED_BACKEND,
        }
        if set(names) != expected:
            raise RegistryValidationError(
                f"backend registry must contain exactly {sorted(expected)}"
            )
        advertised: dict[str, str] = {}
        for backend in self.backends:
            backend.validate()
            for support in backend.support:
                if support.corpus_id in advertised:
                    raise RegistryValidationError(
                        f"{support.corpus_id} is advertised by multiple backends"
                    )
                advertised[support.corpus_id] = backend.name
                leaf = corpus.resolve(support.corpus_id)
                if backend.name != leaf.backend:
                    raise RegistryValidationError(
                        f"{support.corpus_id} registry/backend mismatch"
                    )
                if (support.family, support.subfamily) != (leaf.family, leaf.subfamily):
                    raise RegistryValidationError(
                        f"{support.corpus_id} family/subfamily mismatch"
                    )
                if support.embodiments != leaf.supported_embodiments:
                    raise RegistryValidationError(
                        f"{support.corpus_id} embodiment capability mismatch"
                    )
                unknown_variants = sorted(
                    set(support.implemented_task_variants) - set(leaf.task_variants)
                )
                if unknown_variants:
                    raise RegistryValidationError(
                        f"{support.corpus_id} advertises unknown implemented task variants "
                        f"{unknown_variants}"
                    )
                if (
                    support.execution_state is ExecutionState.REVIEW
                    and support.implemented_task_variants != leaf.task_variants
                ):
                    raise RegistryValidationError(
                        f"review-executable support for {support.corpus_id} must implement "
                        "every declared task variant in canonical order"
                    )
        if set(advertised) != EXPECTED_CORPUS_LEAVES:
            raise RegistryValidationError(
                "backend capability registry must cover every corpus leaf exactly once"
            )

    def resolve(
        self,
        backend_name: str,
        corpus_id: str,
        embodiment: str,
        *,
        task_variant: str | None = None,
        purpose: str | None = None,
        corpus: CorpusRegistry | None = None,
        activation_report: Mapping[str, Any] | str | Path | None = None,
        require_released: bool = False,
    ) -> BackendCapability:
        try:
            backend = self.by_name[str(backend_name)]
        except KeyError as error:
            raise UnsupportedScenarioError(f"unknown backend {backend_name!r}") from error
        try:
            support = backend.support_by_leaf[str(corpus_id)]
        except KeyError as error:
            raise UnsupportedScenarioError(
                f"backend {backend.name} does not support corpus leaf {corpus_id!r}"
            ) from error
        if embodiment not in support.embodiments:
            raise UnsupportedScenarioError(
                f"backend {backend.name}/{corpus_id} does not support embodiment "
                f"{embodiment!r}; allowed={list(support.embodiments)}"
            )
        normalized_purpose = _generation_purpose(purpose)
        legacy_backend_only_release_check = bool(
            require_released and normalized_purpose is None
        )
        if require_released:
            if normalized_purpose not in {None, "production"}:
                raise ValueError(
                    "require_released cannot be combined with a non-production purpose"
                )
            normalized_purpose = "production"
        if normalized_purpose is not None and not legacy_backend_only_release_check:
            if task_variant is None:
                raise UnsupportedScenarioError(
                    f"{normalized_purpose} resolution requires task_variant"
                )
            if task_variant not in support.implemented_task_variants:
                raise BackendNotReleasedError(
                    f"backend {backend.name}/{corpus_id} is blocked for "
                    f"{normalized_purpose}: task_variant_not_implemented:{task_variant}"
                )
            corpus_registry = corpus or load_corpus_registry()
            corpus_registry.resolve(
                corpus_id,
                embodiment=embodiment,
                task_variant=task_variant,
                purpose=normalized_purpose,
            )
        if normalized_purpose in {"pilot", "production"} and (
            backend.release_state is not ReleaseState.RELEASED
        ):
            if legacy_backend_only_release_check:
                raise BackendNotReleasedError(
                    f"backend {backend.name} is {backend.release_state.value}: "
                    f"{', '.join(backend.blockers)}"
                )
            raise BackendNotReleasedError(
                f"backend {backend.name} must be released for {normalized_purpose}; "
                f"current state is {backend.release_state.value}: "
                f"{', '.join(backend.blockers)}"
            )
        if normalized_purpose == "pilot":
            validate_pilot_activation_report(activation_report, corpus_id=corpus_id)
        return backend


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RegistryValidationError(f"{label} must be a mapping")
    return value


def _generation_purpose(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized == "preview":
        normalized = "review"
    if normalized not in {"review", "pilot", "production"}:
        raise ValueError(f"unknown generation purpose {value!r}")
    return normalized


def validate_pilot_activation_report(
    value: Mapping[str, Any] | str | Path | None,
    *,
    corpus_id: str,
) -> str:
    """Validate immutable fixed-six human approval before pilot construction.

    A self-authored mapping is not evidence. Pilot callers must provide the
    ``activation_report.json`` path inside a complete external publication so
    its ledger, manifest, dataset binding, and sealed source bytes can all be
    revalidated together.
    """

    if value is None:
        raise BackendNotReleasedError(
            f"corpus leaf {corpus_id} lacks a hash-bound pilot activation report"
        )
    if not isinstance(value, (str, Path)):
        raise BackendNotReleasedError(
            "pilot activation requires the path to a complete external review publication"
        )
    try:
        from .review_finalize import validate_external_review_publication

        publication = validate_external_review_publication(value, corpus_id=corpus_id)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise BackendNotReleasedError(
            f"pilot activation publication is invalid for {corpus_id}: {error}"
        ) from error
    return publication.report_sha256


def _release_state(value: Any) -> ReleaseState:
    try:
        return ReleaseState(str(value))
    except ValueError as error:
        raise RegistryValidationError(f"unknown release state {value!r}") from error


def _execution_state(value: Any) -> ExecutionState:
    try:
        return ExecutionState(str(value))
    except ValueError as error:
        raise RegistryValidationError(f"unknown execution state {value!r}") from error


def _read_yaml(path: Path) -> Mapping[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RegistryValidationError(f"registry file is unavailable: {path}") from error
    return _mapping(value, f"registry {path}")


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


@lru_cache(maxsize=1)
def _load_default_corpus_registry() -> CorpusRegistry:
    return load_corpus_registry(_repository_root() / "configs/corpus/dynamic_manipulation_v2.yaml")


def load_corpus_registry(path: str | Path | None = None) -> CorpusRegistry:
    """Load and deeply validate the authoritative 20-leaf corpus registry."""

    if path is None:
        return _load_default_corpus_registry()
    raw = _read_yaml(Path(path))
    if raw.get("schema_version") != CORPUS_REGISTRY_SCHEMA_VERSION:
        raise RegistryValidationError("unsupported corpus registry schema_version")
    media = _mapping(raw.get("canonical_media"), "canonical_media")
    release_policy = _mapping(raw.get("release_policy"), "release_policy")
    leaves_value = raw.get("leaves", ())
    if not isinstance(leaves_value, Sequence) or isinstance(leaves_value, (str, bytes)):
        raise RegistryValidationError("corpus leaves must be a sequence")
    leaves = tuple(CorpusLeaf.from_dict(_mapping(item, "corpus leaf")) for item in leaves_value)
    if int(raw.get("leaf_count", -1)) != len(leaves):
        raise RegistryValidationError("declared leaf_count does not match the registry")
    result = CorpusRegistry(
        registry_id=str(raw.get("registry_id", "")),
        contract_version=str(raw.get("contract_version", "")),
        leaves=leaves,
        canonical_width=int(media.get("width", 0)),
        canonical_height=int(media.get("height", 0)),
        canonical_video_hz=int(media.get("video_hz", 0)),
        canonical_cameras=tuple(str(item) for item in media.get("cameras", ())),
        fixed_review_rollouts=int(release_policy.get("fixed_review_rollouts", 0)),
    )
    result.validate()
    return result


@lru_cache(maxsize=1)
def _load_default_backend_registry() -> BackendCapabilityRegistry:
    return load_backend_capability_registry(
        _repository_root() / "configs/backends/capabilities_v1.yaml",
        corpus=_load_default_corpus_registry(),
    )


def load_backend_capability_registry(
    path: str | Path | None = None,
    *,
    corpus: CorpusRegistry | None = None,
) -> BackendCapabilityRegistry:
    """Load capabilities and require exact agreement with the corpus registry."""

    if path is None and corpus is None:
        return _load_default_backend_registry()
    if path is None:
        path = _repository_root() / "configs/backends/capabilities_v1.yaml"
    raw = _read_yaml(Path(path))
    if raw.get("schema_version") != BACKEND_CAPABILITY_SCHEMA_VERSION:
        raise RegistryValidationError("unsupported backend capability schema_version")
    backends_value = raw.get("backends", ())
    if not isinstance(backends_value, Sequence) or isinstance(backends_value, (str, bytes)):
        raise RegistryValidationError("backends must be a sequence")
    result = BackendCapabilityRegistry(
        registry_id=str(raw.get("registry_id", "")),
        backends=tuple(
            BackendCapability.from_dict(_mapping(item, "backend capability"))
            for item in backends_value
        ),
    )
    result.validate(corpus or load_corpus_registry())
    return result


__all__ = [
    "BACKEND_CAPABILITY_SCHEMA_VERSION",
    "BackendCapability",
    "BackendCapabilityRegistry",
    "BackendNotReleasedError",
    "BackendSupport",
    "CANONICAL_CONTRACT_VERSION",
    "CORPUS_REGISTRY_SCHEMA_VERSION",
    "CorpusLeaf",
    "CorpusRegistry",
    "EXPECTED_BACKEND_BY_LEAF",
    "EXPECTED_CORPUS_LEAVES",
    "ExecutionState",
    "KNOWN_EMBODIMENTS",
    "PERMANENTLY_BLOCKED_BACKEND",
    "PILOT_ACTIVATION_REPORT_SCHEMA",
    "RateSpec",
    "RegistryValidationError",
    "ReleaseState",
    "UnsupportedScenarioError",
    "load_backend_capability_registry",
    "load_corpus_registry",
    "validate_pilot_activation_report",
]
