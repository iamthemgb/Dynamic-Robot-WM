"""Separate, read-only namespace for assisted or state-scripted legacy data."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import yaml

from .hashing import sha256_file, sha256_json


LEGACY_QUARANTINE_SCHEMA = "dynamic-robot-legacy-quarantine/v1"
LEGACY_ASSISTED_NAMESPACE = "legacy_assisted"


@dataclass(frozen=True, slots=True)
class LegacyCollection:
    collection_id: str
    corpus_leaf_id: str
    root: str
    episode_count: int
    default_release_tier: str
    prohibited_mechanisms_observed: tuple[str, ...]
    blockers: tuple[str, ...]
    snapshot_sha256: Mapping[str, str]

    def validate(self) -> None:
        if not self.collection_id or not self.corpus_leaf_id or not self.root:
            raise ValueError("legacy collection identity and root are required")
        if self.episode_count <= 0:
            raise ValueError("legacy collection episode_count must be positive")
        if self.default_release_tier not in {"assisted_contact", "scripted_motion"}:
            raise ValueError("legacy collections must use an explicitly quarantined tier")
        if not self.prohibited_mechanisms_observed or not self.blockers:
            raise ValueError("legacy collections require mechanism evidence and blockers")
        if not self.snapshot_sha256:
            raise ValueError("legacy collection requires snapshot hashes")
        for relative, digest in self.snapshot_sha256.items():
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts or not relative:
                raise ValueError("legacy snapshot paths must be portable relative paths")
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError("legacy snapshot hashes must be lowercase SHA-256")

    def verify_snapshot(self) -> None:
        """Rehash the small metadata snapshot without modifying the source tree."""

        self.validate()
        root = Path(self.root).resolve(strict=True)
        for relative, expected in self.snapshot_sha256.items():
            candidate = (root / relative).resolve(strict=True)
            try:
                candidate.relative_to(root)
            except ValueError as error:
                raise ValueError(f"legacy snapshot path escapes its root: {relative}") from error
            if not candidate.is_file() or sha256_file(candidate) != expected:
                raise ValueError(f"legacy snapshot differs from its read-only pin: {relative}")


@dataclass(frozen=True, slots=True)
class LegacyQuarantinePolicy:
    namespace: str
    read_only: bool
    training_eligible: bool
    release_eligible: bool
    counts_toward_generated_hours: bool
    may_satisfy_review_or_scale_gate: bool
    copy_into_canonical_namespace: bool
    collections: tuple[LegacyCollection, ...]
    schema_version: str = LEGACY_QUARANTINE_SCHEMA

    def validate(self) -> None:
        if self.schema_version != LEGACY_QUARANTINE_SCHEMA:
            raise ValueError("unsupported legacy quarantine schema")
        if self.namespace != LEGACY_ASSISTED_NAMESPACE:
            raise ValueError("assisted legacy data must remain in legacy_assisted")
        if not self.read_only or any(
            (
                self.training_eligible,
                self.release_eligible,
                self.counts_toward_generated_hours,
                self.may_satisfy_review_or_scale_gate,
                self.copy_into_canonical_namespace,
            )
        ):
            raise ValueError("legacy quarantine policy cannot enable release, hours, or copying")
        identifiers = [value.collection_id for value in self.collections]
        if not identifiers or len(identifiers) != len(set(identifiers)):
            raise ValueError("legacy quarantine requires unique collections")
        for collection in self.collections:
            collection.validate()

    @property
    def policy_sha256(self) -> str:
        self.validate()
        return sha256_json(asdict(self))


def _default_path() -> Path:
    return Path(__file__).resolve().parents[3] / "configs/corpus/legacy_assisted_quarantine_v1.yaml"


@lru_cache(maxsize=1)
def load_legacy_quarantine_policy(
    path: str | Path | None = None,
    *,
    verify_external: bool = False,
) -> LegacyQuarantinePolicy:
    source = _default_path() if path is None else Path(path)
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("legacy quarantine config must be a mapping")
    policy = raw.get("policy")
    if not isinstance(policy, Mapping):
        raise ValueError("legacy quarantine config lacks policy")
    collections: list[LegacyCollection] = []
    for value in raw.get("collections", ()):
        if not isinstance(value, Mapping):
            raise ValueError("legacy quarantine collection must be a mapping")
        hashes = value.get("snapshot_sha256")
        if not isinstance(hashes, Mapping):
            raise ValueError("legacy quarantine collection lacks snapshot hashes")
        collections.append(
            LegacyCollection(
                collection_id=str(value.get("collection_id") or ""),
                corpus_leaf_id=str(value.get("corpus_leaf_id") or ""),
                root=str(value.get("root") or ""),
                episode_count=int(value.get("episode_count", 0)),
                default_release_tier=str(value.get("default_release_tier") or ""),
                prohibited_mechanisms_observed=tuple(
                    str(item) for item in value.get("prohibited_mechanisms_observed", ())
                ),
                blockers=tuple(str(item) for item in value.get("blockers", ())),
                snapshot_sha256={str(key): str(item) for key, item in hashes.items()},
            )
        )
    result = LegacyQuarantinePolicy(
        schema_version=str(raw.get("schema_version") or ""),
        namespace=str(raw.get("namespace") or ""),
        read_only=bool(policy.get("read_only")),
        training_eligible=bool(policy.get("training_eligible")),
        release_eligible=bool(policy.get("release_eligible")),
        counts_toward_generated_hours=bool(
            policy.get("counts_toward_generated_hours")
        ),
        may_satisfy_review_or_scale_gate=bool(
            policy.get("may_satisfy_review_or_scale_gate")
        ),
        copy_into_canonical_namespace=bool(
            policy.get("copy_into_canonical_namespace")
        ),
        collections=tuple(collections),
    )
    result.validate()
    if verify_external:
        for collection in result.collections:
            collection.verify_snapshot()
    return result


__all__ = [
    "LEGACY_ASSISTED_NAMESPACE",
    "LEGACY_QUARANTINE_SCHEMA",
    "LegacyCollection",
    "LegacyQuarantinePolicy",
    "load_legacy_quarantine_policy",
]
