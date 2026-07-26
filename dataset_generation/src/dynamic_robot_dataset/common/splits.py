"""Deterministic counterfactual-, lineage-, and scene-aware dataset splits."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

from .hashing import sha256_json, stable_uint64
from .schema import EpisodeRecord, Split


def _as_mapping(record: EpisodeRecord | Mapping[str, Any]) -> Mapping[str, Any]:
    return record.to_dict() if isinstance(record, EpisodeRecord) else record


def _value(record: Mapping[str, Any], field: str) -> Any:
    value = record.get(field)
    if value is not None and str(value) != "":
        return value
    extras = record.get("extras")
    return extras.get(field) if isinstance(extras, Mapping) else None


class _DisjointSet:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


@dataclass(slots=True, frozen=True)
class SplitAssignment:
    """One auditable episode-to-split assignment."""

    episode_uuid: str
    episode_index: int
    split_group_id: str
    split: str


@dataclass(slots=True)
class SplitDiagnostics:
    """Coverage diagnostics emitted without weakening leakage grouping."""

    requested_fractions: dict[str, float]
    actual_counts: dict[str, int]
    stratum_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    sparse_strata: list[str] = field(default_factory=list)
    partition_policy_conflicts: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.sparse_strata and not self.partition_policy_conflicts


@dataclass(slots=True)
class SplitAssigner:
    """Assign connected leakage groups to stable, stratified 80/10/10 splits."""

    seed: int | str = 0
    train_fraction: float = 0.80
    validation_fraction: float = 0.10
    test_fraction: float = 0.10

    def __post_init__(self) -> None:
        fractions = self.train_fraction + self.validation_fraction + self.test_fraction
        if abs(fractions - 1.0) > 1e-9 or min(
            self.train_fraction, self.validation_fraction, self.test_fraction
        ) < 0:
            raise ValueError("Split fractions must be non-negative and sum to one")

    def _components(self, records: Sequence[Mapping[str, Any]]) -> list[list[int]]:
        dsu = _DisjointSet(len(records))
        seen_token: dict[tuple[Any, ...], int] = {}
        uuid_to_index = {str(record.get("episode_uuid")): index for index, record in enumerate(records)}

        for index, record in enumerate(records):
            family_scene = (record.get("family"), record.get("subfamily"), record.get("scene_seed"))
            tokens: list[tuple[Any, ...]] = []
            for field in (
                "counterfactual_bundle_id",
                "physics_counterfactual_family_id",
                "split_group_id",
                "initial_state_hash",
                "trajectory_hash",
                "scene_instance_id",
                "rerender_lineage_id",
                "background_scene_id",
                "camera_rig_hash",
                "appearance_instance_hash",
            ):
                value = _value(record, field)
                if value is not None and str(value) != "":
                    tokens.append((field, str(value)))
            if family_scene[2] is not None:
                tokens.append(("family_scene_seed", *family_scene))
            for token in tokens:
                if token in seen_token:
                    dsu.union(index, seen_token[token])
                else:
                    seen_token[token] = index
            parent = record.get("parent_episode_uuid")
            if parent is not None and str(parent) in uuid_to_index:
                dsu.union(index, uuid_to_index[str(parent)])

        groups: dict[int, list[int]] = defaultdict(list)
        for index in range(len(records)):
            groups[dsu.find(index)].append(index)
        return list(groups.values())

    def _split_for_group(self, group_id: str) -> Split:
        unit = stable_uint64(group_id, namespace=f"dataset-split:{self.seed}") / float(1 << 64)
        if unit < self.train_fraction:
            return Split.TRAIN
        if unit < self.train_fraction + self.validation_fraction:
            return Split.VALIDATION
        return Split.TEST

    @staticmethod
    def _range_partition(record: Mapping[str, Any]) -> str | None:
        physics = record.get("physics")
        if not isinstance(physics, Mapping):
            return None
        provenance = physics.get("parameter_range_provenance")
        if not isinstance(provenance, Mapping):
            return None
        value = provenance.get("partition")
        return None if value in {None, ""} else str(value)

    @staticmethod
    def _stratum(record: Mapping[str, Any]) -> tuple[str, ...]:
        """Return the release-balancing signature for one logical episode."""

        randomization = record.get("randomization")
        randomization = randomization if isinstance(randomization, Mapping) else {}
        extras = record.get("extras")
        extras = extras if isinstance(extras, Mapping) else {}
        physics = record.get("physics")
        physics = physics if isinstance(physics, Mapping) else {}
        range_provenance = physics.get("parameter_range_provenance")
        range_provenance = (
            range_provenance if isinstance(range_provenance, Mapping) else {}
        )
        physics_bin = (
            extras.get("physics_bin")
            or extras.get("parameter_bin")
            or range_provenance.get("partition")
            or "<missing>"
        )
        return (
            str(record.get("family", "<missing>")),
            str(record.get("subfamily", "<missing>")),
            str(
                record.get("actual_outcome_class")
                or record.get("actual_outcome", "<missing>")
            ),
            str(randomization.get("background_style") or randomization.get("scene_style") or "<missing>"),
            str(randomization.get("object_asset_id") or "<missing>"),
            str(record.get("tool_type") or "<missing>"),
            str(physics_bin),
        )

    @staticmethod
    def _stratum_name(stratum: tuple[str, ...]) -> str:
        return " | ".join(stratum)

    def assign_with_diagnostics(
        self, records: Iterable[EpisodeRecord | Mapping[str, Any]]
    ) -> tuple[list[SplitAssignment], SplitDiagnostics]:
        """Return assignments and coverage diagnostics without splitting siblings."""

        values = [_as_mapping(record) for record in records]
        if not values:
            return [], SplitDiagnostics(
                {
                    "train": self.train_fraction,
                    "validation": self.validation_fraction,
                    "test": self.test_fraction,
                },
                {"train": 0, "validation": 0, "test": 0},
            )
        assignments: list[SplitAssignment | None] = [None] * len(values)
        components = self._components(values)
        groups: list[
            tuple[str, list[int], Counter[tuple[str, ...]], Split | None]
        ] = []
        partition_policy_conflicts: list[str] = []
        for indices in components:
            episode_ids = sorted(str(values[index]["episode_uuid"]) for index in indices)
            declared_group_ids = {
                str(_value(values[index], "split_group_id"))
                for index in indices
                if _value(values[index], "split_group_id") not in {None, ""}
            }
            # Preserve a generator-declared leakage identity. Replacing it
            # would make finalized episode rows disagree with the immutable
            # pre-simulation counterfactual declaration table.
            group_id = (
                next(iter(declared_group_ids))
                if len(declared_group_ids) == 1
                else "connected-split-group-" + sha256_json(episode_ids)[:20]
            )
            partitions = {
                self._range_partition(values[index]) for index in indices
            } - {None}
            forced_split: Split | None = None
            if "test_ood" in partitions:
                forced_split = Split.TEST
            elif "validation_id" in partitions:
                forced_split = Split.VALIDATION
            if "test_ood" in partitions and "validation_id" in partitions:
                partition_policy_conflicts.append(
                    f"connected group {group_id} mixes test_ood and validation_id"
                )
            groups.append(
                (
                    group_id,
                    indices,
                    Counter(self._stratum(values[index]) for index in indices),
                    forced_split,
                )
            )
        # Stable hash ordering randomizes semantic groups, while remaining-capacity
        # assignment keeps finite datasets near the requested 80/10/10 proportions.
        groups.sort(key=lambda item: stable_uint64(item[0], namespace=f"dataset-split-order:{self.seed}"))
        targets = {
            Split.TRAIN: len(values) * self.train_fraction,
            Split.VALIDATION: len(values) * self.validation_fraction,
            Split.TEST: len(values) * self.test_fraction,
        }
        counts: dict[Split, int] = {split: 0 for split in targets}
        total_strata = Counter(self._stratum(record) for record in values)
        stratum_targets = {
            stratum: {split: count * fraction for split, fraction in (
                (Split.TRAIN, self.train_fraction),
                (Split.VALIDATION, self.validation_fraction),
                (Split.TEST, self.test_fraction),
            )}
            for stratum, count in total_strata.items()
        }
        stratum_counts: dict[tuple[str, ...], dict[Split, int]] = {
            stratum: {Split.TRAIN: 0, Split.VALIDATION: 0, Split.TEST: 0}
            for stratum in total_strata
        }
        tie_order = {Split.TRAIN: 0, Split.VALIDATION: 1, Split.TEST: 2}
        group_splits: dict[str, Split] = {}
        def allocation_error(
            candidate_counts: Mapping[Split, int],
            candidate_strata: Mapping[tuple[str, ...], Mapping[Split, int]],
        ) -> float:
            overall = sum(
                ((candidate_counts[split] - target) / max(target, 1.0)) ** 2
                for split, target in targets.items()
            )
            stratified = 0.0
            for stratum, per_split in candidate_strata.items():
                for split, target in stratum_targets[stratum].items():
                    stratified += ((per_split[split] - target) / max(target, 1.0)) ** 2
            return overall + stratified

        for group_id, indices, histogram, forced_split in groups:
            candidates: list[tuple[float, int, Split]] = []
            candidate_splits = (
                (forced_split,) if forced_split is not None else tuple(targets)
            )
            for candidate in candidate_splits:
                candidate_counts = dict(counts)
                candidate_counts[candidate] += len(indices)
                candidate_strata = {
                    stratum: dict(per_split) for stratum, per_split in stratum_counts.items()
                }
                for stratum, amount in histogram.items():
                    candidate_strata[stratum][candidate] += amount
                candidates.append(
                    (allocation_error(candidate_counts, candidate_strata), tie_order[candidate], candidate)
                )
            split = min(candidates)[2]
            group_splits[group_id] = split
            counts[split] += len(indices)
            for stratum, amount in histogram.items():
                stratum_counts[stratum][split] += amount

        # A large connected component encountered near a capacity boundary can
        # overshoot one bucket even when later small groups could repair the
        # ratio.  Deterministic single-group moves minimize total squared count
        # error without ever breaking a leakage component.
        while True:
            current_error = allocation_error(counts, stratum_counts)
            best_move: tuple[
                tuple[float, int, int], str, Split, Split, dict[Split, int],
                dict[tuple[str, ...], dict[Split, int]],
            ] | None = None
            for group_index, (
                group_id,
                indices,
                histogram,
                forced_split,
            ) in enumerate(groups):
                if forced_split is not None:
                    continue
                source = group_splits[group_id]
                size = len(indices)
                for destination in targets:
                    if destination == source:
                        continue
                    candidate_counts = dict(counts)
                    candidate_counts[source] -= size
                    candidate_counts[destination] += size
                    candidate_strata = {
                        stratum: dict(per_split) for stratum, per_split in stratum_counts.items()
                    }
                    for stratum, amount in histogram.items():
                        candidate_strata[stratum][source] -= amount
                        candidate_strata[stratum][destination] += amount
                    error = allocation_error(candidate_counts, candidate_strata)
                    key = (error, group_index, tie_order[destination])
                    if error < current_error and (best_move is None or key < best_move[0]):
                        best_move = (
                            key,
                            group_id,
                            source,
                            destination,
                            candidate_counts,
                            candidate_strata,
                        )
            if best_move is None:
                break
            _, group_id, _, destination, counts, stratum_counts = best_move
            group_splits[group_id] = destination

        for group_id, indices, _, _ in groups:
            split = group_splits[group_id]
            for index in indices:
                record = values[index]
                assignments[index] = SplitAssignment(
                    episode_uuid=str(record["episode_uuid"]),
                    episode_index=int(record["episode_index"]),
                    # Preserve the declared group identity in metadata. Extra
                    # duplicate/lineage edges can join several declared groups
                    # for assignment without rewriting their counterfactual
                    # contracts; the whole connected component still receives
                    # the same split.
                    split_group_id=str(
                        _value(record, "split_group_id") or group_id
                    ),
                    split=split.value,
                )
        result = [assignment for assignment in assignments if assignment is not None]
        component_sets: dict[tuple[str, ...], set[str]] = defaultdict(set)
        for group_id, _, histogram, _ in groups:
            for stratum in histogram:
                component_sets[stratum].add(group_id)
        nonzero_split_count = sum(
            fraction > 0
            for fraction in (self.train_fraction, self.validation_fraction, self.test_fraction)
        )
        sparse = sorted(
            self._stratum_name(stratum)
            for stratum, group_ids in component_sets.items()
            if len(group_ids) < nonzero_split_count
            or any(
                stratum_counts[stratum][split] == 0
                for split, fraction in (
                    (Split.TRAIN, self.train_fraction),
                    (Split.VALIDATION, self.validation_fraction),
                    (Split.TEST, self.test_fraction),
                )
                if fraction > 0
            )
        )
        diagnostics = SplitDiagnostics(
            requested_fractions={
                "train": self.train_fraction,
                "validation": self.validation_fraction,
                "test": self.test_fraction,
            },
            actual_counts={split.value: counts[split] for split in counts},
            stratum_counts={
                self._stratum_name(stratum): {
                    split.value: count for split, count in per_split.items()
                }
                for stratum, per_split in sorted(stratum_counts.items())
            },
            sparse_strata=sparse,
            partition_policy_conflicts=sorted(partition_policy_conflicts),
        )
        return result, diagnostics

    def assign(self, records: Iterable[EpisodeRecord | Mapping[str, Any]]) -> list[SplitAssignment]:
        """Return assignments without mutating input records."""

        assignments, _ = self.assign_with_diagnostics(records)
        return assignments

    def assignment_map(self, records: Iterable[EpisodeRecord | Mapping[str, Any]]) -> dict[str, str]:
        """Return ``episode_uuid -> split`` for convenience."""

        return {assignment.episode_uuid: assignment.split for assignment in self.assign(records)}


def validate_no_split_leakage(
    records: Iterable[EpisodeRecord | Mapping[str, Any]],
    assignments: Iterable[SplitAssignment | Mapping[str, Any]] | None = None,
) -> list[str]:
    """Return leakage descriptions; an empty list means grouping is consistent."""

    values = [_as_mapping(record) for record in records]
    if assignments is None:
        split_by_uuid = {str(value["episode_uuid"]): str(value.get("split", "unassigned")) for value in values}
    else:
        split_by_uuid = {
            str(item.episode_uuid if isinstance(item, SplitAssignment) else item["episode_uuid"]): str(
                item.split if isinstance(item, SplitAssignment) else item["split"]
            )
            for item in assignments
        }
    problems: list[str] = []
    missing = sorted(
        str(value["episode_uuid"])
        for value in values
        if str(value["episode_uuid"]) not in split_by_uuid
    )
    if missing:
        problems.append(f"missing split assignments: {missing}")
    for component in SplitAssigner()._components(values):
        episode_ids = [str(values[index]["episode_uuid"]) for index in component]
        splits = {split_by_uuid[episode_id] for episode_id in episode_ids if episode_id in split_by_uuid}
        if len(splits) > 1:
            problems.append(
                f"connected leakage group {sorted(episode_ids)} crosses splits: {sorted(splits)}"
            )
    problems.extend(validate_partition_split_policy(values, split_by_uuid))
    return sorted(problems)


def validate_partition_split_policy(
    records: Iterable[EpisodeRecord | Mapping[str, Any]],
    split_by_uuid: Mapping[str, str],
) -> list[str]:
    """Require held-out physics partitions to remain in held-out splits.

    A connected physics family may contain both train-ID controls and test-OOD
    interventions. In that case the entire leakage component is assigned to
    test; train-ID is an admissible range label, not a requirement to train on
    every sampled episode.
    """

    problems: list[str] = []
    for source in records:
        record = _as_mapping(source)
        episode_uuid = str(record["episode_uuid"])
        split = split_by_uuid.get(episode_uuid)
        partition = SplitAssigner._range_partition(record)
        required = {
            "validation_id": Split.VALIDATION.value,
            "test_ood": Split.TEST.value,
        }.get(partition)
        if required is not None and split != required:
            problems.append(
                f"episode {episode_uuid} with physics partition {partition} "
                f"must be assigned to {required}, not {split or '<missing>'}"
            )
    return sorted(problems)


def apply_assignments(
    records: Iterable[Mapping[str, Any]], assignments: Iterable[SplitAssignment]
) -> list[dict[str, Any]]:
    """Return copied records with ``split`` and ``split_group_id`` populated."""

    by_uuid = {assignment.episode_uuid: assignment for assignment in assignments}
    output: list[dict[str, Any]] = []
    for source in records:
        value = dict(source)
        assignment = by_uuid[str(value["episode_uuid"])]
        value["split"] = assignment.split
        value["split_group_id"] = assignment.split_group_id
        output.append(value)
    return output


def ood_manifest(records: Iterable[Mapping[str, Any]], held_out: Mapping[str, set[Any]]) -> list[str]:
    """Return episode UUIDs matching any explicitly held-out OOD attribute."""

    return sorted(
        str(record["episode_uuid"])
        for record in records
        if any(record.get(field) in values for field, values in held_out.items())
    )
