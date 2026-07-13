"""Deterministic counterfactual-, lineage-, and scene-aware dataset splits."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
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
class SplitAssigner:
    """Assign connected leakage groups to stable 90/5/5 splits."""

    seed: int = 0
    train_fraction: float = 0.90
    validation_fraction: float = 0.05
    test_fraction: float = 0.05

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

    def assign(self, records: Iterable[EpisodeRecord | Mapping[str, Any]]) -> list[SplitAssignment]:
        """Return assignments without mutating input records."""

        values = [_as_mapping(record) for record in records]
        if not values:
            return []
        assignments: list[SplitAssignment | None] = [None] * len(values)
        components = self._components(values)
        groups: list[tuple[str, list[int]]] = []
        for indices in components:
            episode_ids = sorted(str(values[index]["episode_uuid"]) for index in indices)
            group_id = "split-group-" + sha256_json(episode_ids)[:20]
            groups.append((group_id, indices))
        # Stable hash ordering randomizes semantic groups, while remaining-capacity
        # assignment keeps finite datasets near the requested 90/5/5 proportions.
        groups.sort(key=lambda item: stable_uint64(item[0], namespace=f"dataset-split-order:{self.seed}"))
        targets = {
            Split.TRAIN: len(values) * self.train_fraction,
            Split.VALIDATION: len(values) * self.validation_fraction,
            Split.TEST: len(values) * self.test_fraction,
        }
        counts = {split: 0 for split in targets}
        tie_order = {Split.TRAIN: 0, Split.VALIDATION: 1, Split.TEST: 2}
        for group_id, indices in groups:
            split = max(
                targets,
                key=lambda candidate: (targets[candidate] - counts[candidate], -tie_order[candidate]),
            )
            counts[split] += len(indices)
            for index in indices:
                record = values[index]
                assignments[index] = SplitAssignment(
                    episode_uuid=str(record["episode_uuid"]),
                    episode_index=int(record["episode_index"]),
                    split_group_id=group_id,
                    split=split.value,
                )
        return [assignment for assignment in assignments if assignment is not None]

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
