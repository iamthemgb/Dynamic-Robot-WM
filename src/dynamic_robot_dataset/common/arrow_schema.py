"""Versioned, fail-closed Arrow schemas for episode sidecar tables.

The dataset stores one Parquet file per episode and table kind.  Relying on
``pyarrow`` inference for those files makes a column's physical type depend on
the values in that particular episode (most visibly ``list<null>`` for empty
actions).  This module fixes the canonical columns and deterministically types
extensions.  Extensions without any type evidence must be declared by the
writer configuration rather than silently acquiring Arrow's null type.
"""

from __future__ import annotations

import json
import math
import numbers
from collections.abc import Mapping, Sequence
from typing import Any

from .hashing import sha256_json


ARROW_SIDECAR_SCHEMA_VERSION = "dynamic-robot-arrow-sidecars/v1"
TABLE_KINDS = ("frame", "high_rate", "events", "transitions", "objects")

_TYPE_NAMES = frozenset(
    {
        "bool",
        "int64",
        "float64",
        "string",
        "list<bool>",
        "list<int64>",
        "list<float64>",
        "list<string>",
        "json",
    }
)

# Order is part of the contract.  All fields are nullable at the Arrow layer;
# semantic requiredness is checked before table construction.
_CORE_FIELDS: dict[str, tuple[tuple[str, str], ...]] = {
    "frame": (
        ("episode_index", "int64"),
        ("task_index", "int64"),
        ("frame_index", "int64"),
        ("video_frame_index", "int64"),
        ("timestamp", "float64"),
        ("simulation_timestamp", "float64"),
        ("synchronization_error_s", "float64"),
        ("action.actuator_command", "list<float64>"),
        ("simulator.applied_actuator_ctrl", "list<float64>"),
        ("action.mode", "string"),
        ("contact.active", "bool"),
        ("event.contact", "bool"),
        ("assistance.active", "bool"),
        ("assistance.assisted_grasp", "bool"),
        ("assistance.assisted_retention", "bool"),
        ("assistance.equality_constraint_active", "bool"),
        ("assistance.latch_active", "bool"),
        ("assistance.mechanism_ids", "list<string>"),
    ),
    "high_rate": (
        ("episode_index", "int64"),
        ("timestamp", "float64"),
        ("action.actuator_command", "list<float64>"),
        ("simulator.applied_actuator_ctrl", "list<float64>"),
        ("action.mode", "string"),
    ),
    "events": (
        ("episode_index", "int64"),
        ("timestamp", "float64"),
        ("object_a", "string"),
        ("object_b", "string"),
        ("point_world_m", "list<float64>"),
        ("normal_world", "list<float64>"),
        ("penetration_depth_m", "float64"),
        ("contact_category", "string"),
        ("normal_force_n", "float64"),
        ("normal_impulse_n_s", "float64"),
        ("relative_velocity_world_m_s", "list<float64>"),
        ("expected_fixture_contact", "bool"),
        ("snag", "bool"),
    ),
    "transitions": (
        ("episode_index", "int64"),
        ("timestamp", "float64"),
        ("event_type", "string"),
        ("from", "string"),
        ("to", "string"),
        ("active_surface", "string"),
    ),
    "objects": (
        ("episode_index", "int64"),
        ("timestamp", "float64"),
        ("object_id", "string"),
    ),
}

_CONTRACTS = {
    "frame": "dynamic-robot-frames/v1",
    "high_rate": "dynamic-robot-high-rate/v1",
    "events": "dynamic-robot-contact-events/v1",
    "transitions": "dynamic-robot-transitions/v1",
    "objects": "dynamic-robot-object-states/v1",
}


def normalize_extra_field_declarations(
    value: Mapping[str, Mapping[str, str]] | None,
) -> dict[str, dict[str, str]]:
    """Validate a portable table/column/type declaration mapping."""

    result = {kind: {} for kind in TABLE_KINDS}
    if value is None:
        return result
    if not isinstance(value, Mapping):
        raise ValueError("arrow_extra_fields must be a mapping by table kind")
    unknown_tables = sorted(set(value) - set(TABLE_KINDS))
    if unknown_tables:
        raise ValueError(f"Unknown Arrow sidecar table kinds: {unknown_tables}")
    for kind, raw_fields in value.items():
        if not isinstance(raw_fields, Mapping):
            raise ValueError(f"arrow_extra_fields.{kind} must be a mapping")
        core_names = {name for name, _ in _CORE_FIELDS[kind]}
        for raw_name, raw_type in raw_fields.items():
            name = str(raw_name)
            type_name = str(raw_type)
            if not name or name in core_names:
                raise ValueError(
                    f"Arrow extra field {kind}.{name!r} is empty or overrides a core field"
                )
            if type_name not in _TYPE_NAMES:
                raise ValueError(
                    f"Unsupported Arrow type {type_name!r} for {kind}.{name}; "
                    f"choose one of {sorted(_TYPE_NAMES)}"
                )
            result[kind][name] = type_name
    return {kind: dict(sorted(fields.items())) for kind, fields in result.items()}


def arrow_schema_identity(extra_fields: Mapping[str, Mapping[str, str]]) -> str:
    """Return the immutable identity bound into writer resume settings."""

    return sha256_json(
        {
            "version": ARROW_SIDECAR_SCHEMA_VERSION,
            "core_fields": _CORE_FIELDS,
            "extra_fields": extra_fields,
        }
    )


def _arrow_type(pa: Any, type_name: str) -> Any:
    scalar = {
        "bool": pa.bool_(),
        "int64": pa.int64(),
        "float64": pa.float64(),
        "string": pa.string(),
        "json": pa.string(),
    }
    if type_name in scalar:
        return scalar[type_name]
    element_name = type_name.removeprefix("list<").removesuffix(">")
    return pa.list_(_arrow_type(pa, element_name))


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _semantic_extension_type(name: str) -> str | None:
    """Type common state/action names even when every value is empty/null."""

    lower = name.lower()
    if lower.endswith("_json"):
        return "string"
    if lower.endswith(("_index", ".index", "_count", ".count")):
        return "int64"
    if lower.startswith(("is_", "has_")) or lower.endswith(
        (".active", ".enabled", "_flag", ".bilateral", ".retained")
    ):
        return "bool"
    if lower.endswith(
        (
            ".position",
            ".joint_position",
            ".linear_velocity",
            ".angular_velocity",
            ".joint_velocity",
            ".actuator_force",
            ".quaternion_wxyz",
            "_world_m",
            "_world_m_s",
            "_world_n_s",
            "_position_m",
            "_velocity_m_s",
        )
    ):
        return "list<float64>"
    if lower.endswith(("_mode", ".mode", "_phase", ".phase", "_role", ".role")):
        return "string"
    return None


def _inferred_extension_type(name: str, values: Sequence[Any]) -> str:
    semantic = _semantic_extension_type(name)
    if semantic is not None:
        return semantic
    present = [value for value in values if value is not None]
    if not present:
        raise ValueError(
            f"Arrow extension field {name!r} is all-null; declare its type in "
            "resolved_config.arrow_extra_fields"
        )
    if all(isinstance(value, bool) for value in present):
        return "bool"
    if all(isinstance(value, numbers.Integral) and not isinstance(value, bool) for value in present):
        # Unknown physical numeric fields are promoted to float64 so an episode
        # containing only integral-valued samples cannot drift from one with a
        # fractional sample. Semantic indices/counts were handled above.
        return "float64"
    if all(isinstance(value, numbers.Real) and not isinstance(value, bool) for value in present):
        return "float64"
    if all(isinstance(value, str) for value in present):
        return "string"
    if all(isinstance(value, Mapping) for value in present):
        return "json"
    if all(_is_sequence(value) for value in present):
        elements = [element for value in present for element in value]
        if not elements:
            raise ValueError(
                f"Arrow extension field {name!r} contains only empty lists; declare its type in "
                "resolved_config.arrow_extra_fields"
            )
        if all(isinstance(value, bool) for value in elements):
            return "list<bool>"
        if all(isinstance(value, numbers.Integral) and not isinstance(value, bool) for value in elements):
            return "list<float64>"
        if all(isinstance(value, numbers.Real) and not isinstance(value, bool) for value in elements):
            return "list<float64>"
        if all(isinstance(value, str) for value in elements):
            return "list<string>"
    raise ValueError(
        f"Arrow extension field {name!r} has mixed or unsupported values; declare a "
        "single portable type or encode it as json"
    )


def _validate_finite(value: Any, label: str) -> None:
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            raise ValueError(f"{label} contains a non-finite numeric value")
    elif _is_sequence(value):
        for item in value:
            _validate_finite(item, label)


def _source_requirements(
    kind: str,
    rows: Sequence[Mapping[str, Any]],
    source_scenario: Mapping[str, Any] | None,
) -> None:
    if source_scenario is None:
        return
    embodiment = source_scenario.get("embodiment")
    if not isinstance(embodiment, Mapping):
        raise ValueError("Source scenario embodiment is missing from persisted metadata")
    action_names = embodiment.get("action_names")
    if not _is_sequence(action_names):
        raise ValueError("Source scenario embodiment action_names must be a sequence")
    expected_action_size = len(action_names)
    expected_semantics = str(embodiment.get("action_semantics", ""))
    if kind in {"frame", "high_rate"}:
        for index, row in enumerate(rows):
            action = row.get("action.actuator_command")
            applied = row.get("simulator.applied_actuator_ctrl")
            if not _is_sequence(action) or len(action) != expected_action_size:
                raise ValueError(
                    f"Source {kind} row {index} requires {expected_action_size} actual actuator commands"
                )
            if not _is_sequence(applied) or len(applied) != expected_action_size:
                raise ValueError(
                    f"Source {kind} row {index} requires an exact simulator applied-control echo"
                )
            if tuple(float(value) for value in applied) != tuple(
                float(value) for value in action
            ):
                raise ValueError(
                    f"Source {kind} row {index} persisted action differs from applied data.ctrl"
                )
            if str(row.get("action.mode", "")) != expected_semantics:
                raise ValueError(
                    f"Source {kind} row {index} action.mode differs from embodiment semantics"
                )
    if kind == "frame":
        for index, row in enumerate(rows):
            if row.get("simulation_timestamp") is None or row.get("synchronization_error_s") is None:
                raise ValueError(
                    f"Source frame row {index} requires simulation_timestamp and synchronization_error_s"
                )
    if kind == "events":
        for index, row in enumerate(rows):
            if not str(row.get("contact_category", "")).strip():
                raise ValueError(f"Source contact row {index} requires contact_category")


def canonical_sidecar_table(
    pa: Any,
    kind: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    declared_extra_fields: Mapping[str, str] | None = None,
    source_scenario: Mapping[str, Any] | None = None,
) -> Any:
    """Build a typed table with stable core and extension physical types."""

    if kind not in TABLE_KINDS:
        raise ValueError(f"Unknown Arrow sidecar table kind: {kind}")
    values = [dict(row) for row in rows]
    _source_requirements(kind, values, source_scenario)
    declared = dict(declared_extra_fields or {})
    core = dict(_CORE_FIELDS[kind])
    extra_names = sorted(({name for row in values for name in row} | set(declared)) - set(core))
    extra_types: dict[str, str] = {}
    for name in extra_names:
        extra_types[name] = declared.get(name) or _inferred_extension_type(
            name, [row.get(name) for row in values]
        )
    declarations = [*_CORE_FIELDS[kind], *sorted(extra_types.items())]
    names = [name for name, _ in declarations]
    normalized_rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(values):
        normalized: dict[str, Any] = {}
        for name, type_name in declarations:
            value = row.get(name)
            _validate_finite(value, f"{kind} row {row_index} field {name!r}")
            if type_name == "json" and value is not None:
                value = json.dumps(
                    value,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                )
            normalized[name] = value
        normalized_rows.append(normalized)
    identity = {
        "version": ARROW_SIDECAR_SCHEMA_VERSION,
        "contract": _CONTRACTS[kind],
        "fields": declarations,
    }
    schema = pa.schema(
        [pa.field(name, _arrow_type(pa, type_name)) for name, type_name in declarations],
        metadata={
            b"contract": _CONTRACTS[kind].encode("utf-8"),
            b"schema_strategy": ARROW_SIDECAR_SCHEMA_VERSION.encode("utf-8"),
            b"schema_identity_sha256": sha256_json(identity).encode("ascii"),
            b"field_declarations_json": json.dumps(
                declarations, separators=(",", ":"), ensure_ascii=True
            ).encode("ascii"),
        },
    )
    # ``from_pylist`` with an explicit schema also guarantees empty tables
    # retain the full physical contract.
    table = pa.Table.from_pylist(normalized_rows, schema=schema)
    if table.schema.names != names:  # pragma: no cover - defensive Arrow API guard
        raise RuntimeError("PyArrow changed the declared sidecar column order")
    return table
