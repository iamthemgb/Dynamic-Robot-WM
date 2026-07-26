"""Measured-response acceptance checks for counterfactual physics sweeps.

This module is shared by suite execution and later readiness recomputation so
an execution-report boolean can never substitute for the underlying five-point
evidence.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .schema import EpisodeRecord


def sweep_acceptance_evidence(
    planned: Sequence[Any],
    committed_records: Sequence[EpisodeRecord],
    requirements: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate finite, sampled, monotonic sweep responses from saved records.

    ``planned`` elements follow :class:`PlannedSuiteEpisode`'s small public
    interface.  Keeping the helper structurally typed avoids importing the
    suite planner back into this low-level evidence module.
    """

    records_by_uuid = {record.episode_uuid: record for record in committed_records}
    family_reports: dict[str, Any] = {}
    overall = True
    for subfamily, raw in sorted(requirements.items()):
        requirement = dict(raw)
        members = [item for item in planned if item.case.subfamily == subfamily]
        expected_count = int(requirement.get("exact_member_count", 5))
        requested_field = str(requirement["requested_field"])
        response_metric = str(requirement["response_metric"])
        sample_count_metric = str(requirement["sample_count_metric"])
        minimum_samples = int(requirement.get("minimum_samples", 1))
        direction = str(requirement.get("direction", "increasing"))
        tolerance = float(requirement.get("monotonic_tolerance", 1e-9))
        minimum_endpoint_effect = float(
            requirement.get("minimum_endpoint_effect", 0.0)
        )
        maximum_target_error = requirement.get("maximum_absolute_target_error")
        problems: list[str] = []
        rows: list[dict[str, Any]] = []
        if len(members) != expected_count:
            problems.append(f"planned member count {len(members)} != {expected_count}")
        family_ids = {
            item.case.physics_counterfactual_family_id for item in members
        }
        if None in family_ids or len(family_ids) != 1:
            problems.append("members do not form one declared physics family")
        for item in members:
            record = records_by_uuid.get(item.episode_uuid)
            if record is None:
                problems.append(f"missing committed member {item.episode_uuid}")
                continue
            field = item.episode_plan.physics.get(requested_field)
            raw_target = field.get("value") if isinstance(field, Mapping) else None
            try:
                if isinstance(raw_target, Sequence) and not isinstance(
                    raw_target, (str, bytes)
                ):
                    target = math.sqrt(sum(float(value) ** 2 for value in raw_target))
                else:
                    target = float(raw_target)
            except (TypeError, ValueError):
                problems.append(
                    f"{item.episode_uuid} lacks numeric requested field {requested_field}"
                )
                continue
            persisted = record.extras.get("physics_qc")
            checks = (
                persisted.get("checks") if isinstance(persisted, Mapping) else None
            )
            if not isinstance(checks, Mapping):
                problems.append(f"{item.episode_uuid} lacks persisted physics evidence")
                continue
            response = checks.get(response_metric)
            sample_count = checks.get(sample_count_metric)
            if (
                isinstance(response, bool)
                or not isinstance(response, (int, float))
                or not math.isfinite(float(response))
            ):
                problems.append(f"{item.episode_uuid} lacks finite {response_metric}")
                continue
            if (
                isinstance(sample_count, bool)
                or not isinstance(sample_count, (int, float))
                or int(sample_count) < minimum_samples
            ):
                problems.append(
                    f"{item.episode_uuid} has insufficient {sample_count_metric}"
                )
                continue
            rows.append(
                {
                    "episode_uuid": item.episode_uuid,
                    "physics_variant": item.case.physics_variant,
                    "requested": target,
                    "measured": float(response),
                    "sample_count": int(sample_count),
                    "physics_qc_pass": bool(record.physics_qc_pass),
                }
            )
        rows.sort(key=lambda value: value["requested"])
        if len(rows) == expected_count:
            requested_values = [value["requested"] for value in rows]
            measured_values = [value["measured"] for value in rows]
            if any(
                right <= left + tolerance
                for left, right in zip(requested_values, requested_values[1:])
            ):
                problems.append("requested sweep values are not strictly increasing")
            signed_differences = [
                right - left
                for left, right in zip(measured_values, measured_values[1:])
            ]
            monotonic = (
                all(value > tolerance for value in signed_differences)
                if direction == "increasing"
                else all(value < -tolerance for value in signed_differences)
                if direction == "decreasing"
                else False
            )
            if not monotonic:
                problems.append(f"measured response is not strictly {direction}")
            endpoint_effect = abs(measured_values[-1] - measured_values[0])
            if endpoint_effect + tolerance < minimum_endpoint_effect:
                problems.append(
                    f"endpoint effect {endpoint_effect:.9g} < {minimum_endpoint_effect:.9g}"
                )
            if maximum_target_error is not None:
                maximum_error = max(
                    abs(value["measured"] - value["requested"]) for value in rows
                )
                if maximum_error > float(maximum_target_error) + tolerance:
                    problems.append(
                        f"maximum target error {maximum_error:.9g} > "
                        f"{float(maximum_target_error):.9g}"
                    )
        else:
            problems.append(f"measured member count {len(rows)} != {expected_count}")
        passed = not problems
        overall = overall and passed
        family_reports[subfamily] = {
            "passed": passed,
            "problems": problems,
            "measurements": rows,
        }
    return {"passed": overall and bool(requirements), "families": family_reports}


__all__ = ["sweep_acceptance_evidence"]
