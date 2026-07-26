"""Versioned physics-range validation and measured-effective calibration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .hashing import sha256_file, sha256_json


@dataclass(slots=True, frozen=True)
class CalibrationCheck:
    name: str
    passed: bool
    measured: Any
    tolerance: Any
    message: str = ""


@dataclass(slots=True)
class CalibrationReport:
    schema_version: str
    catalog_id: str
    catalog_hash: str
    passed: bool
    release_eligible: bool
    checks: list[CalibrationCheck]
    admitted_support: dict[str, dict[str, float]]
    missing_oracles: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "checks": [asdict(value) for value in self.checks],
        }


def load_range_catalog(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("Physics range catalog must contain a mapping")
    result = dict(value)
    if result.get("schema_version") != "dynamic-robot-physics-ranges/v1":
        raise ValueError("Unsupported physics range schema_version")
    fields = result.get("fields")
    if not isinstance(fields, Mapping) or not fields:
        raise ValueError("Physics range catalog has no fields")
    for name, raw in fields.items():
        if not isinstance(raw, Mapping) or not raw.get("unit"):
            raise ValueError(f"Physics field {name} requires a unit")
        candidate = raw.get("candidate") or {}
        if "minimum" in candidate and "maximum" in candidate:
            minimum, maximum = float(candidate["minimum"]), float(candidate["maximum"])
            nominal = float(raw["nominal"])
            if not minimum <= nominal <= maximum:
                raise ValueError(f"Physics field {name} nominal lies outside candidate support")
    return result


def free_fall_oracle(
    timestamps_s: Sequence[float],
    vertical_positions_m: Sequence[float],
    *,
    gravity_m_s2: float = -9.81,
) -> dict[str, float]:
    if len(timestamps_s) != len(vertical_positions_m) or len(timestamps_s) < 3:
        raise ValueError("Free-fall calibration requires at least three aligned samples")
    accelerations: list[float] = []
    for index in range(1, len(timestamps_s) - 1):
        left_dt = timestamps_s[index] - timestamps_s[index - 1]
        right_dt = timestamps_s[index + 1] - timestamps_s[index]
        if left_dt <= 0 or right_dt <= 0:
            raise ValueError("Free-fall timestamps must be strictly increasing")
        left_velocity = (vertical_positions_m[index] - vertical_positions_m[index - 1]) / left_dt
        right_velocity = (vertical_positions_m[index + 1] - vertical_positions_m[index]) / right_dt
        accelerations.append(2.0 * (right_velocity - left_velocity) / (left_dt + right_dt))
    rmse = math.sqrt(sum((value - gravity_m_s2) ** 2 for value in accelerations) / len(accelerations))
    return {"gravity_acceleration_rmse_m_s2": rmse, "measured_acceleration_m_s2": sum(accelerations) / len(accelerations)}


def bounce_oracle(preimpact_normal_velocity_m_s: float, postimpact_normal_velocity_m_s: float) -> dict[str, float]:
    if preimpact_normal_velocity_m_s >= 0 or postimpact_normal_velocity_m_s < 0:
        raise ValueError("Bounce calibration requires approaching pre-impact and separating post-impact velocities")
    effective = -postimpact_normal_velocity_m_s / preimpact_normal_velocity_m_s
    return {"measured_effective_restitution": effective, "normal_energy_ratio": effective * effective}


def sliding_oracle(
    timestamps_s: Sequence[float], speeds_m_s: Sequence[float], *, gravity_m_s2: float = 9.81
) -> dict[str, float]:
    if len(timestamps_s) != len(speeds_m_s) or len(timestamps_s) < 2:
        raise ValueError("Sliding calibration requires aligned speed samples")
    estimates = []
    for left_t, right_t, left_v, right_v in zip(
        timestamps_s, timestamps_s[1:], speeds_m_s, speeds_m_s[1:]
    ):
        if right_t <= left_t:
            raise ValueError("Sliding timestamps must be strictly increasing")
        estimates.append(max(0.0, (left_v - right_v) / (right_t - left_t) / gravity_m_s2))
    return {"measured_effective_dynamic_friction": sum(estimates) / len(estimates)}


def rolling_slip_oracle(
    linear_speeds_m_s: Sequence[float], angular_speeds_rad_s: Sequence[float], radius_m: float
) -> dict[str, float]:
    if len(linear_speeds_m_s) != len(angular_speeds_rad_s) or not linear_speeds_m_s:
        raise ValueError("Rolling calibration requires aligned linear/angular speeds")
    slip = [abs(linear - angular * radius_m) for linear, angular in zip(linear_speeds_m_s, angular_speeds_rad_s)]
    return {"mean_slip_speed_m_s": sum(slip) / len(slip), "maximum_slip_speed_m_s": max(slip)}


def _verified_source_artifacts(
    evidence: Mapping[str, Any],
) -> tuple[bool, dict[str, str], str]:
    artifacts = evidence.get("source_artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        return False, {}, "source_artifacts must be a non-empty list"
    verified: dict[str, str] = {}
    try:
        for artifact in artifacts:
            if not isinstance(artifact, Mapping):
                raise ValueError("source artifact entries must be mappings")
            role = str(artifact.get("role", "")).strip()
            path = Path(str(artifact.get("path", ""))).resolve(strict=True)
            expected = str(artifact.get("sha256", ""))
            if not role or role in verified:
                raise ValueError("source artifact roles must be unique and non-empty")
            if sha256_file(path) != expected:
                raise ValueError(f"source artifact hash mismatch for {role}")
            verified[role] = expected
    except (FileNotFoundError, OSError, ValueError) as error:
        return False, verified, str(error)
    required_roles = {"episodes", "qc_report"}
    missing = sorted(required_roles - set(verified))
    if missing:
        return False, verified, f"missing source artifact roles: {missing}"
    return True, verified, ""


def _recompute_oracle(name: str, raw: Any, minimum_trials: int) -> tuple[bool, Any, str]:
    if not isinstance(raw, Mapping):
        return False, None, "oracle evidence must be a mapping"
    measurements = raw.get("measurements")
    trials = measurements.get("trials") if isinstance(measurements, Mapping) else None
    if not isinstance(trials, list) or len(trials) < minimum_trials:
        return False, {"trial_count": 0 if not isinstance(trials, list) else len(trials)}, (
            f"at least {minimum_trials} raw trials are required"
        )
    recomputed: list[dict[str, float]] = []
    try:
        for trial in trials:
            if not isinstance(trial, Mapping):
                raise ValueError("raw calibration trials must be mappings")
            if name == "free_fall":
                value = free_fall_oracle(
                    trial["timestamps_s"],
                    trial["vertical_positions_m"],
                    gravity_m_s2=float(trial.get("gravity_m_s2", -9.81)),
                )
                if value["gravity_acceleration_rmse_m_s2"] > 0.25:
                    raise ValueError("free-fall acceleration RMSE exceeds 0.25 m/s^2")
            elif name == "bounce":
                value = bounce_oracle(
                    float(trial["preimpact_normal_velocity_m_s"]),
                    float(trial["postimpact_normal_velocity_m_s"]),
                )
                target = float(trial["requested_effective_restitution"])
                if (
                    abs(value["measured_effective_restitution"] - target) > 0.25
                    or value["normal_energy_ratio"] > 1.05
                ):
                    raise ValueError("measured bounce response fails target/energy bounds")
            elif name == "sliding_deceleration":
                value = sliding_oracle(
                    trial["timestamps_s"],
                    trial["speeds_m_s"],
                    gravity_m_s2=float(trial.get("gravity_m_s2", 9.81)),
                )
                target = float(trial["requested_effective_dynamic_friction"])
                if abs(value["measured_effective_dynamic_friction"] - target) > 0.15:
                    raise ValueError("measured sliding friction differs from its request")
            elif name == "rolling_slip":
                value = rolling_slip_oracle(
                    trial["linear_speeds_m_s"],
                    trial["angular_speeds_rad_s"],
                    float(trial["radius_m"]),
                )
                if value["maximum_slip_speed_m_s"] > 0.12:
                    raise ValueError("rolling slip exceeds 0.12 m/s")
            elif name == "contact_stability":
                value = {
                    "maximum_penetration_m": float(trial["maximum_penetration_m"]),
                    "maximum_contact_energy_ratio": float(
                        trial["maximum_contact_energy_ratio"]
                    ),
                    "nonfinite_state_count": float(trial["nonfinite_state_count"]),
                    "post_initialization_state_reset_count": float(
                        trial["post_initialization_state_reset_count"]
                    ),
                }
                if (
                    value["maximum_penetration_m"] > 0.01
                    or value["maximum_contact_energy_ratio"] > 1.05
                    or value["nonfinite_state_count"] != 0
                    or value["post_initialization_state_reset_count"] != 0
                ):
                    raise ValueError("contact stability trial violates a hard bound")
            else:
                raise ValueError(f"unsupported calibration oracle {name}")
            recomputed.append(value)
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        return False, {"trial_count": len(trials), "recomputed": recomputed}, str(error)
    return True, {"trial_count": len(trials), "recomputed": recomputed}, ""


def _verified_approval(
    evidence: Mapping[str, Any], catalog_hash: str
) -> tuple[bool, dict[str, Any], str]:
    approval_ref = evidence.get("approval_artifact")
    if not isinstance(approval_ref, Mapping):
        return False, {}, "approval_artifact is required"
    try:
        path = Path(str(approval_ref.get("path", ""))).resolve(strict=True)
        expected_hash = str(approval_ref.get("sha256", ""))
        if sha256_file(path) != expected_hash:
            raise ValueError("approval artifact hash mismatch")
        approval = json.loads(path.read_text(encoding="utf-8"))
        observation_hash = sha256_json(
            {key: value for key, value in evidence.items() if key != "approval_artifact"}
        )
        valid = (
            approval.get("schema_version")
            == "dynamic-robot-calibration-approval/v1"
            and approval.get("approved") is True
            and str(approval.get("reviewer", "")).strip() != ""
            and approval.get("catalog_hash") == catalog_hash
            and approval.get("observations_hash") == observation_hash
        )
        return valid, approval, "" if valid else "approval does not bind this catalog/evidence"
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as error:
        return False, {}, str(error)


def calibrate_physics_catalog(
    catalog: str | Path | Mapping[str, Any],
    *,
    observations: str | Path | Mapping[str, Any] | None = None,
) -> CalibrationReport:
    """Validate a catalog and admit support only when every oracle is measured.

    Calling this without observations deliberately returns a blocked report;
    loading plausible numeric ranges is not calibration evidence.
    """

    config = load_range_catalog(catalog) if isinstance(catalog, (str, Path)) else dict(catalog)
    if observations is None:
        evidence: dict[str, Any] = {}
    elif isinstance(observations, (str, Path)):
        evidence = json.loads(Path(observations).read_text(encoding="utf-8"))
    else:
        evidence = dict(observations)
    catalog_hash = sha256_json(config)
    schema_valid = (
        evidence.get("schema_version")
        == "dynamic-robot-native-calibration-observations/v1"
    )
    catalog_bound = evidence.get("catalog_hash") == catalog_hash
    source_valid, source_hashes, source_message = _verified_source_artifacts(evidence)
    oracle_evidence = evidence.get("oracles")
    oracle_evidence = oracle_evidence if isinstance(oracle_evidence, Mapping) else {}
    required = list((config.get("calibration") or {}).get("required_oracles", ()))
    missing = sorted(name for name in required if name not in oracle_evidence)
    checks: list[CalibrationCheck] = [
        CalibrationCheck(
            "observation_schema",
            schema_valid,
            evidence.get("schema_version"),
            "dynamic-robot-native-calibration-observations/v1",
        ),
        CalibrationCheck(
            "catalog_binding",
            catalog_bound,
            evidence.get("catalog_hash"),
            catalog_hash,
        ),
        CalibrationCheck(
            "source_artifacts_verified",
            source_valid,
            source_hashes,
            ["episodes", "qc_report"],
            source_message,
        ),
        CalibrationCheck(
            "all_required_oracles_present",
            not missing,
            sorted(oracle_evidence),
            sorted(required),
            "Candidate bounds remain development-only until native measurements exist",
        )
    ]
    minimum_trials = int((config.get("calibration") or {}).get("candidate_trials_per_value", 1))
    for name in required:
        valid, recomputed, message = _recompute_oracle(
            name, oracle_evidence.get(name), minimum_trials
        )
        checks.append(
            CalibrationCheck(
                f"oracle.{name}",
                valid,
                recomputed,
                {
                    "minimum_trial_count": minimum_trials,
                    "raw_measurements_recomputed": True,
                },
                message or "Raw native measurements were recomputed by this command",
            )
        )
    admitted: dict[str, dict[str, float]] = {}
    for name, raw in config["fields"].items():
        candidate = dict(raw.get("candidate") or {})
        if "minimum" not in candidate or "maximum" not in candidate:
            continue
        measured = evidence.get("admitted_support", {}).get(name)
        if not isinstance(measured, Mapping):
            continue
        minimum = float(measured["minimum"])
        maximum = float(measured["maximum"])
        candidate_minimum = float(candidate["minimum"])
        candidate_maximum = float(candidate["maximum"])
        passed = candidate_minimum <= minimum <= maximum <= candidate_maximum
        checks.append(
            CalibrationCheck(
                f"support.{name}",
                passed,
                {"minimum": minimum, "maximum": maximum},
                {"candidate_minimum": candidate_minimum, "candidate_maximum": candidate_maximum},
            )
        )
        if passed:
            width = maximum - minimum
            admitted[name] = {
                "minimum": minimum,
                "maximum": maximum,
                "train_id_minimum": minimum + 0.10 * width,
                "train_id_maximum": maximum - 0.10 * width,
                "ood_low_maximum": minimum + 0.10 * width,
                "ood_high_minimum": maximum - 0.10 * width,
            }
    required_support = set(
        str(value)
        for value in (config.get("calibration") or {}).get("required_admitted_support", ())
    )
    missing_support = sorted(required_support - set(admitted))
    checks.append(
        CalibrationCheck(
            "all_required_support_admitted",
            not missing_support,
            sorted(admitted),
            sorted(required_support),
        )
    )
    approval_valid, approval, approval_message = _verified_approval(
        evidence, catalog_hash
    )
    checks.append(
        CalibrationCheck(
            "release_approval_bound",
            approval_valid,
            approval,
            "dynamic-robot-calibration-approval/v1 bound to catalog and observations",
            approval_message,
        )
    )
    passed = not missing and all(check.passed for check in checks)
    return CalibrationReport(
        schema_version="dynamic-robot-calibration-report/v1",
        catalog_id=str(config["catalog_id"]),
        catalog_hash=catalog_hash,
        passed=passed,
        release_eligible=passed,
        checks=checks,
        admitted_support=admitted,
        missing_oracles=missing,
    )
