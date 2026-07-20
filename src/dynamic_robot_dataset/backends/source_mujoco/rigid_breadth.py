"""Versioned contracts for the rigid-breadth review scenarios.

This module owns only deterministic construction data.  It does not run a
simulator, retry an outcome, or mutate an object after initialization.  The
catalog is deliberately small: a candidate enters it only after the geometry
has an anchored support plan, a reachable interception envelope, task-volume
clearance, and a recorded timestep comparison policy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Mapping

import numpy as np

from ...common.hashing import sha256_json


RIGID_BREADTH_PROFILE_SCHEMA = "dynamic-robot-rigid-breadth-profile/v1"
RIGID_BREADTH_PROFILE_VERSION = "rigid-breadth-review-2026-07-v1"
ARBITRARY_SURFACE_CATALOG_VERSION = "rigid-arbitrary-surfaces/v1"


def _finite_vector(value: tuple[float, ...], size: int, label: str) -> None:
    if len(value) != size or any(not math.isfinite(float(item)) for item in value):
        raise ValueError(f"{label} must contain {size} finite values")


def _normal_from_euler(euler_rad: tuple[float, float, float]) -> tuple[float, float, float]:
    """Return a box's local +Z face normal for XYZ Euler angles.

    The admitted arbitrary planes currently vary only their pitch and yaw,
    but the complete expression keeps the persisted normal tied to the exact
    sampled transform rather than to a task-name convention.
    """

    roll, pitch, yaw = (float(value) for value in euler_rad)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    # Rz(yaw) @ Ry(pitch) @ Rx(roll) @ [0, 0, 1].
    value = np.asarray(
        (
            cy * sp * cr + sy * sr,
            sy * sp * cr - cy * sr,
            cp * cr,
        ),
        dtype=np.float64,
    )
    value /= np.linalg.norm(value)
    return tuple(float(item) for item in value)


@dataclass(frozen=True, slots=True)
class SurfaceTransitionContract:
    support_surface_id: str
    support_normal_world_xyz: tuple[float, float, float]
    minimum_support_contact_s: float = 0.08
    minimum_free_flight_s: float = 0.08
    require_ordered_termination: bool = True
    schema_version: str = "surface-to-free-flight/v1"

    def validate(self) -> None:
        if self.schema_version != "surface-to-free-flight/v1":
            raise ValueError("unsupported surface-transition contract")
        if not self.support_surface_id:
            raise ValueError("surface-transition contract lacks a stable surface ID")
        _finite_vector(self.support_normal_world_xyz, 3, "support normal")
        if abs(np.linalg.norm(self.support_normal_world_xyz) - 1.0) > 1e-6:
            raise ValueError("support normal must be normalized")
        if self.minimum_support_contact_s <= 0 or self.minimum_free_flight_s <= 0:
            raise ValueError("surface-transition durations must be positive")
        if not self.require_ordered_termination:
            raise ValueError("support termination must remain ordered")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True, slots=True)
class OrderedContactContract:
    ordered_surface_ids: tuple[str, ...]
    ordered_surface_normals_world_xyz: tuple[tuple[float, float, float], ...]
    minimum_separated_pre_post_samples: int = 2
    minimum_inter_contact_free_flight_s: float = 0.04
    reject_contact_chatter: bool = True
    schema_version: str = "ordered-surface-contacts/v1"

    def validate(self) -> None:
        if self.schema_version != "ordered-surface-contacts/v1":
            raise ValueError("unsupported ordered-contact contract")
        if len(self.ordered_surface_ids) < 2 or len(set(self.ordered_surface_ids)) != len(
            self.ordered_surface_ids
        ):
            raise ValueError("ordered contacts require at least two unique stable IDs")
        if len(self.ordered_surface_normals_world_xyz) != len(self.ordered_surface_ids):
            raise ValueError("ordered surface normals do not match the ID sequence")
        for normal in self.ordered_surface_normals_world_xyz:
            _finite_vector(normal, 3, "ordered surface normal")
            if abs(np.linalg.norm(normal) - 1.0) > 1e-6:
                raise ValueError("ordered surface normals must be normalized")
        if self.minimum_separated_pre_post_samples < 2:
            raise ValueError("ordered contacts require separated pre/post samples")
        if self.minimum_inter_contact_free_flight_s <= 0:
            raise ValueError("ordered contacts require a positive free-flight gap")
        if not self.reject_contact_chatter:
            raise ValueError("contact chatter rejection cannot be disabled")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SurfaceAdmission:
    grounded_supported: bool = False
    reachability_checked: bool = False
    swept_volume_clearance_checked: bool = False
    background_clearance_checked: bool = False
    calibrated_600_1200: bool = False

    def validate(self) -> None:
        if any(not isinstance(value, bool) for value in asdict(self).values()):
            raise ValueError("surface admission fields must be explicit booleans")

    @property
    def admitted(self) -> bool:
        self.validate()
        return all(asdict(self).values())


@dataclass(frozen=True, slots=True)
class ArbitrarySurfaceCandidate:
    candidate_id: str
    task_variant: str
    role: str
    position_m: tuple[float, float, float]
    half_size_m: tuple[float, float, float]
    euler_rad: tuple[float, float, float]
    contact_profile: str
    admission: SurfaceAdmission = SurfaceAdmission()
    admission_evidence_sha256: Mapping[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.candidate_id or self.task_variant not in {
            "random_plane_bounce",
            "random_barrier_bounce",
        }:
            raise ValueError("arbitrary surface candidate identity is invalid")
        expected_role = "table" if self.task_variant == "random_plane_bounce" else "wall"
        if self.role != expected_role:
            raise ValueError("arbitrary surface role differs from its task variant")
        _finite_vector(self.position_m, 3, "surface position")
        _finite_vector(self.half_size_m, 3, "surface half-size")
        _finite_vector(self.euler_rad, 3, "surface Euler transform")
        if any(value <= 0 for value in self.half_size_m):
            raise ValueError("surface half-size must be positive")
        if self.contact_profile not in {"rebound_pad", "rebound_wall"}:
            raise ValueError("arbitrary surface uses an uncalibrated contact profile")
        self.admission.validate()
        admitted_names = {
            name for name, admitted in asdict(self.admission).items() if admitted
        }
        if set(self.admission_evidence_sha256) != admitted_names or any(
            len(str(digest)) != 64
            or any(character not in "0123456789abcdef" for character in str(digest))
            for digest in self.admission_evidence_sha256.values()
        ):
            raise ValueError(
                "surface admission booleans require exact SHA-256 evidence bindings"
            )

    @property
    def normal_world_xyz(self) -> tuple[float, float, float]:
        if self.role == "table":
            return _normal_from_euler(self.euler_rad)
        # Barrier recipes approach the local -Y face.  Rz(yaw) @ [0,-1,0].
        yaw = float(self.euler_rad[2])
        return (math.sin(yaw), -math.cos(yaw), 0.0)


@dataclass(frozen=True, slots=True)
class SampledSurfaceContract:
    candidate_id: str
    task_variant: str
    role: str
    contact_profile: str
    source_seed: int
    position_m: tuple[float, float, float]
    euler_rad: tuple[float, float, float]
    normal_world_xyz: tuple[float, float, float]
    half_size_m: tuple[float, float, float]
    admission: Mapping[str, bool]
    admission_evidence_sha256: Mapping[str, str]
    catalog_sha256: str
    catalog_version: str = ARBITRARY_SURFACE_CATALOG_VERSION
    schema_version: str = "sampled-admitted-surface/v1"

    def validate(self) -> None:
        if self.schema_version != "sampled-admitted-surface/v1":
            raise ValueError("unsupported sampled-surface contract")
        if self.catalog_version != ARBITRARY_SURFACE_CATALOG_VERSION:
            raise ValueError("sampled surface uses an unknown admitted catalog")
        if (
            not self.candidate_id
            or self.task_variant not in {
                "random_plane_bounce",
                "random_barrier_bounce",
            }
            or self.role not in {"table", "wall"}
            or self.contact_profile not in {"rebound_pad", "rebound_wall"}
            or isinstance(self.source_seed, bool)
            or not (
            0 <= int(self.source_seed) < 2**64
            )
        ):
            raise ValueError("sampled surface identity/seed is invalid")
        if len(self.catalog_sha256) != 64 or any(
            value not in "0123456789abcdef" for value in self.catalog_sha256
        ):
            raise ValueError("sampled surface lacks a catalog content hash")
        _finite_vector(self.position_m, 3, "sampled surface position")
        _finite_vector(self.euler_rad, 3, "sampled surface transform")
        _finite_vector(self.normal_world_xyz, 3, "sampled surface normal")
        _finite_vector(self.half_size_m, 3, "sampled surface half-size")
        if abs(np.linalg.norm(self.normal_world_xyz) - 1.0) > 1e-6:
            raise ValueError("sampled surface normal must be normalized")
        required = set(asdict(SurfaceAdmission()))
        if set(self.admission) != required or any(
            not isinstance(self.admission[name], bool) for name in required
        ):
            raise ValueError("sampled surface admission evidence is malformed")
        admitted_names = {
            name for name, admitted in self.admission.items() if admitted
        }
        if set(self.admission_evidence_sha256) != admitted_names or any(
            len(str(digest)) != 64
            or any(character not in "0123456789abcdef" for character in str(digest))
            for digest in self.admission_evidence_sha256.values()
        ):
            raise ValueError(
                "sampled surface admission is not bound to calibration evidence"
            )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


# These transforms are conservative review candidates derived from the owned
# bounce-pad and rebound-wall envelopes.  Sampling selects a construction; it
# never samples until an intended outcome appears.
_ARBITRARY_SURFACE_CANDIDATES = (
    ArbitrarySurfaceCandidate(
        candidate_id="plane_low_170",
        task_variant="random_plane_bounce",
        role="table",
        position_m=(0.58, 0.0, 0.154),
        half_size_m=(0.12, 0.20, 0.016),
        euler_rad=(0.0, 0.0, 0.0),
        contact_profile="rebound_pad",
    ),
    ArbitrarySurfaceCandidate(
        candidate_id="plane_mid_285",
        task_variant="random_plane_bounce",
        role="table",
        position_m=(0.58, 0.0, 0.269),
        half_size_m=(0.14, 0.18, 0.016),
        euler_rad=(0.0, 0.0, 0.0),
        contact_profile="rebound_pad",
    ),
    ArbitrarySurfaceCandidate(
        candidate_id="plane_table_400",
        task_variant="random_plane_bounce",
        role="table",
        position_m=(0.58, 0.0, 0.384),
        half_size_m=(0.10, 0.22, 0.016),
        euler_rad=(0.0, 0.0, 0.0),
        contact_profile="rebound_pad",
    ),
    ArbitrarySurfaceCandidate(
        candidate_id="barrier_straight",
        task_variant="random_barrier_bounce",
        role="wall",
        position_m=(0.42, 0.32, 0.78),
        half_size_m=(0.20, 0.02, 0.78),
        euler_rad=(0.0, 0.0, 0.0),
        contact_profile="rebound_wall",
    ),
    ArbitrarySurfaceCandidate(
        candidate_id="barrier_yaw_neg_15",
        task_variant="random_barrier_bounce",
        role="wall",
        position_m=(0.42, 0.32, 0.78),
        half_size_m=(0.18, 0.02, 0.78),
        euler_rad=(0.0, 0.0, math.radians(-15.0)),
        contact_profile="rebound_wall",
    ),
    ArbitrarySurfaceCandidate(
        candidate_id="barrier_yaw_neg_25",
        task_variant="random_barrier_bounce",
        role="wall",
        position_m=(0.42, 0.32, 0.78),
        half_size_m=(0.18, 0.02, 0.78),
        euler_rad=(0.0, 0.0, math.radians(-25.0)),
        contact_profile="rebound_wall",
    ),
)


def admitted_surface_catalog() -> tuple[ArbitrarySurfaceCandidate, ...]:
    for candidate in _ARBITRARY_SURFACE_CANDIDATES:
        candidate.validate()
    return _ARBITRARY_SURFACE_CANDIDATES


def sample_surface_candidate(task_variant: str, *, source_seed: int) -> ArbitrarySurfaceCandidate:
    """Select one review candidate using only the declared physics RNG seed."""

    if isinstance(source_seed, bool) or not 0 <= int(source_seed) < 2**64:
        raise ValueError("surface sampler source_seed must be a uint64")
    candidates = tuple(
        item for item in admitted_surface_catalog() if item.task_variant == task_variant
    )
    if not candidates:
        raise ValueError(f"no admitted arbitrary surfaces for {task_variant!r}")
    # SeedSequence/PCG64 selection is stable and does not consume any camera,
    # asset, controller, or scene-construction stream.
    rng = np.random.Generator(np.random.PCG64(np.uint64(source_seed)))
    return candidates[int(rng.integers(0, len(candidates), endpoint=False))]


def sample_admitted_surface(task_variant: str, *, source_seed: int) -> ArbitrarySurfaceCandidate:
    """Select an admitted candidate, failing closed while calibration is pending."""

    candidate = sample_surface_candidate(task_variant, source_seed=source_seed)
    if not candidate.admission.admitted:
        raise ValueError(
            f"arbitrary surface {candidate.candidate_id} is not fully admitted"
        )
    return candidate


def sampled_surface_contract(
    candidate: ArbitrarySurfaceCandidate,
    *,
    source_seed: int,
) -> SampledSurfaceContract:
    candidate.validate()
    contract = SampledSurfaceContract(
        candidate_id=candidate.candidate_id,
        task_variant=candidate.task_variant,
        role=candidate.role,
        contact_profile=candidate.contact_profile,
        source_seed=int(source_seed),
        position_m=candidate.position_m,
        euler_rad=candidate.euler_rad,
        normal_world_xyz=candidate.normal_world_xyz,
        half_size_m=candidate.half_size_m,
        admission=asdict(candidate.admission),
        admission_evidence_sha256=dict(candidate.admission_evidence_sha256),
        catalog_sha256=catalog_sha256(),
    )
    contract.validate()
    return contract


def catalog_sha256() -> str:
    return sha256_json(
        {
            "schema_version": RIGID_BREADTH_PROFILE_SCHEMA,
            "profile_version": RIGID_BREADTH_PROFILE_VERSION,
            "catalog_version": ARBITRARY_SURFACE_CATALOG_VERSION,
            "candidates": [asdict(item) for item in admitted_surface_catalog()],
        }
    )


__all__ = [
    "ARBITRARY_SURFACE_CATALOG_VERSION",
    "ArbitrarySurfaceCandidate",
    "OrderedContactContract",
    "RIGID_BREADTH_PROFILE_SCHEMA",
    "RIGID_BREADTH_PROFILE_VERSION",
    "SampledSurfaceContract",
    "SurfaceTransitionContract",
    "admitted_surface_catalog",
    "catalog_sha256",
    "sample_admitted_surface",
    "sample_surface_candidate",
    "sampled_surface_contract",
]
