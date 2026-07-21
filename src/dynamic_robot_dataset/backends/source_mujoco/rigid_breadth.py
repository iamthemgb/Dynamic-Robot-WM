"""Compatibility exports for rigid-breadth scenario contracts.

Canonical ownership moved to the clearly labelled F2 scenario modules. New
code should import from :mod:\`dynamic_robot_dataset.scenarios\` directly.
"""

from ...scenarios.f2b_ramp_launch import SurfaceTransitionContract
from ...scenarios.f2e_multi_surface_rebound import OrderedContactContract
from ...scenarios.f2f_arbitrary_surface_bounce import (
    ARBITRARY_SURFACE_CATALOG_VERSION,
    ArbitrarySurfaceCandidate,
    RIGID_BREADTH_PROFILE_SCHEMA,
    RIGID_BREADTH_PROFILE_VERSION,
    SampledSurfaceContract,
    SurfaceAdmission,
    admitted_surface_catalog,
    catalog_sha256,
    sample_admitted_surface,
    sample_surface_candidate,
    sampled_surface_contract,
)

__all__ = [
    "ARBITRARY_SURFACE_CATALOG_VERSION",
    "ArbitrarySurfaceCandidate",
    "OrderedContactContract",
    "RIGID_BREADTH_PROFILE_SCHEMA",
    "RIGID_BREADTH_PROFILE_VERSION",
    "SampledSurfaceContract",
    "SurfaceAdmission",
    "SurfaceTransitionContract",
    "admitted_surface_catalog",
    "catalog_sha256",
    "sample_admitted_surface",
    "sample_surface_candidate",
    "sampled_surface_contract",
]
