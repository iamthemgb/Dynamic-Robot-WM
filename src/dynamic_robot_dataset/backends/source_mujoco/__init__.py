"""Owned, fail-closed rigid source backend for fixed review rollouts."""

from .backend import IKDiagnostics, SourceMujocoBackend, SourceMujocoRunResult
from .compiler import (
    IMPLEMENTED_REVIEW_VARIANTS,
    PhysicalSurface,
    SourceMujocoCompiledScenario,
    SourceMujocoUnsupported,
    compile_review_case,
)
from .profiles import (
    RIGID_REVIEW_PROFILE,
    exact_frame_schedule,
    minimum_jerk_fraction,
    timestep_comparison_failures,
)
from .provenance import (
    PINNED_SOURCE_MANIFEST_SHA256,
    SourceDependencyError,
    resolve_robocasa_dependency,
    resolve_source_dependency,
)
from .source_spec import prepare_review_case

__all__ = [
    "IKDiagnostics",
    "IMPLEMENTED_REVIEW_VARIANTS",
    "PINNED_SOURCE_MANIFEST_SHA256",
    "PhysicalSurface",
    "RIGID_REVIEW_PROFILE",
    "SourceDependencyError",
    "SourceMujocoBackend",
    "SourceMujocoCompiledScenario",
    "SourceMujocoRunResult",
    "SourceMujocoUnsupported",
    "compile_review_case",
    "exact_frame_schedule",
    "minimum_jerk_fraction",
    "prepare_review_case",
    "resolve_robocasa_dependency",
    "resolve_source_dependency",
    "timestep_comparison_failures",
]
