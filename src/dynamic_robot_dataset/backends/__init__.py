"""Simulation backend registry and stable public types."""

from .base import (
    BackendRunResult,
    CameraSpec,
    ControllerPhase,
    InitialStateSpec,
    IntendedBranch,
    NativeBackendError,
    NativeEpisodePlan,
    NativeFamily,
    PhysicsRangeProvenance,
    RigidObjectSpec,
    RigidScenario,
    RigidShape,
    ScenarioSpec,
    SimulationBackend,
    ToolKind,
    ToolSpec,
)


def get_backend(name: str, **kwargs: object) -> SimulationBackend:
    """Construct a backend without importing optional dependencies eagerly."""

    normalized = name.strip().lower().replace("-", "_")
    if normalized in {"native_mujoco", "mujoco_native", "mujoco"}:
        from .mujoco_native import NativeMuJoCoBackend

        return NativeMuJoCoBackend(**kwargs)
    raise KeyError(f"unknown simulation backend {name!r}; expected native_mujoco")


__all__ = [
    "BackendRunResult",
    "CameraSpec",
    "ControllerPhase",
    "InitialStateSpec",
    "IntendedBranch",
    "NativeBackendError",
    "NativeEpisodePlan",
    "NativeFamily",
    "PhysicsRangeProvenance",
    "RigidObjectSpec",
    "RigidScenario",
    "RigidShape",
    "ScenarioSpec",
    "SimulationBackend",
    "ToolKind",
    "ToolSpec",
    "get_backend",
]
