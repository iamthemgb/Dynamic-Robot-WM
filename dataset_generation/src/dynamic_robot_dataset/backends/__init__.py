"""Capability-gated simulation backend construction.

The legacy native backend remains importable for diagnostics and regression
tests, but cannot be selected for review, pilot, or production generation.
"""

from pathlib import Path
from typing import Any, Mapping

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


def get_backend(
    name: str,
    *,
    purpose: str = "diagnostic",
    corpus_leaf_id: str | None = None,
    embodiment: str | None = None,
    task_variant: str | None = None,
    activation_report: Mapping[str, object] | str | Path | None = None,
    **kwargs: object,
) -> Any:
    """Construct a backend only when its registry capability admits the use.

    ``diagnostic`` is retained solely for old backend tests.  Review/preview
    callers must name the exact leaf, embodiment, and task variant.  Production
    additionally requires a released backend, which currently fails closed.
    """

    normalized = name.strip().lower().replace("-", "_")
    if purpose not in {"diagnostic", "preview", "review", "pilot", "production"}:
        raise ValueError(f"unknown backend purpose {purpose!r}")
    if normalized in {"native_mujoco", "mujoco_native", "mujoco"}:
        if purpose != "diagnostic":
            from ..common.corpus_registry import BackendNotReleasedError

            raise BackendNotReleasedError(
                "native_mujoco is permanently blocked and may only run diagnostics"
            )
        from .mujoco_native import NativeMuJoCoBackend

        return NativeMuJoCoBackend(**kwargs)
    canonical = {
        "source_mujoco": "source_mujoco",
        "source_genesis_fluid": "source_genesis_fluid",
        "source_mujoco_deformable": "source_mujoco_deformable",
    }.get(normalized)
    if canonical is None:
        raise KeyError(
            f"unknown simulation backend {name!r}; consult the backend capability registry"
        )

    from ..common.corpus_registry import (
        BackendNotReleasedError,
        UnsupportedScenarioError,
        load_backend_capability_registry,
        load_corpus_registry,
    )

    corpus = load_corpus_registry()
    registry = load_backend_capability_registry(corpus=corpus)
    capability = registry.by_name[canonical]
    support = None
    if purpose != "diagnostic":
        if not corpus_leaf_id or not embodiment or not task_variant:
            raise UnsupportedScenarioError(
                "review/production backend selection requires corpus_leaf_id, "
                "embodiment, and task_variant"
            )
        registry.resolve(
            canonical,
            corpus_leaf_id,
            embodiment,
            task_variant=task_variant,
            purpose=purpose,
            corpus=corpus,
            activation_report=activation_report,
        )
        support = capability.support_by_leaf[corpus_leaf_id]

    if canonical == "source_mujoco":
        from .source_mujoco import SourceMujocoBackend

        return SourceMujocoBackend(**kwargs)
    blockers = support.blockers if support is not None else capability.blockers
    raise BackendNotReleasedError(
        f"{canonical} has no executable owned backend yet: {', '.join(blockers)}"
    )


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
