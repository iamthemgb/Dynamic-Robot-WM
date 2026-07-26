from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from dynamic_robot_dataset.common.corpus_registry import (
    BackendNotReleasedError,
    ReleaseState,
    load_backend_capability_registry,
    load_corpus_registry,
)


def _registries(*, leaf_state: ReleaseState, backend_released: bool):
    corpus = load_corpus_registry()
    leaves = tuple(
        replace(
            leaf,
            release_state=leaf_state,
            blockers=() if leaf_state is ReleaseState.RELEASED else ("activation_pending",),
        )
        if leaf.corpus_id == "F1a"
        else leaf
        for leaf in corpus.leaves
    )
    corpus = replace(corpus, leaves=leaves)
    backends = load_backend_capability_registry(corpus=load_corpus_registry())
    capabilities = tuple(
        replace(
            backend,
            release_state=(
                ReleaseState.RELEASED if backend_released else ReleaseState.BLOCKED
            ),
            blockers=() if backend_released else ("backend_acceptance_pending",),
        )
        if backend.name == "source_mujoco"
        else backend
        for backend in backends.backends
    )
    backends = replace(backends, backends=capabilities)
    corpus.validate()
    backends.validate(corpus)
    return corpus, backends


def _resolve(corpus, backends, purpose: str, *, activation_report=None):
    return backends.resolve(
        "source_mujoco",
        "F1a",
        "franka_hand",
        task_variant="catch_retain",
        purpose=purpose,
        corpus=corpus,
        activation_report=activation_report,
    )


def test_review_requires_implemented_tuple_but_not_release_or_acceptance() -> None:
    corpus, backends = _registries(
        leaf_state=ReleaseState.BLOCKED,
        backend_released=False,
    )
    assert _resolve(corpus, backends, "review").name == "source_mujoco"

    source = backends.by_name["source_mujoco"]
    support = source.support_by_leaf["F1a"]
    blocked_support = replace(
        support,
        implemented_task_variants=(),
        execution_state=support.execution_state.BLOCKED,
        blockers=("implementation_pending",),
    )
    source = replace(
        source,
        support=tuple(
            blocked_support if value.corpus_id == "F1a" else value
            for value in source.support
        ),
    )
    backends = replace(
        backends,
        backends=tuple(
            source if value.name == "source_mujoco" else value
            for value in backends.backends
        ),
    )
    with pytest.raises(BackendNotReleasedError, match="task_variant_not_implemented"):
        _resolve(corpus, backends, "review")

    implemented_but_blocked = replace(
        blocked_support,
        implemented_task_variants=("catch_retain", "catch_miss"),
    )
    source = replace(
        source,
        support=tuple(
            implemented_but_blocked if value.corpus_id == "F1a" else value
            for value in source.support
        ),
    )
    backends = replace(
        backends,
        backends=tuple(
            source if value.name == "source_mujoco" else value
            for value in backends.backends
        ),
    )
    assert _resolve(corpus, backends, "review").name == "source_mujoco"


def test_pilot_requires_released_backend_and_pilot_or_released_leaf(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    blocked_leaf, released_backend = _registries(
        leaf_state=ReleaseState.BLOCKED,
        backend_released=True,
    )
    with pytest.raises(BackendNotReleasedError, match="cannot run a pilot"):
        _resolve(blocked_leaf, released_backend, "pilot")

    pilot_leaf, blocked_backend = _registries(
        leaf_state=ReleaseState.PILOT,
        backend_released=False,
    )
    with pytest.raises(BackendNotReleasedError, match="must be released for pilot"):
        _resolve(pilot_leaf, blocked_backend, "pilot")

    pilot_leaf, released_backend = _registries(
        leaf_state=ReleaseState.PILOT,
        backend_released=True,
    )
    with pytest.raises(BackendNotReleasedError, match="lacks a hash-bound"):
        _resolve(pilot_leaf, released_backend, "pilot")
    publication = tmp_path / "review--0123456789abcdef"
    publication.mkdir()
    activation_path = publication / "activation_report.json"
    activation_path.write_text("{}", encoding="utf-8")
    from dynamic_robot_dataset.common import review_finalize

    monkeypatch.setattr(
        review_finalize,
        "validate_external_review_publication",
        lambda value, *, corpus_id: SimpleNamespace(report_sha256="a" * 64),
    )
    assert _resolve(
        pilot_leaf,
        released_backend,
        "pilot",
        activation_report=activation_path,
    ).name == "source_mujoco"

    with pytest.raises(BackendNotReleasedError, match="requires the path"):
        _resolve(
            pilot_leaf,
            released_backend,
            "pilot",
            activation_report={"report_sha256": "a" * 64},
        )


def test_production_requires_leaf_and_backend_both_released() -> None:
    pilot_leaf, released_backend = _registries(
        leaf_state=ReleaseState.PILOT,
        backend_released=True,
    )
    with pytest.raises(BackendNotReleasedError, match="F1a is pilot"):
        _resolve(pilot_leaf, released_backend, "production")

    released_leaf, released_backend = _registries(
        leaf_state=ReleaseState.RELEASED,
        backend_released=True,
    )
    assert _resolve(released_leaf, released_backend, "production").name == (
        "source_mujoco"
    )


def test_preview_is_an_alias_for_review_and_unknown_purpose_fails() -> None:
    corpus, backends = _registries(
        leaf_state=ReleaseState.BLOCKED,
        backend_released=False,
    )
    assert _resolve(corpus, backends, "preview").name == "source_mujoco"
    with pytest.raises(ValueError, match="unknown generation purpose"):
        _resolve(corpus, backends, "training")
