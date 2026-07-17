from __future__ import annotations

from dataclasses import replace

import pytest

from dynamic_robot_dataset.common.corpus_registry import (
    BackendCapabilityRegistry,
    BackendNotReleasedError,
    EXPECTED_BACKEND_BY_LEAF,
    EXPECTED_CORPUS_LEAVES,
    ExecutionState,
    PERMANENTLY_BLOCKED_BACKEND,
    RegistryValidationError,
    ReleaseState,
    UnsupportedScenarioError,
    load_backend_capability_registry,
    load_corpus_registry,
)


def test_corpus_registry_is_exact_authoritative_twenty_leaf_taxonomy() -> None:
    registry = load_corpus_registry()

    assert len(registry.leaves) == 20
    assert set(registry.by_id) == EXPECTED_CORPUS_LEAVES
    assert {leaf.corpus_id: leaf.backend for leaf in registry.leaves} == dict(
        EXPECTED_BACKEND_BY_LEAF
    )
    assert (registry.canonical_width, registry.canonical_height) == (832, 480)
    assert registry.canonical_video_hz == 30
    assert registry.canonical_cameras == ("main", "secondary")
    assert registry.fixed_review_rollouts == 6


def test_every_leaf_declares_embodiment_release_blockers_and_source_hash_metadata() -> None:
    registry = load_corpus_registry()

    for leaf in registry.leaves:
        assert leaf.supported_embodiments
        assert leaf.task_variants
        assert leaf.evaluator
        assert leaf.release_state is ReleaseState.BLOCKED
        assert leaf.blockers
        assert "source_hashes" in leaf.required_metadata
    assert registry.resolve("D1").supported_embodiments == ("franka_hand",)
    assert registry.resolve("D2").supported_embodiments == ("franka_hand",)
    assert registry.resolve("F3c").backend == "source_genesis_fluid"


def test_backend_capabilities_cover_every_leaf_once_and_retire_native_backend() -> None:
    corpus = load_corpus_registry()
    registry = load_backend_capability_registry(corpus=corpus)

    support = {
        item.corpus_id: backend.name
        for backend in registry.backends
        for item in backend.support
    }
    assert support == dict(EXPECTED_BACKEND_BY_LEAF)
    native = registry.by_name[PERMANENTLY_BLOCKED_BACKEND]
    assert native.release_state is ReleaseState.PERMANENTLY_BLOCKED
    assert native.support == ()
    assert native.source_hashes


def test_review_execution_is_per_leaf_and_does_not_release_production() -> None:
    corpus = load_corpus_registry()
    registry = load_backend_capability_registry(corpus=corpus)
    expected_review = {
        "P0a",
        "P0b",
        "P0c",
        "P0d",
        "F1a",
            "F1b",
            "F1c",
            "F1d",
        }

    review_support = {
        support.corpus_id
        for backend in registry.backends
        for support in backend.support
        if support.execution_state is ExecutionState.REVIEW
    }
    assert review_support == expected_review
    assert all(
        leaf.release_state is ReleaseState.BLOCKED for leaf in corpus.leaves
    )
    assert all(
        not backend.release_state.allows_production for backend in registry.backends
    )
    for backend in registry.backends:
        for support in backend.support:
            leaf = corpus.resolve(support.corpus_id)
            if support.execution_state is ExecutionState.REVIEW:
                assert support.implemented_task_variants == leaf.task_variants
                assert support.blockers == ()
            else:
                assert support.blockers


def test_source_mujoco_capabilities_match_the_owned_compiler_dispatch() -> None:
    from dynamic_robot_dataset.backends.source_mujoco.compiler import (
        IMPLEMENTED_REVIEW_VARIANTS,
    )

    corpus = load_corpus_registry()
    registry = load_backend_capability_registry(corpus=corpus)
    source = registry.by_name["source_mujoco"]

    for support in source.support:
        assert support.implemented_task_variants == IMPLEMENTED_REVIEW_VARIANTS.get(
            support.corpus_id, ()
        )


def test_unsupported_backend_embodiment_and_task_fail_closed() -> None:
    corpus = load_corpus_registry()
    backends = load_backend_capability_registry(corpus=corpus)

    with pytest.raises(UnsupportedScenarioError, match="unknown corpus leaf"):
        corpus.resolve("F9z")
    with pytest.raises(UnsupportedScenarioError, match="does not support embodiment"):
        corpus.resolve("D1", embodiment="robotiq_2f85_thick_pad")
    with pytest.raises(UnsupportedScenarioError, match="does not support task variant"):
        corpus.resolve("F1a", task_variant="scripted_magic_catch")
    with pytest.raises(UnsupportedScenarioError, match="does not support corpus leaf"):
        backends.resolve("source_genesis_fluid", "F1a", "franka_hand")
    with pytest.raises(UnsupportedScenarioError, match="unknown backend"):
        backends.resolve("implicit_fallback", "F1a", "franka_hand")


def test_blocked_leaf_and_backend_cannot_be_requested_for_production() -> None:
    corpus = load_corpus_registry()
    backends = load_backend_capability_registry(corpus=corpus)

    with pytest.raises(BackendNotReleasedError, match="F1a is blocked"):
        corpus.resolve("F1a", require_released=True)
    with pytest.raises(BackendNotReleasedError, match="source_mujoco is blocked"):
        backends.resolve(
            "source_mujoco", "F1a", "franka_hand", require_released=True
        )


def test_capability_registry_rejects_backend_taxonomy_drift() -> None:
    corpus = load_corpus_registry()
    registry = load_backend_capability_registry(corpus=corpus)
    mujoco = registry.by_name["source_mujoco"]
    wrong_support = replace(mujoco.support[0], family="silent_fallback_family")
    wrong_mujoco = replace(mujoco, support=(wrong_support, *mujoco.support[1:]))
    mutated = BackendCapabilityRegistry(
        registry_id=registry.registry_id,
        backends=tuple(
            wrong_mujoco if item.name == mujoco.name else item
            for item in registry.backends
        ),
    )

    with pytest.raises(RegistryValidationError, match="family/subfamily mismatch"):
        mutated.validate(corpus)


def test_review_execution_rejects_partial_variant_coverage() -> None:
    corpus = load_corpus_registry()
    registry = load_backend_capability_registry(corpus=corpus)
    mujoco = registry.by_name["source_mujoco"]
    direct = mujoco.support_by_leaf["F2a"]
    incomplete_review = replace(
        direct,
        execution_state=ExecutionState.REVIEW,
        blockers=(),
    )
    mutated_mujoco = replace(
        mujoco,
        support=tuple(
            incomplete_review if item.corpus_id == "F2a" else item
            for item in mujoco.support
        ),
    )
    mutated = BackendCapabilityRegistry(
        registry_id=registry.registry_id,
        backends=tuple(
            mutated_mujoco if item.name == mujoco.name else item
            for item in registry.backends
        ),
    )

    with pytest.raises(RegistryValidationError, match="must implement every declared"):
        mutated.validate(corpus)


def test_released_backend_must_have_source_hashes() -> None:
    corpus = load_corpus_registry()
    registry = load_backend_capability_registry(corpus=corpus)
    original = registry.by_name["source_mujoco"]
    invalid = replace(
        original,
        release_state=ReleaseState.RELEASED,
        source_hashes={},
        blockers=(),
    )

    with pytest.raises(RegistryValidationError, match="requires pinned source hashes"):
        invalid.validate()
