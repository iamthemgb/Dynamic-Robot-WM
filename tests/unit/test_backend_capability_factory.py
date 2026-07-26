from __future__ import annotations

from unittest.mock import patch

import pytest

from dynamic_robot_dataset.backends import get_backend
from dynamic_robot_dataset.common.corpus_registry import (
    BackendNotReleasedError,
    UnsupportedScenarioError,
)


def test_native_backend_is_diagnostic_only() -> None:
    with pytest.raises(BackendNotReleasedError, match="permanently blocked"):
        get_backend("native_mujoco", purpose="review")


def test_review_backend_requires_exact_supported_tuple() -> None:
    with pytest.raises(UnsupportedScenarioError, match="requires corpus_leaf_id"):
        get_backend("source_mujoco", purpose="review")
    with pytest.raises(BackendNotReleasedError, match="blocked for review"):
        get_backend(
            "source_mujoco",
            purpose="review",
            corpus_leaf_id="F2b",
            embodiment="franka_hand",
            task_variant="ramp_catch",
        )


def test_review_capability_constructs_only_owned_source_backend() -> None:
    sentinel = object()
    with patch(
        "dynamic_robot_dataset.backends.source_mujoco.SourceMujocoBackend",
        return_value=sentinel,
    ):
        assert get_backend(
            "source_mujoco",
            purpose="review",
            corpus_leaf_id="P0a",
            embodiment="no_robot",
            task_variant="nominal_freefall",
        ) is sentinel


def test_unimplemented_validated_backend_stays_blocked() -> None:
    with pytest.raises(BackendNotReleasedError, match="no executable owned backend"):
        get_backend(
            "source_genesis_fluid",
            purpose="diagnostic",
        )
