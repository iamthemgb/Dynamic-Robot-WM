from __future__ import annotations

from dataclasses import replace

import pytest

from dynamic_robot_dataset.common.legacy_quarantine import (
    LEGACY_ASSISTED_NAMESPACE,
    load_legacy_quarantine_policy,
)


def test_collaborator_conversion_is_pinned_to_nonrelease_legacy_namespace() -> None:
    policy = load_legacy_quarantine_policy()
    assert policy.namespace == LEGACY_ASSISTED_NAMESPACE
    assert policy.read_only
    assert not policy.training_eligible
    assert not policy.release_eligible
    assert not policy.counts_toward_generated_hours
    assert not policy.may_satisfy_review_or_scale_gate
    assert not policy.copy_into_canonical_namespace
    assert len(policy.collections) == 1
    collection = policy.collections[0]
    assert collection.corpus_leaf_id == "F1a"
    assert collection.episode_count == 1500
    assert collection.default_release_tier == "assisted_contact"
    assert "post_capture_object_state_rewrite" in collection.prohibited_mechanisms_observed


def test_legacy_quarantine_cannot_be_made_training_eligible() -> None:
    policy = load_legacy_quarantine_policy()
    with pytest.raises(ValueError, match="cannot enable"):
        replace(policy, training_eligible=True).validate()
