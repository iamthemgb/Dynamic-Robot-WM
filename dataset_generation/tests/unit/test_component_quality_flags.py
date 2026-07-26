from __future__ import annotations

from types import SimpleNamespace

from dynamic_robot_dataset.cli import _component_quality_flags


def _result(*, native: bool, production_eligible: bool):
    return SimpleNamespace(
        simulator={
            "native_mujoco": native,
            "production_eligible": production_eligible,
        }
    )


def test_native_aggregate_release_failure_keeps_only_real_blockers() -> None:
    flags = _component_quality_flags(
        _result(native=True, production_eligible=False),
        {
            "production_eligible": False,
            "renderer": "mujoco.Renderer",
            "backend_provenance": {
                "backend": "native_mujoco",
                "renderer": "mujoco.Renderer",
                "production_eligible": False,
            },
            "videos": {
                "observation.images.main": [object()],
                "observation.images.secondary": [object()],
            },
            "quality_flags": [
                "physics_range_uncalibrated",
                "tool_calibration_unverified",
                "visual_style_unvalidated",
            ],
        },
    )

    assert flags == [
        "physics_range_uncalibrated",
        "tool_calibration_unverified",
        "visual_style_unvalidated",
    ]


def test_native_missing_integrated_render_evidence_gets_only_renderer_flag() -> None:
    flags = _component_quality_flags(
        _result(native=True, production_eligible=False),
        {
            "production_eligible": False,
            "renderer": "mujoco.Renderer",
            "videos": {},
            "quality_flags": ["render_disabled"],
        },
    )

    assert flags == ["render_disabled", "renderer_not_production_verified"]


def test_diagnostic_payload_retains_conservative_component_flags() -> None:
    flags = _component_quality_flags(
        _result(native=False, production_eligible=False),
        {
            "production_eligible": False,
            "renderer": "diagnostic.state_renderer",
            "videos": {"observation.images.main": [object()]},
            "quality_flags": ["diagnostic_only"],
        },
    )

    assert flags == [
        "diagnostic_only",
        "non_production_family_adapter",
        "renderer_not_production_verified",
    ]


def test_admitted_non_native_components_do_not_receive_generic_flags() -> None:
    flags = _component_quality_flags(
        _result(native=False, production_eligible=True),
        {
            "production_eligible": True,
            "renderer": "verified.external.Renderer",
            "videos": {"observation.images.main": [object()]},
            "quality_flags": [],
        },
    )

    assert flags == []
