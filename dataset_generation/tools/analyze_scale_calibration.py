#!/usr/bin/env python3
"""Correlate sampled initial states with strict QC verdicts for one block."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    root = Path(args.dataset).resolve(strict=True)

    plan = json.loads((root / ".run_plan.json").read_text(encoding="utf-8"))
    report = json.loads(
        (root / "qc" / "dataset_report.json").read_text(encoding="utf-8")
    )
    qc_by_uuid = {item["episode_uuid"]: item for item in report["episodes"]}

    rows = []
    for entry in plan["episodes"]:
        declaration = entry["declaration"]
        spec = declaration["source_scenario_spec"]
        contract = spec["initial_state"].get("initial_state_sampling_contract") or {}
        qc = qc_by_uuid[entry["episode_uuid"]]
        failures = " | ".join(qc.get("hard_failures") or [])
        rows.append(
            {
                "index": entry["episode_index"],
                "branch": declaration["intended_branch"],
                "embodiment": declaration["embodiment"],
                "passed": qc["passed"],
                "launch_height_m": contract.get("sampled_launch_height_m"),
                "flight_time_s": contract.get("sampled_flight_time_s"),
                "vz0": contract.get("sampled_initial_vertical_velocity_m_s"),
                "distance_m": contract.get("sampled_horizontal_distance_m"),
                "visibility_failure": "not visible" in failures
                or "cropped" in failures
                or "footprint" in failures,
                "physics_failure": "penetration" in failures
                or "physics_qc_pass" in failures,
            }
        )

    def start_z(row) -> float | None:
        if row["launch_height_m"] is None:
            return None
        return row["launch_height_m"]

    visible_pass = [r for r in rows if not r["visibility_failure"]]
    visible_fail = [r for r in rows if r["visibility_failure"]]
    print("episodes:", len(rows))
    print("qc passed:", sum(1 for r in rows if r["passed"]))
    print("visibility failures:", len(visible_fail))
    print("physics failures:", sum(1 for r in rows if r["physics_failure"]))
    heights_pass = sorted(
        r["launch_height_m"] for r in visible_pass if r["launch_height_m"] is not None
    )
    heights_fail = sorted(
        r["launch_height_m"] for r in visible_fail if r["launch_height_m"] is not None
    )
    if heights_pass:
        print(
            "launch height, visibility-clean: "
            f"min={heights_pass[0]:.3f} max={heights_pass[-1]:.3f}"
        )
    if heights_fail:
        print(
            "launch height, visibility-failed: "
            f"min={heights_fail[0]:.3f} max={heights_fail[-1]:.3f}"
        )
    times_pass = sorted(
        r["flight_time_s"] for r in visible_pass if r["flight_time_s"] is not None
    )
    times_fail = sorted(
        r["flight_time_s"] for r in visible_fail if r["flight_time_s"] is not None
    )
    if times_pass:
        print(
            "flight time, visibility-clean: "
            f"min={times_pass[0]:.3f} max={times_pass[-1]:.3f}"
        )
    if times_fail:
        print(
            "flight time, visibility-failed: "
            f"min={times_fail[0]:.3f} max={times_fail[-1]:.3f}"
        )
    def breakdown(name: str, key) -> None:
        clean: dict[str, int] = {}
        failed: dict[str, int] = {}
        for r in rows:
            bucket = failed if r["visibility_failure"] else clean
            value = key(r)
            bucket[value] = bucket.get(value, 0) + 1
        print(f"{name}: clean={clean} failed={failed}")

    breakdown("by embodiment", lambda r: r["embodiment"])
    breakdown("by branch", lambda r: r["branch"])
    for entry in plan["episodes"]:
        declaration = entry["declaration"]
        case = declaration["scale_case"]
        rows[entry["episode_index"]]["scene"] = case["scene_profile"]
        contract = declaration["source_scenario_spec"]["initial_state"].get(
            "initial_state_sampling_contract"
        ) or {}
        rows[entry["episode_index"]]["azimuth"] = contract.get(
            "sampled_incoming_azimuth_deg"
        )
    breakdown("by scene", lambda r: r.get("scene"))
    az_clean = sorted(
        round(r["azimuth"], 1)
        for r in rows
        if not r["visibility_failure"] and r.get("azimuth") is not None
    )
    az_failed = sorted(
        round(r["azimuth"], 1)
        for r in rows
        if r["visibility_failure"] and r.get("azimuth") is not None
    )
    print("azimuth clean:", az_clean)
    print("azimuth failed:", az_failed)

    physics_rows = [r for r in rows if r["physics_failure"]]
    for r in physics_rows:
        entry = plan["episodes"][r["index"]]
        declaration = entry["declaration"]
        contract = declaration["source_scenario_spec"]["initial_state"].get(
            "initial_state_sampling_contract"
        ) or {}
        target = contract.get("sampled_target_position_m") or [None, None, None]
        print(
            "physfail",
            f"idx={r['index']:3d}",
            declaration["embodiment"][:7],
            declaration["intended_branch"][:24],
            declaration["scale_case"]["scene_profile"][:18],
            f"ty={target[1]:+.4f}" if target[1] is not None else "ty=?",
            f"dist={contract.get('sampled_horizontal_distance_m', 0):.3f}",
            f"az={contract.get('sampled_incoming_azimuth_deg', 0):+7.1f}",
            f"vimp={9.81 * r['flight_time_s'] - (r['vz0'] or 0.0):.2f}",
        )
    if physics_rows:
        speeds = sorted(
            9.81 * r["flight_time_s"] - (r["vz0"] or 0.0)
            for r in physics_rows
            if r["flight_time_s"] is not None
        )
        by_embodiment: dict[str, int] = {}
        for r in physics_rows:
            by_embodiment[r["embodiment"]] = by_embodiment.get(r["embodiment"], 0) + 1
        print("physics-failure impact speeds:", [round(v, 2) for v in speeds])
        print("physics failures by embodiment:", by_embodiment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
