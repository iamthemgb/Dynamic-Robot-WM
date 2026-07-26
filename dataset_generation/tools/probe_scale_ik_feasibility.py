#!/usr/bin/env python3
"""Probe IK feasibility of sampled scale cases for one leaf on CPU.

Mints scale cases, runs the same per-case preparation the block planner
uses, and reports each case's sampled ballistic parameters together with
the IK grasp-center residual (or the failure message).  Use this to bound
a leaf's scale envelope on measured feasibility instead of guesswork.
"""

from __future__ import annotations

import argparse
import json
import math
import re

from dynamic_robot_dataset.common.scale_suite import mint_scale_cases
from dynamic_robot_dataset.common.scale_execution import (
    prepare_source_scale_declaration,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leaf", required=True)
    parser.add_argument("--count", type=int, default=60)
    parser.add_argument("--episode-start", type=int, default=18000000)
    args = parser.parse_args()

    cases = mint_scale_cases(
        args.leaf,
        episode_start=args.episode_start,
        count=args.count,
        scale_suite_id=f"ik-probe-{args.leaf}",
    )
    failures = 0
    for case in cases:
        try:
            declaration = prepare_source_scale_declaration(
                case, episode_index=case.episode_index
            )
            spec = declaration["source_scenario_spec"]
            contract = spec["initial_state"]["initial_state_sampling_contract"]
            ip = contract["applied_initial_position_m"]
            tp = contract["applied_physical_target_position_m"]
            dx, dy = ip[0] - tp[0], ip[1] - tp[1]
            print(
                json.dumps(
                    {
                        "case": case.case_id,
                        "variant": case.task_variant,
                        "embodiment": case.embodiment,
                        "branch": case.branch_role,
                        "ok": True,
                        "target": [round(v, 4) for v in tp],
                        "horiz_dist": round(math.hypot(dx, dy), 4),
                        "azimuth_deg": round(math.degrees(math.atan2(dy, dx)), 1),
                    }
                )
            )
        except Exception as error:  # noqa: BLE001 - report and continue
            failures += 1
            text = str(error)
            match = re.search(r"grasp-center error=([0-9.]+)", text)
            print(
                json.dumps(
                    {
                        "case": case.case_id,
                        "variant": case.task_variant,
                        "embodiment": case.embodiment,
                        "branch": case.branch_role,
                        "ok": False,
                        "ik_error_m": float(match.group(1)) if match else None,
                        "message": text.splitlines()[-1][:200],
                    }
                )
            )
    print(f"# {args.leaf}: {len(cases) - failures}/{len(cases)} prepared")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
