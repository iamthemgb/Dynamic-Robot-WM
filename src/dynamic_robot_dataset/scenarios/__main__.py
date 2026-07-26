"""Read-only discovery CLI for canonical scenario modules."""

from __future__ import annotations

import argparse
import json
import math
from typing import Sequence

from ..common.hashing import stable_uint64
from .registry import list_scenario_definitions, load_scenario_definition
from .types import ScenarioBuildContext


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m dynamic_robot_dataset.scenarios")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="list all canonical corpus scenario modules")
    show = commands.add_parser("show", help="show one scenario contract")
    show.add_argument("leaf_id")
    sample = commands.add_parser(
        "sample",
        help="sample deterministic F1d/F2a projectile initial states without simulation",
    )
    sample.add_argument("leaf_id", choices=("F1d", "F2a"))
    sample.add_argument("--count", type=int, default=6)
    sample.add_argument("--seed", type=int, default=0)
    sample.add_argument("--embodiment", default="franka_hand")
    sample.add_argument("--variant")
    sample.add_argument("--branch-role", default="nominal_success")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "list":
        print("LEAF\tSUBFAMILY\tBACKEND\tIMPLEMENTED\tRELEASE\tMODULE")
        for definition in list_scenario_definitions():
            print(
                "\t".join(
                    (
                        definition.leaf_id,
                        definition.subfamily,
                        definition.backend,
                        "yes" if definition.implemented else "no",
                        definition.release_state,
                        definition.module_path,
                    )
                )
            )
        return 0
    definition = load_scenario_definition(args.leaf_id)
    if args.command == "sample":
        if args.count < 1 or args.count > 10_000:
            raise ValueError("sample count must be between 1 and 10000")
        variant = args.variant or definition.variants[0]
        rows = []
        for index in range(args.count):
            identity = {
                "leaf_id": definition.leaf_id,
                "root_seed": args.seed,
                "sample_index": index,
            }
            initial_state_seed = stable_uint64(
                identity, namespace="projectile-initial-state-preview/v1"
            )
            physics_seed = stable_uint64(
                identity, namespace="projectile-physics-preview/v1"
            )
            recipe = definition.build(
                ScenarioBuildContext(
                    leaf_id=definition.leaf_id,
                    task_variant=variant,
                    embodiment=args.embodiment,
                    branch_role=args.branch_role,
                    seed=initial_state_seed,
                    physics_seed=physics_seed,
                    tabletop_height_m=0.0,
                    initial_state_mode="sampled_preview",
                )
            )
            velocity = recipe["object_initial_linear_velocity_m_s"]
            rows.append(
                {
                    "sample_index": index,
                    "initial_state_seed": initial_state_seed,
                    "physics_seed": physics_seed,
                    "leaf_id": definition.leaf_id,
                    "variant": variant,
                    "embodiment": args.embodiment,
                    "branch_role": args.branch_role,
                    "object_radius_m": recipe["object_radius_m"],
                    "object_mass_kg": recipe["object_mass_kg"],
                    "object_initial_position_m": recipe[
                        "object_initial_position_m"
                    ],
                    "object_initial_linear_velocity_m_s": velocity,
                    "object_initial_speed_m_s": math.sqrt(
                        sum(float(value) ** 2 for value in velocity)
                    ),
                    "object_initial_angular_velocity_rad_s": recipe[
                        "object_initial_angular_velocity_rad_s"
                    ],
                    "physical_target_position_m": recipe[
                        "physical_target_position_m"
                    ],
                    "ballistic_event_time_s": recipe["ballistic_event_time_s"],
                    "initial_state_sampling_contract": recipe[
                        "initial_state_sampling_contract"
                    ],
                }
            )
        print(
            json.dumps(
                {
                    "mode": "sampled_preview",
                    "training_eligible": False,
                    "leaf_id": definition.leaf_id,
                    "root_seed": args.seed,
                    "sample_count": len(rows),
                    "samples": rows,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    payload = definition.to_dict()
    payload["commands"] = {
        "fixed_review": (
            "dynamic-robot-dataset review-suite --leaf-id "
            f"{definition.leaf_id} --execute ..."
            if definition.implemented
            else None
        ),
        "inspect": f"python -m dynamic_robot_dataset.scenarios show {definition.leaf_id}",
        "sample_initial_states": (
            f"python -m dynamic_robot_dataset.scenarios sample {definition.leaf_id}"
            if definition.leaf_id in {"F1d", "F2a"}
            else None
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
