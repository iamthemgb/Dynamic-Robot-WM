"""Read-only discovery CLI for canonical scenario modules."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from .registry import list_scenario_definitions, load_scenario_definition


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m dynamic_robot_dataset.scenarios")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="list all canonical corpus scenario modules")
    show = commands.add_parser("show", help="show one scenario contract")
    show.add_argument("leaf_id")
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
    payload = definition.to_dict()
    payload["commands"] = {
        "fixed_review": (
            "dynamic-robot-dataset review-suite --leaf-id "
            f"{definition.leaf_id} --execute ..."
            if definition.implemented
            else None
        ),
        "inspect": f"python -m dynamic_robot_dataset.scenarios show {definition.leaf_id}",
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
