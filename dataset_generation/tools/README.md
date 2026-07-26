# Developer tools

The supported generation interface is `dynamic-robot-dataset`, not an
individual script in this directory.

Maintained helpers include catalog, split, validation, provenance, contact
sheet, and fixed-case inspection utilities. The two `generate_*_examples.py`
programs are historical, non-release diagnostic renderers retained to
reproduce migration evidence; they are not canonical scenario generators.

Canonical scenario behavior lives in `src/dynamic_robot_dataset/scenarios/`,
while shared simulation and writing code lives under `backends/` and `common/`.
