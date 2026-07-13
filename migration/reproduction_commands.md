# Reproduction commands

The accepted diagnostic smoke artifacts were generated from commit
`ef1b3c927dce4297441a268ba2cd7e7a97dfb4dc`. Metadata timestamps and the random
dataset UUID naturally change on a rerun; task plans, trajectories, labels,
media content, splits, and schema structure are deterministic for the recorded
seed and environment.

## Environment and tests

```bash
cd /gpfs/radev/project/sous/zl664/dataset_generation
uv sync --frozen --extra mujoco --extra deformable --extra video --extra test
PYTHONDONTWRITEBYTECODE=1 uv run --frozen pytest -q -p no:cacheprovider
```

The accepted run reported 34 passing tests. The environment digest stored in
`meta/provenance.parquet` is computed from installed distribution metadata and
`uv.lock`; it does not require the optional `pip` module.

## Canonical 120-branch smoke run

This shell invocation corresponds to the script argv stored in provenance
(provenance records `tools/run_smoke_suite.py ...`, without the `uv run python`
launcher prefix):

```bash
cd /gpfs/radev/project/sous/zl664/dataset_generation
uv run python tools/run_smoke_suite.py \
  --config configs/families/smoke_120.yaml \
  --output outputs/smoke_tests/canonical_smoke_120
```

The command refuses to overwrite an existing run. To reproduce without moving
the accepted artifacts, choose a new output basename; its Wan export is written
next to it automatically:

```bash
uv run python tools/run_smoke_suite.py \
  --config configs/families/smoke_120.yaml \
  --output outputs/smoke_tests/reproduced_smoke_120
```

## Standalone verification

```bash
uv run dynamic-robot-dataset validate \
  --dataset-root outputs/smoke_tests/canonical_smoke_120

uv run python tools/validate_physics.py \
  --dataset-root outputs/smoke_tests/canonical_smoke_120
```

The smoke runner already performs finalization, deterministic group-aware
split construction, deep video/metadata/label/duplicate/leakage QC, the Wan
round trip, and contact-sheet creation. The standalone commands rerun the two
main validators read-only.

## Key accepted artifacts

```text
outputs/smoke_tests/canonical_smoke_120/.generation.json
outputs/smoke_tests/canonical_smoke_120/meta/.complete.json
outputs/smoke_tests/canonical_smoke_120/qc/smoke_summary.json
outputs/smoke_tests/canonical_smoke_120/qc/dataset_report.json
outputs/smoke_tests/canonical_smoke_120/qc/contact_sheets/samples.json
outputs/smoke_tests/canonical_smoke_120_wan/manifest.jsonl
outputs/smoke_tests/canonical_smoke_120_wan/checksums.json
outputs/smoke_tests/canonical_smoke_120_wan/source_root.json
```
