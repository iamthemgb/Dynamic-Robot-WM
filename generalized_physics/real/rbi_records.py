"""rbi episode row -> TypedRecord conditioning bundle (2 keys).

The interception corpus varies exactly one thing within a counterfactual
velocity pair: the ball's initial velocity in the robot base frame. The
bundle is deliberately TRIMMED to those two components and nothing else —
constants (radius, mass, physics, scene) are excluded by design rather than
left for the normalizer's constant-drop (the pbc muffling diagnosis,
problem 3), and the action plan is deliberately NOT a record: within every
velocity pair the plan is bit-identical, so it carries zero discriminative
signal for the paired gate and would only compete with the velocity code
(rbi_phase1_problems_and_fixes.tex, problem B1). The action pathway trains
in a separate stage (``rbi_phase_action``) that never touches records.

This 2-key bundle IS the full stage-1 conditioning bundle. Wrong-state
negatives swap the ENTIRE record set between velocity-pair siblings
(``GroupBatcher.paired_batch``); the arm executes the same plan in both, so
the swap isolates velocity exactly.

Values are raw SI (robot base frame, no camera transform — the corpus
records them in that frame and the camera is bit-identical across a group);
``MetadataNormalizer`` whitens from train-split statistics. Both keys live
in ``data/metadata_registry.yaml`` (rbi block).
"""

import numpy as np

from ..models.metadata_records import TypedRecord

#: key -> (scope, unit); every key must exist in metadata_registry.yaml.
FIELD_SPECS = {
    "v0_robot_x": ("primary_object", "m_per_s"),
    "v0_robot_y": ("primary_object", "m_per_s"),
}


def build_records(row):
    """samples.parquet row (itertuples/Series/dict) -> list[TypedRecord]."""
    get = (row.get if hasattr(row, "get")
           else lambda k: getattr(row, k))
    vals = {k: float(get(k)) for k in FIELD_SPECS}
    return [TypedRecord(k, FIELD_SPECS[k][0], FIELD_SPECS[k][1], v)
            for k, v in vals.items() if np.isfinite(v)]
