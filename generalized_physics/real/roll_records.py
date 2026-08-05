"""Ball-rolling episode -> TypedRecord conditioning bundle (4 keys).

The corpus varies exactly two degrees of freedom per episode — rolling speed
and heading — so the bundle is deliberately TRIMMED to the keys that carry
them and nothing else: the camera-frame initial velocity (3 components, a
linear injection of (speed, heading) given the fixed per-family camera) plus
log10 speed. Constants (start pose, radius, mass, physics) are excluded by
design rather than left for the normalizer's constant-drop: with a 2-D
signal the shared token component already dominates, and padding keys only
dilute it further (the pbc muffling diagnosis, problem 3).

All four keys already exist in metadata_registry.yaml (the projectile set),
so the registry — and therefore the teacher's key-embedding table — is
unchanged and old checkpoints stay loadable.

State is expressed in the MAIN camera's frame via the same construction as
``pbc_records.camera_matrix`` (world +z up): the camera relates tokens to
pixels, and the cache stores main-view latents. The camera is bit-identical
across a family, so correct/wrong sibling comparisons are unaffected.

Wrong-state negatives swap this ENTIRE record set between same-group
siblings (``GroupBatcher.paired_batch``). There are no commands or actions
in this passive corpus, so the state bundle IS the full conditioning bundle
— the swap is internally consistent by construction.
"""

import math

import numpy as np

from ..models.metadata_records import TypedRecord
from .pbc_records import camera_matrix

#: key -> (scope, unit); every key must exist in metadata_registry.yaml.
FIELD_SPECS = {
    "v0_cam_x": ("primary_object", "m_per_s"),
    "v0_cam_y": ("primary_object", "m_per_s"),
    "v0_cam_z": ("primary_object", "m_per_s"),
    "log10_speed": ("primary_object", "canonical_sim_unit"),
}


def build_records(record: dict):
    """Canonical episode record (marker) -> list[TypedRecord], RAW SI values."""
    cam = record["extras"]["camera_poses"]["main_camera"]
    R, _t = camera_matrix(cam)
    v0 = np.asarray(record["physics"]["ball_initial_velocity_m_s"],
                    dtype=np.float64)
    v0c = R @ v0                                  # rotation only: a velocity
    speed = float(np.linalg.norm(v0))

    vals = {
        "v0_cam_x": v0c[0], "v0_cam_y": v0c[1], "v0_cam_z": v0c[2],
        "log10_speed": math.log10(max(speed, 1e-6)),
    }
    return [TypedRecord(k, FIELD_SPECS[k][0], FIELD_SPECS[k][1], float(v))
            for k, v in vals.items() if np.isfinite(v)]
