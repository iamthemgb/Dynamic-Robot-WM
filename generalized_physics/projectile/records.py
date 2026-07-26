"""Projectile 14-field schema -> typed metadata records.

The prior smoke runs fed the whitened 14-D vector straight to a projector;
here the SAME whitened values become 14 typed records (key, scope, unit,
value) for the variable-schema teacher. Scopes/units come from the registry;
the whitened values are bit-identical to both prior arms (same
norm_stats.json mean/std), so conditioning-architecture comparisons stay
clean.
"""

import torch

from ..models.metadata_records import MetadataRegistry, TypedRecord

# field -> (scope, unit); keys are registered in data/metadata_registry.yaml
FIELD_SPECS = {
    "log10_ball_mass": ("primary_object", "canonical_sim_unit"),
    "ball_radius": ("primary_object", "m"),
    "p0_cam_x": ("primary_object", "m"),
    "p0_cam_y": ("primary_object", "m"),
    "p0_cam_z": ("primary_object", "m"),
    "v0_cam_x": ("primary_object", "m_per_s"),
    "v0_cam_y": ("primary_object", "m_per_s"),
    "v0_cam_z": ("primary_object", "m_per_s"),
    "log10_speed": ("primary_object", "canonical_sim_unit"),
    "icpt_cam_x": ("global", "m"),
    "icpt_cam_y": ("global", "m"),
    "icpt_cam_z": ("global", "m"),
    "ballistic_intercept_time_s": ("global", "s"),
    "gripper_close_time": ("robot", "s"),
}


class ProjectileRecordSchema:
    """Precomputed id tensors for the fixed projectile field order."""

    def __init__(self, fields, registry=None):
        self.registry = registry or MetadataRegistry()
        self.fields = list(fields)
        # registry validation (scope sanity, allowed scopes) on a template
        self.registry.validate([
            TypedRecord(f, FIELD_SPECS[f][0], FIELD_SPECS[f][1], 0.0)
            for f in self.fields])
        self.key_ids = torch.tensor(
            [self.registry.key_to_id[f] for f in self.fields])
        self.scope_ids = torch.tensor(
            [self.registry.scope_to_id[FIELD_SPECS[f][0]]
             for f in self.fields])
        self.unit_ids = torch.tensor(
            [self.registry.unit_to_id[FIELD_SPECS[f][1]]
             for f in self.fields])

    def batch(self, whitened, device=None):
        """whitened [B, J] -> collate_records-style dict of tensors."""
        B, J = whitened.shape
        assert J == len(self.fields), (J, len(self.fields))
        ids = {"key_ids": self.key_ids, "scope_ids": self.scope_ids,
               "unit_ids": self.unit_ids}
        out = {k: v[None].expand(B, -1).contiguous() for k, v in ids.items()}
        out["values"] = whitened[..., None].float()
        out["valid_mask"] = torch.ones(B, J)
        if device is not None:
            out = {k: v.to(device) for k, v in out.items()}
        return out
