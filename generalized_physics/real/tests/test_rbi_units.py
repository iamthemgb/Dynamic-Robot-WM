"""CPU unit tests for the rbi campaign modules (no Wan, no cache)."""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from generalized_physics.real.rbi_action import ActionTokenEncoder, \
    _ActionShim
from generalized_physics.real.rbi_convert_cache import _assert_pairs, \
    _build_index, _main_rows
from generalized_physics.real.rbi_phase_action import ActionPairBatcher
from generalized_physics.wan.physics_adapter import PhysicsAdapterBank


def test_action_encoder_contract():
    enc = ActionTokenEncoder()
    out = enc(torch.randn(2, 73, 8))
    assert out.shape == (2, 8, 256)
    assert out.dtype == torch.float32
    try:
        enc(torch.randn(2, 73, 9))
    except ValueError:
        pass
    else:
        raise AssertionError("wrong action_dim must raise")


def test_zero_gate_shim_is_noop():
    class Inner(nn.Module):
        def forward(self, x, **kwargs):
            return x + 1.0

    bank = PhysicsAdapterBank({"d_model": 32, "n_blocks": 8},
                              d_phys=16, n_adapters=4, heads=4)
    ref = {"ctx": None}
    shim = _ActionShim(Inner(), bank.adapter_for(bank.block_idx[0]), ref)
    x = torch.randn(2, 5, 32)
    want = x + 1.0
    assert torch.equal(shim(x), want)                  # ctx None: bypassed
    ref["ctx"] = torch.randn(2, 8, 16)
    assert torch.equal(shim(x), want)                  # zero gate: exact
    g_mean, g_max = bank.gate_stats()
    assert g_mean == 0.0 and g_max == 0.0


def test_action_pair_batcher_swaps_within_pair():
    n = 8
    cache = {
        "z": torch.arange(n, dtype=torch.float32).reshape(n, 1, 1, 1, 1),
        "delta": torch.zeros(n, 1, 1, 1, 1),
        "delta_valid": torch.ones(n, 1),
        "actions": (torch.arange(n, dtype=torch.float32)[:, None, None]
                    .repeat(1, 3, 2)),
        "bin_index": torch.zeros(3, dtype=torch.long),
        "rec_batch": {"values": torch.arange(n, dtype=torch.float32)[:, None]},
        "events": [{"impact": None}] * n,
        "temporal_stride": 4,
        "group_id": torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]),
    }
    batch = ActionPairBatcher(cache, seed=0).paired_batch(32)
    for i, w in zip(batch["idx"].tolist(), batch["wrong_idx"].tolist()):
        assert w != i and w // 2 == i // 2, (i, w)     # the other pair member
    assert torch.equal(batch["actions_wrong"],
                       cache["actions"][batch["wrong_idx"]])


def _toy_samples():
    rows = []
    rank = 0
    for gi, split in ((0, "train"), (1, "validation_id")):
        for cell, state, action in (("vA_aA", "A", "A"), ("vA_aB", "A", "B"),
                                    ("vB_aA", "B", "A"), ("vB_aB", "B", "B")):
            for vi, view in enumerate(("main", "side")):
                rows.append({
                    "episode_id": f"rbi_g{gi:06d}_{cell}",
                    "group_id": f"rbi_g{gi:06d}", "group_index": gi,
                    "cell_id": cell, "state_id": state, "action_id": action,
                    "split": split, "contrast_type": "direction",
                    "catch_success": state == action,
                    "v0_robot_x": -0.4 - 0.01 * (state == "B"),
                    "v0_robot_y": 0.05 - 0.1 * (state == "B"),
                    "action_plan_sha256": f"sha_{gi}_{action}",
                    "view": view, "sample_index": 2 * rank + vi,
                })
            rank += 1
    return pd.DataFrame(rows)


def test_converter_row_map(tmp_path):
    _toy_samples().to_parquet(tmp_path / "samples.parquet")
    df = _main_rows(tmp_path)
    assert len(df) == 8
    assert (df["sample_index"].to_numpy() % 2 == 0).all()
    idx = _build_index(df)
    _assert_pairs(idx)
    # velocity-pair members adjacent (rows 2k, 2k+1); action pairs (0,2),(1,3)
    gid = idx["group_id"].to_numpy()
    assert (gid[0::2] == gid[1::2]).all()
    for k in range(0, len(idx), 4):
        quad = idx.iloc[k:k + 4]
        assert quad["rbi_group_id"].nunique() == 1
        ag = quad["action_group_id"].to_numpy()
        assert ag[0] == ag[2] and ag[1] == ag[3] and ag[0] != ag[1]
    assert set(idx["split"]) == {"train", "val"}       # validation_id mapped
    assert (idx["prompt_id"] == 0).all()
    assert (idx["impact_frame"] == -1).all()
    # source rows: sorted by (group, action, state) -> A/B states per action
    first = idx.iloc[0]
    assert (first["state_id"], first["action_id"]) == ("A", "A")
    assert (idx.iloc[1]["state_id"], idx.iloc[1]["action_id"]) == ("B", "A")


def test_main_rows_limit_guard(tmp_path):
    _toy_samples().to_parquet(tmp_path / "samples.parquet")
    try:
        _main_rows(tmp_path, limit=6)
    except SystemExit:
        pass
    else:
        raise AssertionError("--limit % 4 != 0 must raise")
    assert len(_main_rows(tmp_path, limit=4)) == 4
