"""Change-point windows for sliding-window adaptation (plan: Phase 4).

Episodes whose dynamic friction switches at a known frame. For each episode
we emit end-anchored windows entirely before the change, straddling it, and
entirely after it. The teacher target for a window is the metadata of the
CURRENT regime — the regime at the window's last bin — matching deployment,
where the newest evidence defines what the belief should describe.
"""

import torch

from .counterfactual_dataset import GroupBatcher, _gather_rec


class ChangePointWindows:
    def __init__(self, cache, window, seed=0):
        if "change_bin" not in cache:
            raise ValueError("cache was not built with change_frame set")
        self.c = cache
        self.W = window
        self.change_bin = cache["change_bin"]
        self.batcher = GroupBatcher(cache, seed=seed)

    def phases_for(self, end):
        """'before' / 'straddle' / 'after' for an end-anchored window."""
        start = max(0, end - self.W)
        if end <= self.change_bin:
            return "before"
        if start >= self.change_bin:
            return "after"
        return "straddle"

    def batch(self, batch_size, end):
        """Window batch ending at latent bin `end` with current-regime
        record targets."""
        b = self.batcher.episode_batch(batch_size)
        win = self.batcher.window(b, self.W, end=end)
        rec_key = ("rec_batch_after" if end > self.change_bin
                   else "rec_batch")
        win["rec"] = _gather_rec(self.c[rec_key], b["idx"])
        win["phase"] = self.phases_for(end)
        win["idx"] = b["idx"]
        return win

    def sweep_ends(self):
        """All valid window endpoints, covering before/straddle/after."""
        Tz = self.c["z"].shape[2]
        return list(range(max(2, self.W // 2), Tz + 1))
