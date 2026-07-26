"""Frozen-VAE latent caching with provenance (plan: Module 1 + episode
schema).

Every cached episode records the VAE checkpoint hash and preprocessing so a
stale cache can never silently mix encoders. Latents are stored at full
episode length; training slices windows from them. The causal first-chunk
boundary is marked in delta_valid (dZ between bin 0 and bin 1 compares a
1-frame chunk against a stride-frame chunk and is not comparable).
"""

import torch


def control_bin_index(n_control, temporal_stride):
    """Control step -> latent bin under Wan's causal chunking: frame 0 forms
    bin 0 alone; control step t (frame t -> t+1) lands in bin 1 + t//stride."""
    t = torch.arange(n_control)
    return 1 + t // temporal_stride


def encode_episode(vae, frames):
    """frames np/tensor [T, C, H, W] -> dict with z, delta, delta_valid.

    z          [Cz, Tz, Hz, Wz]
    delta      [Cz, Tz, Hz, Wz]   z_t - z_{t-1}; bin 0 zeroed
    delta_valid[Tz]               0 for bins 0 and 1 (causal boundary), else 1
    """
    x = torch.as_tensor(frames)[None].float()
    z = vae.encode(x)[0]                          # [Cz, Tz, Hz, Wz]
    delta = torch.zeros_like(z)
    delta[:, 1:] = z[:, 1:] - z[:, :-1]
    valid = torch.ones(z.shape[1])
    valid[:2] = 0.0
    return {"z": z, "delta": delta, "delta_valid": valid}


def cache_episodes(vae, episodes):
    """episodes: list of synthetic_env.Episode -> stacked tensor cache with
    provenance. Records stay as python lists (variable length is the point);
    collation happens in counterfactual_dataset with a fitted normalizer."""
    enc = [encode_episode(vae, e.frames) for e in episodes]
    n_control = episodes[0].actions.shape[0]
    cache = {
        "z": torch.stack([d["z"] for d in enc]),
        "delta": torch.stack([d["delta"] for d in enc]),
        "delta_valid": torch.stack([d["delta_valid"] for d in enc]),
        "actions": torch.stack(
            [torch.as_tensor(e.actions).float() for e in episodes]),
        "bin_index": control_bin_index(n_control, vae.temporal_stride),
        "vae_checkpoint_hash": vae.checkpoint_hash,
        "temporal_stride": vae.temporal_stride,
    }
    assert all(e.actions.shape[0] == n_control for e in episodes)
    return cache


def verify_provenance(cache, vae):
    if cache["vae_checkpoint_hash"] != vae.checkpoint_hash:
        raise RuntimeError(
            f"latent cache was built with VAE {cache['vae_checkpoint_hash']} "
            f"but the loaded VAE is {vae.checkpoint_hash}; re-run caching")
