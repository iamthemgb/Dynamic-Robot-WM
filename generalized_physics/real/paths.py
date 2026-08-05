"""Cluster-local paths and the per-arm checkpoint specs.

Everything that knows where things live on this filesystem is here, so the
rest of the package stays path-free. Mirrors the role of
``zl664/harness/env.sh`` for the dataset side.
"""

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(os.environ.get(
    "WANPHYS_ROOT",
    "/scratch/zl664_yale/world_model_robotics/Dynamic-Robot-WM-main"))

THIRD_PARTY = ROOT / "third_party"
WAN_REPOS = {"wan21": THIRD_PARTY / "Wan2.1", "wan22": THIRD_PARTY / "Wan2.2"}

CHECKPOINTS = ROOT / "Wan_checkpoint"
DATASET_ROOT = ROOT / "mzl7" / "f1_10h"
OUT_ROOT = ROOT / "mzl7" / "physics_wan"

# The umT5 text encoder and its tokenizer ship with every Wan checkpoint; the
# 1.3B copy is the canonical one so we never load a second 11.4 GB file.
T5_DIR = CHECKPOINTS / "Wan2.1-T2V-1.3B"


@dataclass(frozen=True)
class ArmSpec:
    """One checkpoint under test."""

    name: str            # run-directory name
    model_dir: Path      # dir holding config.json + diffusion_pytorch_model*.safetensors
    repo: str            # "wan21" | "wan22" -- which vendored source tree
    vae_kind: str        # "wan21" | "wan22"
    vae_path: Path
    latent_channels: int  # must equal WanModel.in_dim
    cache_name: str      # which cache dir under OUT_ROOT/cache this arm reads
    shift: float = 5.0   # flow-matching sigma shift; identical across arms

    @property
    def cache_dir(self) -> Path:
        return OUT_ROOT / "cache" / self.cache_name

    @property
    def run_dir(self) -> Path:
        return OUT_ROOT / "runs" / self.name


_W21_VAE = CHECKPOINTS / "Wan2.1-T2V-1.3B" / "Wan2.1_VAE.pth"
_W22_VAE = CHECKPOINTS / "Wan2.2-TI2V-5B" / "Wan2.2_VAE.pth"

ARMS = {
    "wan21_t2v_1p3b": ArmSpec(
        name="wan21_t2v_1p3b",
        model_dir=CHECKPOINTS / "Wan2.1-T2V-1.3B",
        repo="wan21", vae_kind="wan21", vae_path=_W21_VAE,
        latent_channels=16, cache_name="wan21_vae"),
    "wan22_ti2v_5b": ArmSpec(
        name="wan22_ti2v_5b",
        model_dir=CHECKPOINTS / "Wan2.2-TI2V-5B",
        repo="wan22", vae_kind="wan22", vae_path=_W22_VAE,
        latent_channels=48, cache_name="wan22_vae"),
    "wan21_t2v_14b": ArmSpec(
        name="wan21_t2v_14b",
        model_dir=CHECKPOINTS / "Wan2.1-T2V-14B",
        repo="wan21", vae_kind="wan21", vae_path=_W21_VAE,
        latent_channels=16, cache_name="wan21_vae"),
}

# Frames fed to the VAE. The Wan encoder chunks causally as 1 + (T-1)//4 and
# silently drops the tail, so 60 would discard frames 57-59 without saying so.
# 57 makes Tz = 15 exact and deterministic.
N_FRAMES = 57
FPS = 30.0
TEMPORAL_STRIDE = 4
