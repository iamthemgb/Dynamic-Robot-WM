"""Configuration for the generalized adaptive physics embedding pipeline.

Plain dataclasses (no YAML dependency for configs; the metadata *registry*
is YAML because it is data, not config). Values mirror the plan's
"Suggested initial hyperparameters"; tiny_config() shrinks everything so the
full pipeline and unit tests run on CPU.

Contract notes enforced elsewhere:
  * d_model is read from the loaded DiT (wan_config()), never hard-coded;
  * the physics-token interface [K=8, d_phys=256] is fixed and independent
    of both the task schema and the Wan width;
  * belief has no per-sample LayerNorm and the projector accepts the belief
    mean only.
"""

from dataclasses import dataclass, field
from pathlib import Path

REGISTRY_PATH = Path(__file__).parent / "data" / "metadata_registry.yaml"


@dataclass
class EnvConfig:
    """Synthetic multi-family counterfactual environments."""
    n_frames: int = 33
    image_size: int = 64
    dt: float = 1.0 / 16.0
    families: tuple = ("projectile_impact", "push_slide")


@dataclass
class VAEConfig:
    """Mock Wan VAE; the real integration swaps the checkpoint in and reads
    every shape from the loaded model."""
    latent_channels: int = 8
    temporal_stride: int = 4
    spatial_stride: int = 8
    seed: int = 1234


@dataclass
class TeacherConfig:
    record_width: int = 256       # d_r
    key_embed: int = 64
    scope_embed: int = 32
    unit_embed: int = 16
    belief_dim: int = 64          # d_b
    output_scale: float = 1.0
    tanh_output: bool = False


@dataclass
class StudentConfig:
    width: int = 256              # visual/control evidence width
    gru_layers: int = 2
    gru_hidden: int = 256
    ctrl_width: int = 64
    use_delta_z: bool = False     # [Z; dZ] is an optimization ablation
    dropout: float = 0.0


@dataclass
class ProjectorConfig:
    k_tokens: int = 8             # K
    d_phys: int = 256             # fixed Wan-independent token width
    hidden: int = 256


@dataclass
class AdapterConfig:
    n_adapters: int = 4           # quarter-depth placement
    heads: int = 4
    dropout: float = 0.0


@dataclass
class LoRAConfig:
    rank: int = 16
    alpha: float = 16.0
    dropout: float = 0.05
    targets: tuple = ("q", "v")   # attribute names of attention projections


@dataclass
class DiTConfig:
    """Mock Wan DiT; d_model doubles as wan_config()['d_model']."""
    d_model: int = 96
    heads: int = 8
    blocks: int = 4
    prefix_bins: int = 5
    dropout: float = 0.0


@dataclass
class TrainConfig:
    batch_size: int = 8
    lr_teacher_student: float = 3e-4
    lr_projector_adapters: float = 1e-4   # also LoRA (plan hyperparameters)
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    seed: int = 0
    # phase-1 loss weights (normalized starting values from the plan)
    w_fm: float = 1.0
    w_rank: float = 0.5
    w_meta: float = 0.2
    rank_margin: float = 0.05
    p_null: float = 0.10          # learned-null-code batches
    p_noise: float = 0.10         # small-code-noise batches
    noise_scale: float = 0.1
    # phase-3 conditioning mixture
    mix_teacher: float = 0.40
    mix_student: float = 0.40
    mix_corrupt: float = 0.10
    mix_null: float = 0.10
    w_query_s: float = 0.5
    w_fm_s: float = 0.25


@dataclass
class PipelineConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    vae: VAEConfig = field(default_factory=VAEConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    student: StudentConfig = field(default_factory=StudentConfig)
    projector: ProjectorConfig = field(default_factory=ProjectorConfig)
    adapters: AdapterConfig = field(default_factory=AdapterConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    dit: DiTConfig = field(default_factory=DiTConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    window: int = 8               # sliding window W in latent bins
    n_train_groups: int = 60
    n_eval_groups: int = 12
    variants_per_group: int = 4

    phase0_steps: int = 800
    phase1_steps: int = 1000
    phase2_steps: int = 600
    phase3_steps: int = 400
    phase4_windows: int = 64

    workdir: str = "runs/generalized_smoke"
    num_threads: int = 4


def smoke_config() -> PipelineConfig:
    return PipelineConfig()


def tiny_config() -> PipelineConfig:
    """Minimal config for unit tests (fast model construction on CPU)."""
    cfg = PipelineConfig()
    cfg.teacher.record_width = 64
    cfg.teacher.key_embed = 16
    cfg.teacher.scope_embed = 8
    cfg.teacher.unit_embed = 4
    cfg.student.width = 64
    cfg.student.gru_hidden = 64
    cfg.student.ctrl_width = 32
    cfg.projector.hidden = 64
    cfg.lora.rank = 4
    cfg.lora.alpha = 4.0
    cfg.lora.dropout = 0.0
    cfg.dit.d_model = 64
    cfg.dit.blocks = 4
    cfg.dit.prefix_bins = 3
    cfg.env.n_frames = 17
    cfg.env.image_size = 32
    cfg.window = 4
    cfg.n_train_groups = 4
    cfg.n_eval_groups = 2
    cfg.phase0_steps = 5
    cfg.phase1_steps = 5
    cfg.phase2_steps = 5
    cfg.phase3_steps = 5
    cfg.phase4_windows = 4
    return cfg
