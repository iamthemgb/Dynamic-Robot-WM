"""Generalized Adaptive Physics Embedding (plan:
~/scratch/Generalized_Adaptive_Physics_Embedding_Plan.tex).

Six core modules:
  1. frozen Wan VAE + versioned latent cache          (data/cache_wan_latents.py)
  2. variable-record metadata teacher -> 64-D belief  (models/metadata_teacher.py)
  3. causal student from ordered VAE/action history   (models/causal_student.py)
  4. shared projector 64 -> 8 x 256 physics tokens    (models/physics_projector.py)
  5. four zero-gated physics cross-attn adapters      (wan/physics_adapter.py)
  6. rank-16 zero-init LoRA on frozen DiT attention   (wan/dit_lora.py)

Everything runs on CPU against wan/mock_wan.py; the real Wan2.1 insertion
follows the wrapped-block pattern of
Wan2.1/projectile_adaptive_smoke/models/adaptive_wan.py.
"""
