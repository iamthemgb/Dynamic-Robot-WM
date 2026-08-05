"""Injecting the real Wan DiT into the stock phase modules.

Every phase reaches the mock through ``training/common.py`` and does so as
attribute lookups at call time::

    from . import common
    ...
    l_fm, tau, eps = common.fm_loss(dit, batch, tokens, prefix)

so rebinding attributes on that module redirects all five phases at once. The
phase files themselves are never edited.

Audited surface -- across phases 0-4 the only things ever done to ``dit`` are
``common.wan_side_params(cfg, dit, projector)`` and ``common.fm_loss(dit, ...)``
(phases 0, 2 and 4 do not touch it at all), so ``RealWanDiT`` only has to
expose ``.physics`` and be accepted by the two rebound functions.
"""

import contextlib
from functools import partial

import torch
import torch.nn as nn

from ..training import common
from ..wan import adaptive_wan_integration as awi
from . import wan_loader as W
from .flow_match import flow_loss, make_targets, sample_sigmas


@contextlib.contextmanager
def _loader_injected(arm, dtype, device="cpu"):
    """Make GeneralizedAdaptiveWan use our loader instead of from_pretrained.

    ``from_pretrained`` works but materialises fp32 on CPU before casting,
    which is a ~57 GB spike for the 14B. Patching it for the duration of the
    constructor reuses all of GeneralizedAdaptiveWan's adapter/LoRA wiring
    unchanged while avoiding that.
    """
    W.push_wan_repo(arm.repo)
    from wan.modules.model import WanModel

    original = getattr(WanModel, "from_pretrained", None)

    def _load(model_dir, *a, **kw):
        model, _ = W.load_wan_model(model_dir, arm.repo, dtype=dtype,
                                    device=device)
        return model

    WanModel.from_pretrained = staticmethod(_load)
    awi.WAN_REPO = W.WAN_REPOS[arm.repo]
    awi.MODEL_DIR = str(arm.model_dir)
    try:
        yield
    finally:
        if original is not None:
            WanModel.from_pretrained = original


class RealWanDiT(nn.Module):
    """Duck-types the MockWanDiT surface that training/common.py touches."""

    def __init__(self, cfg, arm, device="cuda", dtype=torch.bfloat16):
        super().__init__()
        self.arm = arm
        self.device = device
        # Stream the frozen base straight onto the GPU so host RAM never has to
        # hold a 28 GB (14B) copy; the adapters/LoRA are built on top and moved
        # with the .to() below.
        with _loader_injected(arm, dtype, device=device):
            self.model = awi.GeneralizedAdaptiveWan(
                cfg, model_dir=str(arm.model_dir), torch_dtype=dtype)
        self.model.to(device)
        self.physics = self.model.physics          # wan_side_params reads this
        cfg_in = W.read_model_config(arm.model_dir)["in_dim"]
        if cfg_in != arm.latent_channels:
            raise RuntimeError(
                f"{arm.name}: WanModel.in_dim={cfg_in} but the cache supplies "
                f"{arm.latent_channels} latent channels")

    def wan_config(self):
        return self.model.wan_config()

    def trainable_params(self):
        return self.model.trainable_params()

    def set_lora_enabled(self, enabled: bool):
        self.model.set_lora_enabled(enabled)


def make_real_fm_loss(dit: RealWanDiT, text_ctx, sigma_sampler=None,
                      roi_lambda: float = 0.0):
    """Build the drop-in replacement for ``training.common.fm_loss``.

    Signature and return contract are identical -- ``(loss, tau, eps)`` -- so
    the shared-noise ranking in phase 1 and phase 3 keeps working untouched:
    the caller passes the same ``(tau, eps)`` back in for the "wrong" and
    "null" conditions, and here ``tau`` carries Wan's sigma.

    ``prefix_bins`` is accepted and ignored. The mock splits the latent into a
    conditioning prefix and a future, and cross-attends to the prefix; real Wan
    has no such path, so the flow loss is over the whole clip.

    ``sigma_sampler`` (optional ``fn(batch_size) -> [B] sigmas on device``)
    replaces the shifted-uniform draw for TRAINING steps only: it is consulted
    when ``tau is None and generator is None``. Paired evals thread an
    explicit generator for reproducibility and always get the standard
    schedule.

    ``roi_lambda > 0`` weights the flow MSE with ``1 + roi_lambda * roi``
    (mean-normalized per sample) using ``batch["roi"]``; the weight depends
    only on the episode, so correct/wrong/null comparisons under shared
    (tau, eps) remain exact. Gated on the config value, never on mere key
    presence, so a roi-bearing cache does not silently change other phases.
    """
    device = dit.device
    shift = dit.arm.shift
    per_token_t = dit.arm.repo == "wan22"

    def real_fm_loss(_dit, batch, tokens, prefix_bins, tau=None, eps=None,
                     generator=None, roi_only=False):
        x0 = batch["z"].to(device=device, dtype=torch.float32)
        b = x0.shape[0]
        if tau is None:
            if sigma_sampler is not None and generator is None:
                tau = sigma_sampler(b)
            else:
                tau = sample_sigmas(b, shift, device=device,
                                    generator=generator)
        tau = tau.to(device)
        weight = None
        if roi_only:
            # Eval-only reduction: mean over the ball tube alone — the
            # region where sibling codes can differ at all (paired_eval
            # detects support for this kwarg before using it).
            roi = batch["roi"].to(device=device, dtype=torch.float32)
            w = roi / roi.mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
            weight = w[:, None]
        elif roi_lambda > 0.0:
            roi = batch["roi"].to(device=device, dtype=torch.float32)
            w = 1.0 + roi_lambda * roi                    # [B, Tz, Hz, Wz]
            w = w / w.mean(dim=(1, 2, 3), keepdim=True)
            weight = w[:, None]                           # broadcast over C
        x_t, v_target, t, eps = make_targets(x0, tau, eps=eps,
                                             generator=generator)
        seq_len = W.seq_len_of(x0.shape)
        if per_token_t:
            # Wan2.2 uses per-token time modulation and its 1-D broadcast
            # (`t.expand(t.size(0), seq_len)`) only works at B=1; the 2-D
            # path accepts an explicit [B, seq_len].
            t = t[:, None].expand(-1, seq_len)
        ctx = text_ctx(batch["idx"])
        with torch.autocast("cuda", torch.bfloat16):
            # x must be a list: WanModel does [patch_embedding(u.unsqueeze(0))
            # for u in x]. autocast is mandatory, not an optimisation -- the
            # adapters are fp32 while the DiT is bf16.
            pred = dit.model(x=list(x_t), t=t, context=ctx,
                             seq_len=seq_len,
                             physics_ctx=tokens.to(device))
        return flow_loss(pred, v_target, weight=weight), tau, eps

    return real_fm_loss


class _CachedVAEStub:
    """Phases call build_vae() then discard it when a cache is supplied."""

    def __init__(self, checkpoint_hash, temporal_stride):
        self.checkpoint_hash = checkpoint_hash
        self.temporal_stride = temporal_stride

    def encode(self, *_a, **_k):
        raise RuntimeError("latents are precomputed; encode_cache.py builds them")


def install_real_backend(arm, cache, text_ctx, cfg, device="cuda",
                         dtype=torch.bfloat16, action_dim=8, state_dim=0,
                         sigma_sampler=None, roi_lambda: float = 0.0):
    """Rebind training.common so all five phases run on real data + real Wan.

    Returns the constructed RealWanDiT so the caller can report memory and
    parameter counts. Phases 0/2/4 never build a DiT, so callers that only run
    those can pass ``build_dit=False`` by simply not using the return value --
    construction is lazy inside the rebound ``build_dit``.

    ``sigma_sampler`` / ``roi_lambda`` are forwarded to
    ``make_real_fm_loss``; the defaults reproduce existing behavior exactly.
    """
    holder = {}

    def build_dit(cfg_, action_dim=1, with_lora=True):
        if "dit" not in holder:
            dit = RealWanDiT(cfg_, arm, device=device, dtype=dtype)
            holder["dit"] = dit
            common.fm_loss = make_real_fm_loss(
                dit, text_ctx, sigma_sampler=sigma_sampler,
                roi_lambda=roi_lambda)
        return holder["dit"]

    def wan_side_params(cfg_, dit, projector):
        return common.dedupe_params(list(projector.parameters())
                                    + dit.model.trainable_params())

    def to_device(fn):
        return lambda *a, **kw: fn(*a, **kw).to(device)

    common.build_vae = lambda cfg_: _CachedVAEStub(
        cache["vae_checkpoint_hash"], cache["temporal_stride"])
    common.build_dit = build_dit
    common.wan_side_params = wan_side_params
    common.build_teacher = to_device(common.build_teacher)
    common.build_decoder = to_device(common.build_decoder)
    common.build_projector = to_device(common.build_projector)
    _student = common.build_student
    common.build_student = lambda cfg_, action_dim=action_dim, \
        state_dim=state_dim: _student(cfg_, action_dim, state_dim).to(device)

    # fm_loss is bound when the DiT is first built; until then any call is a
    # bug (a DiT-free phase must never reach it).
    def _unbuilt(*_a, **_k):
        raise RuntimeError("fm_loss called before build_dit; phases 0/2/4 "
                           "should not need it")
    common.fm_loss = _unbuilt
    return holder
