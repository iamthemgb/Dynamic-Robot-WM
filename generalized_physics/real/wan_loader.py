"""Loading real Wan models, VAEs and the T5 text encoder.

Two vendored source trees are in play and **both packages are named ``wan``**,
so exactly one may be imported per process. ``push_wan_repo`` enforces that.

``load_wan_model`` deliberately bypasses ``diffusers``' ``from_pretrained``:
building on ``meta`` and assigning safetensors shards directly avoids the
``accelerate`` dependency, the ``torch_dtype``->``dtype`` kwarg churn across
diffusers versions, and any silent CPU-RAM spike from a full fp32
materialisation of a 14B model.
"""

import hashlib
import json
import sys
from pathlib import Path

import torch

from .paths import T5_DIR, WAN_REPOS

# WanModel.__init__ accepts exactly these; config.json also carries
# _class_name / _diffusers_version, which are diffusers bookkeeping.
_MODEL_KWARGS = {
    "model_type", "patch_size", "text_len", "in_dim", "dim", "ffn_dim",
    "freq_dim", "text_dim", "out_dim", "num_heads", "num_layers",
    "window_size", "qk_norm", "cross_attn_norm", "eps",
}


def push_wan_repo(which: str) -> Path:
    """Put one vendored Wan tree on sys.path; refuse to mix the two."""
    repo = WAN_REPOS[which]
    if not (repo / "wan" / "modules" / "model.py").exists():
        raise FileNotFoundError(f"vendored Wan repo missing: {repo}")
    existing = sys.modules.get("wan")
    if existing is not None:
        have = Path(existing.__file__).resolve().parent.parent
        if have != repo.resolve():
            raise RuntimeError(
                f"'wan' already imported from {have}; cannot also load "
                f"{repo}. Run one arm per process.")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    return repo


def sha256_head(path: Path, n_bytes: int = 64 << 20) -> str:
    """Hash of the first n_bytes -- enough to pin provenance, cheap on NFS."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(n_bytes))
    return h.hexdigest()[:16]


def read_model_config(model_dir: Path) -> dict:
    cfg = json.loads((Path(model_dir) / "config.json").read_text())
    return {k: v for k, v in cfg.items() if k in _MODEL_KWARGS}


def _shard_files(model_dir: Path) -> list:
    model_dir = Path(model_dir)
    index = model_dir / "diffusion_pytorch_model.safetensors.index.json"
    if index.exists():
        wm = json.loads(index.read_text())["weight_map"]
        return [model_dir / f for f in sorted(set(wm.values()))]
    single = model_dir / "diffusion_pytorch_model.safetensors"
    if single.exists():
        return [single]
    shards = sorted(model_dir.glob("diffusion_pytorch_model-*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors under {model_dir}")
    return shards


def load_wan_model(model_dir, repo: str, dtype=torch.bfloat16, device="cpu"):
    """Construct WanModel from config.json and assign the checkpoint weights.

    Tensors are streamed one at a time straight onto ``device``, so peak host
    RAM is a single tensor rather than a whole shard or a whole model. That
    matters: the 14B is ~28 GB in bf16 and diffusers' ``from_pretrained``
    materialises fp32 on CPU first (~57 GB), which OOMs any modestly sized
    allocation.
    """
    from safetensors import safe_open

    push_wan_repo(repo)
    from wan.modules.model import WanModel

    model_dir = Path(model_dir)
    cfg = read_model_config(model_dir)
    with torch.device("meta"):
        model = WanModel(**cfg)

    target = torch.device(device)
    missing = set(n for n, _ in model.named_parameters())
    missing |= set(n for n, _ in model.named_buffers())
    for shard in _shard_files(model_dir):
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            for key in f.keys():
                v = f.get_tensor(key)
                if v.is_floating_point():
                    v = v.to(dtype)
                _assign_tensor(model, key, v.to(target))
                missing.discard(key)
    if missing:
        raise RuntimeError(f"weights absent from checkpoint: {sorted(missing)[:8]}")

    _materialize_plain_tensors(model, cfg, target)
    model.eval().requires_grad_(False)
    return model, cfg


def _assign_tensor(model, dotted: str, tensor):
    """Replace a meta parameter/buffer in-place with a real tensor."""
    obj = model
    parts = dotted.split(".")
    for p in parts[:-1]:
        obj = getattr(obj, p)
    leaf = parts[-1]
    if isinstance(getattr(obj, leaf, None), torch.nn.Parameter):
        setattr(obj, leaf, torch.nn.Parameter(tensor, requires_grad=False))
    else:
        setattr(obj, leaf, tensor)


def _materialize_plain_tensors(model, cfg, device="cpu"):
    """Rebuild tensors held as plain attributes rather than buffers.

    ``WanModel.freqs`` is assigned directly (`self.freqs = ...`) with an
    explicit comment that register_buffer is avoided so ``.to()`` cannot change
    its dtype. That also means it is invisible to ``named_buffers()``, so
    constructing under ``torch.device('meta')`` leaves it dangling and the
    failure only surfaces deep inside ``forward``. Rebuild it here using the
    repo's own ``rope_params`` so the expression is never duplicated, and raise
    on any other meta attribute so a future addition fails loudly and early.
    """
    from wan.modules.model import rope_params

    dim, num_heads = cfg["dim"], cfg["num_heads"]
    d = dim // num_heads
    model.freqs = torch.cat([rope_params(1024, d - 4 * (d // 6)),
                             rope_params(1024, 2 * (d // 6)),
                             rope_params(1024, 2 * (d // 6))],
                            dim=1).to(device)

    stranded = [f"{name or '<root>'}.{attr}"
                for name, mod in model.named_modules()
                for attr, val in vars(mod).items()
                if torch.is_tensor(val) and val.is_meta]
    if stranded:
        raise RuntimeError(
            f"tensor attributes still on meta after load: {stranded[:8]}")


def load_wan_vae(kind: str, vae_path, device="cuda", dtype=torch.float32):
    """Frozen VAE. ``.encode(list_of[C,T,H,W]) -> list_of[Cz,Tz,Hz,Wz]``.

    Input must already be in [-1, 1]; the wrapper applies its own
    ``scale=[mean, 1/std]`` whitening internally, so latents must NOT be
    re-normalised downstream.
    """
    if kind == "wan21":
        push_wan_repo("wan21")
        from wan.modules.vae import WanVAE
        return WanVAE(z_dim=16, vae_pth=str(vae_path), dtype=dtype,
                      device=device)
    if kind == "wan22":
        push_wan_repo("wan22")
        from wan.modules.vae2_2 import Wan2_2_VAE
        return Wan2_2_VAE(z_dim=48, vae_pth=str(vae_path), dtype=dtype,
                          device=device)
    raise ValueError(f"unknown vae kind {kind!r}")


def load_t5(device="cuda", repo: str = "wan21", dtype=torch.bfloat16):
    """umT5-xxl encoder. Used once for prompt caching, then dropped."""
    push_wan_repo(repo)
    from wan.modules.t5 import T5EncoderModel
    return T5EncoderModel(
        text_len=512, dtype=dtype, device=device,
        checkpoint_path=str(T5_DIR / "models_t5_umt5-xxl-enc-bf16.pth"),
        tokenizer_path=str(T5_DIR / "google" / "umt5-xxl"))


def seq_len_of(latent_shape) -> int:
    """Wan token count for a [.., Cz, Tz, Hz, Wz] latent (patch (1,2,2))."""
    _, f, h, w = tuple(latent_shape)[-4:]
    return int(f * (h // 2) * (w // 2))
