"""V3 -- the blocking gate on flow-matching sign convention.

The mock (``wan/mock_wan.py::flow_sample``) and real Wan use OPPOSITE
conventions:

    mock : x_tau = (1-tau)*eps + tau*z1 ,  v* = z1 - eps    (tau=1 -> data)
    wan  : x_t   = (1-sig)*x0  + sig*eps,  v  = eps - x0    (sig=1 -> noise)

If ``real/flow_match.py`` got this backwards the campaign still *runs*: the
loss simply plateaus near 2x its proper value and no conditioning gap ever
opens. Nothing crashes, so this must be checked explicitly.

The frozen pretrained model is the oracle: it was trained to predict
``eps - x0``, so scoring it against that target must beat the negated target
by a wide margin.

    python -m generalized_physics.real.tests.test_flow_convention
"""

import sys

import numpy as np
import torch

from .. import wan_loader as W
from ..flow_match import flow_loss, make_targets, sample_sigmas, shift_sigma
from ..paths import ARMS, DATASET_ROOT, N_FRAMES

SAMPLE_MP4 = (DATASET_ROOT / "F1a" / "block-0000" / "videos" /
              "observation.images.main" / "chunk-000" / "file-000000.mp4")


def real_latent(arm, device="cuda"):
    import av
    c = av.open(str(SAMPLE_MP4))
    frames = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
    c.close()
    x = torch.from_numpy(np.stack(frames[:N_FRAMES]))
    x = x.permute(3, 0, 1, 2).float().div_(127.5).sub_(1.0)
    vae = W.load_wan_vae(arm.vae_kind, arm.vae_path, device=device)
    z = vae.encode([x.to(device)])[0].float()
    del vae
    torch.cuda.empty_cache()
    return z


def main(arm_name="wan21_t2v_1p3b"):
    arm = ARMS[arm_name]
    device = "cuda"
    torch.manual_seed(0)

    x0 = real_latent(arm, device)[None]                 # [1, Cz, Tz, Hz, Wz]
    model, cfg = W.load_wan_model(arm.model_dir, arm.repo, torch.bfloat16)
    model = model.to(device)
    seq_len = W.seq_len_of(x0.shape)
    assert cfg["in_dim"] == arm.latent_channels, (
        f"in_dim {cfg['in_dim']} != latent channels {arm.latent_channels}")

    # Unconditional text slot: WanModel expects a list of [L, text_dim].
    ctx = [torch.zeros(1, cfg["text_dim"] if "text_dim" in cfg else 4096,
                       device=device, dtype=torch.bfloat16)]

    ok = True
    for sigma in (0.3, 0.6, 0.9):
        sig = torch.full((1,), sigma)
        x_t, v_target, t, eps = make_targets(x0, sig)
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            pred = model(x=list(x_t), t=t.to(device), context=ctx,
                         seq_len=seq_len)
        l_correct = float(flow_loss(pred, v_target))     # eps - x0
        l_flipped = float(flow_loss(pred, -v_target))    # x0 - eps
        good = l_correct < l_flipped
        ok &= good
        print(f"  sigma={sigma:.1f}  L(eps-x0)={l_correct:.4f}   "
              f"L(x0-eps)={l_flipped:.4f}   ratio={l_flipped/l_correct:.2f}x  "
              f"{'OK' if good else 'WRONG SIGN'}")

    # shift_sigma must be monotone and fix the endpoints
    u = torch.linspace(0, 1, 11)
    s = shift_sigma(u, 5.0)
    assert torch.all(s[1:] > s[:-1]), "shift_sigma not monotone"
    assert abs(float(s[0])) < 1e-6 and abs(float(s[-1]) - 1.0) < 1e-6
    assert sample_sigmas(64, 5.0).max() <= 1.0

    print("VERDICT", "PASS" if ok else "FAIL -- flow_match.py sign is inverted")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
