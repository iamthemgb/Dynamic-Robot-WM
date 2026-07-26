"""Causal student: ordered Wan-VAE history -> 64-D belief (plan: Module B).

Pipeline:
  1. optional [Z; dZ] concat, causal-boundary masked (ablation, off by
     default — dZ is deterministic from Z);
  2. Conv3D stem with temporal kernel/stride 1 (spatial 2x2), factorized
     sinusoidal time/space positions;
  3. one learned spatial attention-pooling query per latent bin -> e_vis_t;
  4. control encoder: a small GRU over the control samples INSIDE each bin
     (state reset per bin, so within-bin impulse timing survives; mean
     pooling is explicitly not sufficient) -> e_ctrl_t;
  5. x_t = W_x [e_vis_t; e_ctrl_t] + p_t; two-layer causal GRU;
  6. deterministic belief head W_b -> per-bin beliefs [B, W, d_b].

Returning per-bin beliefs makes prefix curves, sliding-window recomputation
(Phase 4), and the causal-masking unit test one forward pass. Actions are
not optional: the same displacement can come from a weak push on a light
object or a strong push on a heavy one.
"""

import torch
import torch.nn as nn


def _sinusoidal(n, d, device=None):
    pos = torch.arange(n, device=device, dtype=torch.float32)[:, None]
    i = torch.arange(d // 2, device=device, dtype=torch.float32)[None]
    ang = pos / torch.pow(10000.0, 2 * i / d)
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


class PerBinControlEncoder(nn.Module):
    """e_ctrl_t = GRU over {(a_tau, s_tau) : tau in I(t)}, reset each bin."""

    def __init__(self, ctrl_dim, width=64):
        super().__init__()
        self.gru = nn.GRU(ctrl_dim, width, batch_first=True)
        self.width = width

    def forward(self, controls, bin_index, n_bins):
        """controls [B, Tc, d]; bin_index [Tc] -> [B, n_bins, width].

        Bins with no control samples (e.g. bin 0 under causal chunking)
        yield zeros.
        """
        B, Tc, d = controls.shape
        out = controls.new_zeros(B, n_bins, self.width)
        counts = torch.bincount(bin_index.clamp(min=0), minlength=n_bins)
        S = int(counts.max().item()) if Tc else 0
        if S == 0:
            return out
        buf = controls.new_zeros(B, n_bins, S, d)
        lengths = torch.zeros(n_bins, dtype=torch.long)
        for b in range(n_bins):
            sel = (bin_index == b).nonzero(as_tuple=True)[0]
            if len(sel):
                buf[:, b, :len(sel)] = controls[:, sel]
                lengths[b] = len(sel)
        h, _ = self.gru(buf.reshape(B * n_bins, S, d))
        h = h.reshape(B, n_bins, S, self.width)
        for b in range(n_bins):
            if lengths[b] > 0:
                out[:, b] = h[:, b, lengths[b] - 1]
        return out


class CausalStudent(nn.Module):
    def __init__(self, latent_channels, action_dim, state_dim=0,
                 belief_dim=64, width=256, gru_layers=2, gru_hidden=256,
                 ctrl_width=64, use_delta_z=False, dropout=0.0):
        super().__init__()
        self.belief_dim = belief_dim
        self.use_delta_z = use_delta_z
        d = width
        in_ch = latent_channels * (2 if use_delta_z else 1)
        self.stem = nn.Conv3d(in_ch, d, kernel_size=(1, 2, 2),
                              stride=(1, 2, 2))
        self.pool_query = nn.Parameter(torch.randn(1, 1, d) / d ** 0.5)
        self.pool = nn.MultiheadAttention(d, 4, dropout, batch_first=True)
        self.ctrl_enc = PerBinControlEncoder(action_dim + state_dim,
                                             ctrl_width)
        self.w_x = nn.Linear(d + ctrl_width, d)
        self.gru = nn.GRU(d, gru_hidden, num_layers=gru_layers,
                          batch_first=True, dropout=dropout)
        self.w_b = nn.Linear(gru_hidden, belief_dim)

    def forward(self, z_window, actions, bin_index, robot_state=None,
                delta_z=None, delta_valid=None):
        """
        z_window    [B, Cz, W, Hz, Wz]  ordered VAE latent window
        actions     [B, Tc, d_action]   control-rate executed actions
        bin_index   [Tc] long           control step -> latent bin (0..W-1)
        robot_state [B, Tc, d_state]    optional deployment-available state
        delta_z     [B, Cz, W, Hz, Wz]  optional temporal differences
        delta_valid [B, W]              1 where delta is comparable
        returns {"belief": [B, W, d_b], "belief_final": [B, d_b]}
        """
        B, C, W, H, Wd = z_window.shape
        x = z_window
        if self.use_delta_z:
            if delta_z is None:
                raise ValueError("use_delta_z=True requires delta_z")
            dz = delta_z * delta_valid[:, None, :, None, None]
            x = torch.cat([z_window, dz], dim=1)
        f = self.stem(x)                                  # [B, d, W, H', W']
        d, Hp, Wp = f.shape[1], f.shape[3], f.shape[4]
        f = f.movedim(1, -1)                              # [B, W, H', W', d]
        pos = (_sinusoidal(W, d, f.device)[:, None, None]
               + _sinusoidal(Hp, d, f.device)[None, :, None]
               + _sinusoidal(Wp, d, f.device)[None, None, :])
        f = f + pos[None]

        tok = f.reshape(B * W, Hp * Wp, d)
        q = self.pool_query.expand(B * W, 1, d)
        e_vis = self.pool(q, tok, tok, need_weights=False)[0]
        e_vis = e_vis.reshape(B, W, d)

        controls = actions if robot_state is None else torch.cat(
            [actions, robot_state], dim=-1)
        e_ctrl = self.ctrl_enc(controls, bin_index, W)

        x_t = self.w_x(torch.cat([e_vis, e_ctrl], dim=-1))
        x_t = x_t + _sinusoidal(W, x_t.shape[-1], x_t.device)[None]
        h, _ = self.gru(x_t)                              # causal by nature
        belief = self.w_b(h)
        return {"belief": belief, "belief_final": belief[:, -1]}
