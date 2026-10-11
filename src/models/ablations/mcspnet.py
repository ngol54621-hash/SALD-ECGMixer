import torch
from torch import nn

from src.models.blocks import BiMamba2
from src.models.mcspnet import (
    MCSPNet, MSCM, ConditionEmbedder, FinalLayer1D, PeriodFold, modulate,
)


class SSPMAblation(nn.Module):
    def __init__(self, dim, top_k, p_min, p_max, variant):
        super().__init__()
        self.use_period_fold = variant != "without_period_fold"
        self.fold = PeriodFold(dim, top_k=top_k, p_min=p_min, p_max=p_max)
        self.ssm = None if variant == "without_bimamba2" else BiMamba2(dim)

    def forward(self, x):
        y = self.fold(x) if self.use_period_fold else torch.zeros_like(x)
        if self.ssm is not None:
            y = y + self.ssm(x)
        return y


class MCSPNetAblationBlock(nn.Module):
    def __init__(self, dim, mlp_ratio, dropout, top_k, p_min, p_max, variant):
        super().__init__()
        self.use_mscm = variant != "without_mscm"
        self.use_sspm = variant != "without_sspm"
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.morph = MSCM(dim)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.rhythm = SSPMAblation(dim, top_k, p_min, p_max, variant)
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout),
        )
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 9 * dim))

    def forward(self, x, c):
        shift1, scale1, gate1, shift2, scale2, gate2, shift3, scale3, gate3 = self.adaLN(c).chunk(9, dim=-1)
        if self.use_mscm:
            x = x + gate1.unsqueeze(1) * self.morph(modulate(self.norm1(x), shift1, scale1))
        if self.use_sspm:
            x = x + gate2.unsqueeze(1) * self.rhythm(modulate(self.norm2(x), shift2, scale2))
        return x + gate3.unsqueeze(1) * self.mlp(modulate(self.norm3(x), shift3, scale3))


class MCSPNetAblation(MCSPNet):
    def __init__(self, cfg):
        nn.Module.__init__(self)
        d = cfg.ldm.mcspnet
        self.latent_channels = int(cfg.vae.z_channels)
        self.latent_len = int(cfg.data.signal_length // (2 ** (len(cfg.vae.ch_mult) - 1)))
        self.dim = int(d.dim)
        self.in_proj = nn.Linear(self.latent_channels, self.dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.latent_len, self.dim))
        self.cond = ConditionEmbedder(cfg, self.dim)
        self.blocks = nn.ModuleList([
            MCSPNetAblationBlock(self.dim, d.mlp_ratio, d.dropout, d.top_k,
                                d.p_min, d.p_max, cfg.ldm.ablation_variant)
            for _ in range(d.depth)
        ])
        self.final = FinalLayer1D(self.dim, self.latent_channels)
        self._init_weights()
