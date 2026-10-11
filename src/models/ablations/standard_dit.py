import torch
from torch import nn

from src.models.mcspnet import ConditionEmbedder, FinalLayer1D, modulate

class StandardDiTBlock1D(nn.Module):

    def __init__(self, dim, heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-06)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-06)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(approximate='tanh'), nn.Dropout(dropout), nn.Linear(hidden, dim), nn.Dropout(dropout))
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(self, x, c):
        (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) = self.adaLN(c).chunk(6, dim=-1)
        h = modulate(self.norm1(x), shift_msa, scale_msa)
        (h, _) = self.attn(h, h, h, need_weights=False)
        x = x + gate_msa.unsqueeze(1) * h
        h = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(h)
        return x

class StandardDiT1D(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        d = cfg.ldm.get('standard_dit', {})
        self.latent_channels = int(cfg.vae.z_channels)
        self.latent_len = int(cfg.data.signal_length // 2 ** (len(cfg.vae.ch_mult) - 1))
        self.dim = int(d.get('dim', 256))
        self.depth = int(d.get('depth', 6))
        self.heads = int(d.get('heads', 8))
        self.mlp_ratio = float(d.get('mlp_ratio', 4.0))
        self.dropout = float(d.get('dropout', 0.0))
        self.in_proj = nn.Linear(self.latent_channels, self.dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.latent_len, self.dim))
        self.cond = ConditionEmbedder(cfg, self.dim)
        self.blocks = nn.ModuleList([StandardDiTBlock1D(self.dim, self.heads, self.mlp_ratio, self.dropout) for _ in range(self.depth)])
        self.final = FinalLayer1D(self.dim, self.latent_channels)
        self._init_weights()

    def _init_weights(self):

        def basic(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(basic)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.in_proj.weight, std=0.02)
        if self.in_proj.bias is not None:
            nn.init.zeros_(self.in_proj.bias)
        for block in self.blocks:
            nn.init.xavier_uniform_(block.attn.in_proj_weight)
            if block.attn.in_proj_bias is not None:
                nn.init.zeros_(block.attn.in_proj_bias)
            nn.init.constant_(block.adaLN[-1].weight, 0)
            nn.init.constant_(block.adaLN[-1].bias, 0)
        nn.init.constant_(self.final.adaLN[-1].weight, 0)
        nn.init.constant_(self.final.adaLN[-1].bias, 0)
        nn.init.constant_(self.final.linear.weight, 0)
        nn.init.constant_(self.final.linear.bias, 0)

    def forward(self, x, t, labels):
        h = self.in_proj(x.transpose(1, 2)) + self.pos_embed
        c = self.cond(t, labels)
        for block in self.blocks:
            h = block(h, c)
        return self.final(h, c).transpose(1, 2)
