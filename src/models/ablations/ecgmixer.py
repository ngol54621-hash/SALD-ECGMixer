import torch
from torch import nn

from src.models import ecgmixer as full_components

class ECGMixerAblation(nn.Module):
    use_rhythm = False
    use_fuse = False

    def __init__(self, in_ch=12, num_classes=5, lead_view_dim=32, fused_dim=128, num_heads=8, stage_ratios=(0.5, 1.0), stem_kernel=15, stage_kernels=(11, 7), mamba_d_state=64, mamba_d_conv=4, mamba_expand=2, mamba_headdim=64, drop=0.1, drop_path_rate=0.1):
        super().__init__()
        num_stages = len(stage_kernels)
        drop_path_rates = torch.linspace(0.0, drop_path_rate, num_stages + 1).tolist()
        self.num_leads = in_ch
        self.lead_view_dim = lead_view_dim
        self.local_dim = in_ch * lead_view_dim
        self.fused_dim = fused_dim
        self.combined_dim = self.local_dim + (fused_dim if self.use_fuse else 0)
        self.downsample_factor = 2 ** num_stages
        self.lead_backbone = full_components.LeadWiseBackbone(in_ch=in_ch, per_lead_dim=lead_view_dim, stage_ratios=stage_ratios, stem_kernel=stem_kernel, stage_kernels=stage_kernels, drop=drop * 0.5, drop_path_rates=drop_path_rates[:num_stages])
        self.lead_rhythm = full_components.PerLeadRhythmEncoder(dim=lead_view_dim, d_state=mamba_d_state, d_conv=mamba_d_conv, expand=mamba_expand, headdim=mamba_headdim, drop=drop, drop_path=drop_path_rates[-1]) if self.use_rhythm else None
        self.fuse = full_components.LFA(local_dim=self.local_dim, num_leads=in_ch, lead_view_dim=lead_view_dim, fused_dim=fused_dim, drop=drop) if self.use_fuse else None
        self.head = full_components.UnifiedMeanMaxHead(dim=self.combined_dim, num_classes=num_classes, drop=drop)
        initialized_modules = [self.lead_backbone, self.head]
        if self.fuse is not None:
            initialized_modules.append(self.fuse)
        for module in initialized_modules:
            module.apply(full_components._init_weights)

    def forward_features(self, x):
        leadwise = self.lead_backbone(x)
        if self.lead_rhythm is not None:
            leadwise = self.lead_rhythm(leadwise)
        local = leadwise.flatten(start_dim=2)
        fused = self.fuse(local) if self.fuse is not None else None
        complementary = fused
        combined = torch.cat([local, complementary], dim=-1) if complementary is not None else local
        return combined

    def forward(self, x):
        return self.head(self.forward_features(x))

class LocalEncoder(ECGMixerAblation):
    pass

class RhythmEncoder(ECGMixerAblation):
    use_rhythm = True

class LeadAttention(ECGMixerAblation):
    use_rhythm = True
    use_fuse = True

class LeadAddition(nn.Module):

    def __init__(self, local_dim, num_leads, lead_view_dim, fused_dim=128, drop=0.1):
        super().__init__()
        self.num_leads = num_leads
        self.lead_view_dim = lead_view_dim
        self.input_dim = local_dim
        self.output_dim = fused_dim
        self.proj = nn.Linear(lead_view_dim, fused_dim)
        self.out_norm = nn.LayerNorm(fused_dim)
        self.activation = nn.GELU()
        self.drop = nn.Dropout(drop) if drop > 0.0 else nn.Identity()

    def forward(self, local):
        (batch, tokens, width) = local.shape
        leads = local.reshape(batch, tokens, self.num_leads, self.lead_view_dim)
        summed = leads.sum(dim=2)
        return self.drop(self.activation(self.out_norm(self.proj(summed))))

class AdditiveFusion(full_components.ECGMixer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fuse = LeadAddition(local_dim=self.local_dim, num_leads=self.num_leads, lead_view_dim=self.lead_view_dim, fused_dim=self.fused_dim, drop=kwargs.get('drop', 0.1))
        self.fuse.apply(full_components._init_weights)
