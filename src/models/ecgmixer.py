import math
import torch
import torch.nn as nn

class DropPath(nn.Module):

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

class LeadWiseResidualStage(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, num_leads, drop=0.0, drop_path=0.0):
        super().__init__()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.main = nn.Sequential(nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2, groups=num_leads, bias=False), nn.BatchNorm1d(out_ch))
        self.shortcut = nn.Identity() if in_ch == out_ch else nn.Sequential(nn.Conv1d(in_ch, out_ch, kernel_size=1, groups=num_leads, bias=False), nn.BatchNorm1d(out_ch))
        self.feature_drop = nn.Dropout1d(drop) if drop > 0.0 else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.pool(x)
        residual = self.drop_path(self.feature_drop(self.main(x)))
        return self.activation(self.shortcut(x) + residual)

class LeadWiseBackbone(nn.Module):

    def __init__(self, in_ch=12, per_lead_dim=32, stage_ratios=(0.5, 1.0), stem_kernel=15, stage_kernels=(11, 7), drop=0.0, drop_path_rates=None):
        super().__init__()
        if drop_path_rates is None:
            drop_path_rates = (0.0,) * len(stage_kernels)
        self.num_leads = in_ch
        self.groups = in_ch
        self.out_dim = per_lead_dim
        first_dim = max(1, int(round(per_lead_dim * stage_ratios[0])))
        stem_ch = in_ch * first_dim
        self.stem = nn.Sequential(nn.Conv1d(in_ch, stem_ch, stem_kernel, padding=stem_kernel // 2, groups=in_ch, bias=False), nn.BatchNorm1d(stem_ch), nn.GELU())
        stages = []
        current_ch = stem_ch
        for (index, (ratio, kernel, drop_path)) in enumerate(zip(stage_ratios, stage_kernels, drop_path_rates)):
            stage_dim = per_lead_dim if index == len(stage_ratios) - 1 else max(1, int(round(per_lead_dim * ratio)))
            stage_ch = in_ch * stage_dim
            stages.append(LeadWiseResidualStage(current_ch, stage_ch, kernel, num_leads=in_ch, drop=drop, drop_path=drop_path))
            current_ch = stage_ch
        self.stages = nn.Sequential(*stages)

    def forward(self, x):
        x = self.stages(self.stem(x))
        (batch, _, length) = x.shape
        return x.reshape(batch, self.num_leads, self.out_dim, length).permute(0, 3, 1, 2).contiguous()

class BiMamba2(nn.Module):

    def __init__(self, dim, d_state=64, d_conv=4, expand=2, headdim=64):
        super().__init__()
        from mamba_ssm import Mamba2 as Mamba2Core
        kwargs = {'d_model': dim, 'd_state': d_state, 'd_conv': d_conv, 'expand': expand, 'headdim': headdim}
        self.forward_mamba = Mamba2Core(**kwargs)
        self.backward_mamba = Mamba2Core(**kwargs)

    def forward(self, x):
        forward_features = self.forward_mamba(x)
        backward_features = self.backward_mamba(x.flip(1)).flip(1)
        return forward_features + backward_features

class PerLeadRhythmEncoder(nn.Module):

    def __init__(self, dim, d_state=64, d_conv=4, expand=2, headdim=64, drop=0.1, drop_path=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.bimamba = BiMamba2(dim=dim, d_state=d_state, d_conv=d_conv, expand=expand, headdim=headdim)
        self.drop = nn.Dropout(drop) if drop > 0.0 else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        (batch, tokens, leads, dim) = x.shape
        sequence = x.permute(0, 2, 1, 3).reshape(batch * leads, tokens, dim)
        rhythm = self.bimamba(self.norm(sequence))
        sequence = sequence + torch.sigmoid(self.residual_scale) * self.drop_path(self.drop(rhythm))
        return sequence.reshape(batch, leads, tokens, dim).permute(0, 2, 1, 3).contiguous()

class LFA(nn.Module):

    def __init__(self, local_dim, num_leads, lead_view_dim, fused_dim=128, num_heads=4, drop=0.1):
        super().__init__()
        self.num_leads = num_leads
        self.lead_view_dim = lead_view_dim
        self.input_dim = local_dim
        self.output_dim = fused_dim
        self.norm = nn.LayerNorm(lead_view_dim)
        self.attn = nn.MultiheadAttention(embed_dim=lead_view_dim, num_heads=num_heads, dropout=drop, batch_first=True)
        self.proj = nn.Linear(local_dim, fused_dim)
        self.out_norm = nn.LayerNorm(fused_dim)
        self.activation = nn.GELU()
        self.drop = nn.Dropout(drop) if drop > 0.0 else nn.Identity()

    def forward(self, local):
        (batch, tokens, _) = local.shape
        leads = local.reshape(batch * tokens, self.num_leads, self.lead_view_dim)
        leads_norm = self.norm(leads)
        (attended, _) = self.attn(leads_norm, leads_norm, leads_norm)
        leads = leads + attended
        flat = leads.reshape(batch, tokens, self.num_leads * self.lead_view_dim)
        return self.drop(self.activation(self.out_norm(self.proj(flat))))

class CSDA(nn.Module):

    def __init__(self, local_dim, fused_dim=128, num_heads=8, drop=0.1, lambda_max=0.5, lambda_init=0.1, residual_scale_init=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = fused_dim // num_heads
        self.scale = self.head_dim ** (-0.5)
        self.lambda_max = float(lambda_max)
        self.local_norm = nn.LayerNorm(local_dim)
        self.fused_norm = nn.LayerNorm(fused_dim)
        self.q_local = nn.Linear(local_dim, fused_dim)
        self.k_local = nn.Linear(local_dim, fused_dim)
        self.q_fused = nn.Linear(fused_dim, fused_dim)
        self.k_fused = nn.Linear(fused_dim, fused_dim)
        self.v_fused = nn.Linear(fused_dim, fused_dim)
        init_ratio = lambda_init / lambda_max
        init_logit = math.log(init_ratio / (1.0 - init_ratio))
        self.lambda_param = nn.Parameter(torch.full((num_heads,), init_logit))
        self.attn_drop = nn.Dropout(drop) if drop > 0.0 else nn.Identity()
        self.out_proj = nn.Linear(fused_dim, fused_dim)
        self.out_drop = nn.Dropout(drop) if drop > 0.0 else nn.Identity()
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))
        self.out_norm = nn.LayerNorm(fused_dim)

    def _split_heads(self, x):
        (batch, tokens, _) = x.shape
        return x.reshape(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x):
        (batch, heads, tokens, head_dim) = x.shape
        return x.transpose(1, 2).contiguous().reshape(batch, tokens, heads * head_dim)

    def forward(self, local, fused):
        local = self.local_norm(local)
        fused_norm = self.fused_norm(fused)
        q_local = self._split_heads(self.q_local(local))
        k_local = self._split_heads(self.k_local(local))
        q_fused = self._split_heads(self.q_fused(fused_norm))
        k_fused = self._split_heads(self.k_fused(fused_norm))
        v_fused = self._split_heads(self.v_fused(fused_norm))
        local_scores = torch.matmul(q_local, k_local.transpose(-2, -1)) * self.scale
        fused_scores = torch.matmul(q_fused, k_fused.transpose(-2, -1)) * self.scale
        local_attention = torch.softmax(local_scores.float(), dim=-1).to(local_scores.dtype)
        fused_attention = torch.softmax(fused_scores.float(), dim=-1).to(fused_scores.dtype)
        local_attention = self.attn_drop(local_attention)
        fused_attention = self.attn_drop(fused_attention)
        lambda_per_head = self.lambda_max * torch.sigmoid(self.lambda_param)
        lambda_per_head = lambda_per_head.view(1, self.num_heads, 1, 1)
        diff = torch.relu(fused_attention - lambda_per_head * local_attention)
        differential_attention = diff / (diff.sum(dim=-1, keepdim=True) + 1e-06)
        complementary = torch.matmul(differential_attention, v_fused)
        complementary = self._merge_heads(complementary)
        complementary = self.out_drop(self.out_proj(complementary))
        output = self.out_norm(fused + torch.sigmoid(self.residual_scale) * complementary)
        return output

class UnifiedMeanMaxHead(nn.Module):

    def __init__(self, dim, num_classes, drop=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(drop) if drop > 0.0 else nn.Identity()
        self.classifier = nn.Linear(dim * 2, num_classes)

    def pooled_features(self, x):
        x = self.norm(x)
        return torch.cat([x.mean(dim=1), x.amax(dim=1)], dim=-1)

    def forward(self, x):
        return self.classifier(self.drop(self.pooled_features(x)))

class ECGMixer(nn.Module):

    def __init__(self, in_ch=12, num_classes=5, lead_view_dim=32, fused_dim=128, num_heads=8, stage_ratios=(0.5, 1.0), stem_kernel=15, stage_kernels=(11, 7), mamba_d_state=64, mamba_d_conv=4, mamba_expand=2, mamba_headdim=64, differential_lambda_max=0.5, differential_lambda_init=0.1, drop=0.1, drop_path_rate=0.1):
        super().__init__()
        num_stages = len(stage_kernels)
        drop_path_rates = torch.linspace(0.0, drop_path_rate, num_stages + 1).tolist()
        self.num_leads = in_ch
        self.lead_view_dim = lead_view_dim
        self.local_dim = in_ch * lead_view_dim
        self.fused_dim = fused_dim
        self.combined_dim = self.local_dim + fused_dim
        self.downsample_factor = 2 ** num_stages
        self.lead_backbone = LeadWiseBackbone(in_ch=in_ch, per_lead_dim=lead_view_dim, stage_ratios=stage_ratios, stem_kernel=stem_kernel, stage_kernels=stage_kernels, drop=drop * 0.5, drop_path_rates=drop_path_rates[:num_stages])
        self.lead_rhythm = PerLeadRhythmEncoder(dim=lead_view_dim, d_state=mamba_d_state, d_conv=mamba_d_conv, expand=mamba_expand, headdim=mamba_headdim, drop=drop, drop_path=drop_path_rates[-1])
        self.fuse = LFA(local_dim=self.local_dim, num_leads=in_ch, lead_view_dim=lead_view_dim, fused_dim=fused_dim, drop=drop)
        self.differential_attention = CSDA(local_dim=self.local_dim, fused_dim=fused_dim, num_heads=num_heads, drop=drop, lambda_max=differential_lambda_max, lambda_init=differential_lambda_init)
        self.head = UnifiedMeanMaxHead(dim=self.combined_dim, num_classes=num_classes, drop=drop)
        for module in (self.lead_backbone, self.fuse, self.differential_attention, self.head):
            module.apply(_init_weights)

    def forward_features(self, x):
        leadwise = self.lead_rhythm(self.lead_backbone(x))
        local = leadwise.flatten(start_dim=2)
        fused = self.fuse(local)
        complementary = self.differential_attention(local, fused)
        combined = torch.cat([local, complementary], dim=-1)
        return combined

    def forward(self, x):
        return self.head(self.forward_features(x))

def _init_weights(module):
    if isinstance(module, nn.Conv1d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
