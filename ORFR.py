import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualFusion(nn.Module):
    def __init__(self, in_channels):
        super(ResidualFusion, self).__init__()
        self.conv1 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.position_embed = nn.Parameter(torch.randn(1, in_channels, 1, 1))

    def forward(self, local_feat, global_feat):
        x = torch.cat([local_feat+self.position_embed, global_feat], dim=1)  # [B, 2C, H, W]
        residual = self.conv2(self.relu(self.conv1(x)))
        return residual + local_feat


class AgentAttention(nn.Module):
    def __init__(self, in_channels):
        super(AgentAttention, self).__init__()
        self.q_linear = nn.Linear(in_channels, in_channels)
        self.k_linear = nn.Linear(in_channels, in_channels)
        self.v_linear = nn.Linear(in_channels, in_channels)
        self.agent_bias1 = nn.Parameter(torch.zeros(1))
        self.agent_bias2 = nn.Parameter(torch.zeros(1))
        self.dwconv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)

    def forward(self, FM):
        B, C, H, W = FM.shape
        FM_flat = FM.view(B, C, -1).permute(0, 2, 1)
        pooled_q = F.adaptive_max_pool2d(FM, (1, 1)).view(B, C)
        q = self.q_linear(pooled_q)
        k = self.k_linear(FM_flat)
        attn_weights1 = torch.bmm(k, q.unsqueeze(2)).squeeze(2) / (C ** 0.5)
        attn_weights1 = F.softmax(attn_weights1, dim=1).unsqueeze(2)
        v = self.v_linear(FM_flat)
        PV = (attn_weights1 * v).sum(dim=1) + self.agent_bias1
        pooled_k = self.k_linear(pooled_q)
        q2 = self.q_linear(FM_flat)
        attn_weights2 = torch.bmm(q2, pooled_k.unsqueeze(2)).squeeze(2) / (C ** 0.5)
        attn_weights2 = F.softmax(attn_weights2, dim=1).unsqueeze(2)
        PV_prime = (attn_weights2 * PV.unsqueeze(1)).sum(dim=1) + self.agent_bias2
        FM_enhanced = self.dwconv(FM) + PV_prime.unsqueeze(2).unsqueeze(3)
        return FM_enhanced


class AMCFeatureIntegrator(nn.Module):
    def __init__(self, in_channels):
        super(AMCFeatureIntegrator, self).__init__()
        self.fusion = ResidualFusion(in_channels)
        self.agent_attn = AgentAttention(in_channels)
        self.position_embed = nn.Parameter(torch.randn(1, in_channels, 1, 1))
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=False)

    def forward(self, local_feat, global_feat):
        global_feat = global_feat+self.position_embed
        FM = self.fusion(local_feat, global_feat)
        M_att = self.agent_attn(FM)
        M_combined = M_att + global_feat
        M_final = self.conv1x1(M_combined)
        M_final_upsampled = self.upsample(M_final)
        return M_final_upsampled
