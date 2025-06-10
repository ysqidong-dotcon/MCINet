import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F

class FrequencyGuidedEnhancement(nn.Module):
    def __init__(self, channels):
        super(FrequencyGuidedEnhancement, self).__init__()
        self.channels = channels
        self.pool1 = nn.AvgPool2d(kernel_size=1)
        self.pool2 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
        self.guidance_conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.center_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1)
        )
        self.mu = nn.Parameter(torch.tensor(0.5))
        self.delta = nn.Parameter(torch.tensor(0.2))

    def forward(self, x):
        B, C, H, W = x.shape
        x_fft = torch.fft.fft2(x, norm='ortho')
        real = x_fft.real
        imag = x_fft.imag
        r = x.abs().mean(dim=1, keepdim=True)  # [B, 1, H, W]
        p1 = self.pool1(r)
        p2 = self.pool2(r)
        p3 = self.pool3(r)
        g_input = torch.cat([p1, p2, p3], dim=1)
        G = self.guidance_conv(g_input)  # [B, 1, H, W]
        fG = self.center_conv(G)  # [B, 1, H, W]
        sigma = torch.exp(-((fG - self.mu) ** 2) / (2 * self.delta ** 2))  # [B, 1, H, W]
        sigma = sigma.expand(-1, C, -1, -1)  # broadcast 到所有通道
        real_mod = real * sigma
        imag_mod = imag * sigma
        x_fft_mod = torch.complex(real_mod, imag_mod)
        x_recon = torch.fft.ifft2(x_fft_mod, norm='ortho').real  # 取实部
        out = x + x_recon
        return out


class TokenWiseTextVisualInteraction(nn.Module):
    def __init__(self, visual_dim, text_dim, shared_dim=256):
        super().__init__()
        self.visual_proj = nn.Linear(visual_dim, shared_dim)  # W1
        self.text_proj = nn.Linear(text_dim, shared_dim)      # W2

        self.fusion_mlp = nn.Sequential(
            nn.Linear(visual_dim + shared_dim, visual_dim),
            nn.ReLU(),
            nn.Linear(visual_dim, visual_dim)
        )

    def forward(self, M_pre, L):
        """
        M_pre: [B, C, H, W]
        L:     [B, C_l, N]
        return: [B, C, H, W]
        """
        B, C, H, W = M_pre.shape
        _, C_l, N = L.shape
        HW = H * W
        M_flat = M_pre.view(B, C, HW).permute(0, 2, 1)       # [B, HW, C]
        M_proj = self.visual_proj(M_flat)                    # [B, HW, D]
        L_flat = L.permute(0, 2, 1)                          # [B, N, C_l]
        L_proj = self.text_proj(L_flat)                     # [B, N, D]
        D = M_proj.size(-1)
        A = torch.bmm(L_proj, M_proj.transpose(1, 2)) / (D ** 0.5)  # [B, N, HW]
        A = F.softmax(A, dim=-1)
        M_L = torch.bmm(A.transpose(1, 2), L_proj)  # [B, HW, D]
        fusion_input = torch.cat([M_proj, M_L], dim=-1)  # [B, HW, C + D]
        fused = self.fusion_mlp(fusion_input)            # [B, HW, C]
        fused = fused + M_flat
        fused = fused.permute(0, 2, 1).view(B, C, H, W)
        return fused

class SentenceGuidedVisualInteractionLite(nn.Module):
    def __init__(self, visual_dim, text_dim, shared_dim=256, down_ratio=4):
        super().__init__()
        self.down_ratio = down_ratio
        self.shrink = nn.Conv2d(visual_dim, visual_dim, kernel_size=down_ratio, stride=down_ratio)
        self.expand = nn.ConvTranspose2d(visual_dim, visual_dim, kernel_size=down_ratio, stride=down_ratio)

        self.visual_proj = nn.Linear(visual_dim, shared_dim)
        self.text_proj = nn.Linear(text_dim, shared_dim)

        self.fusion_mlp = nn.Sequential(
            nn.Linear(visual_dim + shared_dim, visual_dim),
            nn.ReLU(),
            nn.Linear(visual_dim, visual_dim)
        )

    def forward(self, M_pre, L):
        """
        M_pre: [B, C, H, W]
        L:     [B, C_l]
        return: [B, C, H, W]
        """
        B, C, H, W = M_pre.shape
        Hs, Ws = H // self.down_ratio, W // self.down_ratio

        V_down = self.shrink(M_pre)                         # [B, C, Hs, Ws]
        M_flat = V_down.view(B, C, Hs * Ws).permute(0, 2, 1)  # [B, Hs*Ws, C]
        M_proj = self.visual_proj(M_flat)                    # [B, Hs*Ws, D]

        L_proj = self.text_proj(L).unsqueeze(1)              # [B, 1, D]

        A = torch.bmm(M_proj, L_proj.transpose(1, 2))        # [B, Hs*Ws, 1]
        A = F.softmax(A / (M_proj.size(-1) ** 0.5), dim=1)
        L_broadcasted = L_proj.expand(-1, Hs * Ws, -1)       # [B, Hs*Ws, D]
        M_L = A * L_broadcasted                              # [B, Hs*Ws, D]

        fusion_input = torch.cat([M_flat, M_L], dim=-1)      # [B, Hs*Ws, C+D]
        fused = self.fusion_mlp(fusion_input)                # [B, Hs*Ws, C]
        fused = fused + M_flat                               #
        fused = fused.permute(0, 2, 1).view(B, C, Hs, Ws)     # [B, C, Hs, Ws]

        out = self.expand(fused)                             # [B, C, H, W]
        return out
