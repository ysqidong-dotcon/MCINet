import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DWConv2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x, size_2d):
        T, B, C = x.shape
        H, W = size_2d
        x = x.permute(1, 2, 0).view(B, C, H, W)
        x = self.dw_conv(x)
        return x.view(B, C, -1).permute(2, 0, 1)

def silu(x):
    return x * torch.sigmoid(x)

class LongShortAttention(nn.Module):
    def __init__(self,
                 mode,                  # 'long' or 'short'
                 d_qk,
                 d_vu,
                 dropout=0.,
                 num_head=1,
                 expand_ratio=2.,
                 d_att=None,
                 use_linear=True,
                 max_mem_len_ratio=-1,
                 top_k=-1,
                 use_dis=False,
                 enable_corr=True,
                 max_dis=7,
                 dilation=1):
        super().__init__()
        assert mode in ['long', 'short'], "mode must be 'long' or 'short'"
        self.mode = mode
        self.use_linear = use_linear
        self.num_head = num_head
        self.use_dis = use_dis
        self.dropout = nn.Dropout(dropout)
        self.expand_d_vu = int(d_vu * expand_ratio)
        self.d_att = d_qk if d_att is None else d_att
        self.hidden_dim = self.expand_d_vu // num_head
        self.projection = nn.Linear(self.expand_d_vu, d_vu)
        self.dw_conv = DWConv2d(self.expand_d_vu)

        if mode == 'long':
            self.linear_Q = nn.Linear(d_qk, self.d_att)
            self.linear_K = nn.Linear(d_qk, self.d_att)
            self.linear_V = nn.Linear(d_vu, self.expand_d_vu)
            self.linear_U = nn.Linear(d_vu, self.expand_d_vu)
            self.max_mem_len_ratio = max_mem_len_ratio
            self.top_k = top_k

        elif mode == 'short':
            self.d_qk = d_qk
            self.d_vu = d_vu
            self.window_size = 2 * max_dis + 1
            self.max_dis = max_dis
            self.dilation = dilation
            self.enable_corr = enable_corr
            self.local_mask = None
            self.last_size_2d = None
            self.qk_mask = None

            if use_linear:
                self.linear_QK = nn.Conv2d(d_qk, self.d_att * num_head, kernel_size=1)
                self.linear_V = nn.Conv2d(d_vu, self.expand_d_vu, kernel_size=1, groups=2)
                self.linear_U = nn.Conv2d(d_vu, self.expand_d_vu, kernel_size=1, groups=2)

            self.relative_emb_k = nn.Conv2d(self.d_att * num_head,
                                            num_head * self.window_size * self.window_size,
                                            kernel_size=1,
                                            groups=num_head)

            if enable_corr:
                from spatial_correlation_sampler import SpatialCorrelationSampler
                self.correlation_sampler = SpatialCorrelationSampler(
                    kernel_size=1,
                    patch_size=self.window_size,
                    stride=1,
                    padding=0,
                    dilation=1,
                    dilation_patch=self.dilation)

    def forward(self, Q, K, V, U, size_2d):
        if self.mode == 'long':
            return self.forward_long(Q, K, V, U, size_2d)
        else:
            return self.forward_short(Q, K, V, U, size_2d)

    def forward_long(self, Q, K, V, U, size_2d):
        l, bs, _ = Q.size()

        if self.use_linear:
            Q = self.linear_Q(Q)
            K = self.linear_K(K)
            V = silu(self.linear_V(V))
            U = silu(self.linear_U(U))

        Q_ = Q.permute(1, 0, 2)  # [B, T, C]
        K_ = K.permute(1, 0, 2)
        V_ = V.permute(1, 0, 2)

        attn_output, attn_weights = F.scaled_dot_product_attention(Q_, K_, V_, dropout_p=self.dropout.p if self.training else 0.0)
        attn_output = attn_output.permute(1, 0, 2)
        output = attn_output * U

        output = self.dw_conv(output, size_2d)
        output = self.projection(output)
        return output, attn_weights

    def forward_short(self, q, k, v, u, size_2d):
        n, c, h, w = v.size()
        hidden_dim = self.hidden_dim

        if self.use_linear:
            q = k = self.linear_QK(q)
            v = silu(self.linear_V(v))
            u = silu(self.linear_U(u))
            if self.num_head > 1:
                v = v.view(-1, 2, self.num_head, hidden_dim // 2,
                           h * w).permute(0, 2, 1, 3, 4).reshape(n, -1, h, w)
                u = u.view(-1, 2, self.num_head, hidden_dim // 2,
                           h * w).permute(4, 0, 2, 1, 3).reshape(h * w, n, -1)
            else:
                u = u.permute(2, 3, 0, 1).reshape(h * w, n, -1)

        if self.qk_mask is None or (h, w) != self.last_size_2d:
            memory_mask = torch.ones((1, 1, h, w), device=v.device)
            unfolded_k_mask = self.pad_and_unfold(memory_mask).view(
                1, 1, self.window_size * self.window_size, h * w)
            self.qk_mask = 1 - unfolded_k_mask

        q = q / (self.d_att**0.5)
        relative_emb = self.relative_emb_k(q).view(n, self.num_head,
                                                   self.window_size * self.window_size,
                                                   h * w)

        q = q.view(-1, self.d_att, h, w)
        k = k.view(-1, self.d_att, h, w)
        v = v.view(-1, self.num_head, hidden_dim, h * w)

        if self.enable_corr:
            qk = self.correlation_sampler(q, k).view(
                n, self.num_head, self.window_size * self.window_size, h * w)
        else:
            unfolded_k = self.pad_and_unfold(k).view(
                n * self.num_head, hidden_dim,
                self.window_size * self.window_size, h, w)
            qk = (q.unsqueeze(2) * unfolded_k).sum(dim=1).view(
                n, self.num_head, self.window_size * self.window_size, h * w)

        if self.use_dis:
            qk = 2 * qk - self.pad_and_unfold(
                k.pow(2).sum(dim=1, keepdim=True)).view(
                    n, self.num_head, self.window_size * self.window_size,
                    h * w)

        qk += relative_emb
        qk -= self.qk_mask * 1e8

        attn = F.softmax(qk, dim=2)
        attn = self.dropout(attn)

        global_attn = self.local2global(attn, h, w)

        agg = (global_attn @ v.transpose(-2, -1)).permute(
            2, 0, 1, 3).reshape(h * w, n, -1)
        out = self.dw_conv(agg * u, size_2d)
        out = self.projection(out)

        self.last_size_2d = (h, w)
        return out, attn

    def pad_and_unfold(self, x):
        pad = self.max_dis * self.dilation
        x = F.pad(x, (pad, pad, pad, pad))
        return F.unfold(x,
                        kernel_size=(self.window_size, self.window_size),
                        stride=1,
                        dilation=self.dilation)

    def local2global(self, local_attn, h, w):
        B = local_attn.size(0)
        pad_h, pad_w = h + 2 * self.max_dis, w + 2 * self.max_dis

        if self.local_mask is None or (h, w) != self.last_size_2d:
            ky, kx = torch.meshgrid([
                torch.arange(0, pad_h, device=local_attn.device),
                torch.arange(0, pad_w, device=local_attn.device)
            ])
            qy, qx = torch.meshgrid([
                torch.arange(0, h, device=local_attn.device),
                torch.arange(0, w, device=local_attn.device)
            ])

            oy = qy.reshape(-1, 1) - ky.reshape(1, -1) + self.max_dis
            ox = qx.reshape(-1, 1) - kx.reshape(1, -1) + self.max_dis

            mask = (oy.abs() <= self.max_dis) & (ox.abs() <= self.max_dis)
            self.local_mask = mask.view(1, 1, h * w, pad_h, pad_w)

        global_attn = torch.zeros(
            (B, self.num_head, h * w, pad_h, pad_w),
            device=local_attn.device)
        global_attn[self.local_mask.expand(B, self.num_head, -1, -1, -1)] = local_attn.transpose(
            -1, -2).reshape(-1)

        return global_attn[:, :, :, self.max_dis:-self.max_dis,
                           self.max_dis:-self.max_dis].reshape(B, self.num_head, h * w, h * w)
