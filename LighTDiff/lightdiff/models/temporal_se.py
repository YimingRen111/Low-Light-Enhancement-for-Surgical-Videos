# ========== TemporalSE ==========
import torch
import torch.nn as nn

class TemporalSE(nn.Module):
    """
    输入:  x ∈ [B, T, C, H, W] (T=5等)
    输出:  y ∈ [B, C, H, W]    (按时间加权融合后的单帧)
    机制:  每帧做GAP→MLP打分→softmax→时间加权和→1x1微调
    """
    def __init__(self, channels=3, t=5, hidden=16):
        super().__init__()
        self.t = t
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1)  # 每帧一个logit
        )
        self.post = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):  # x: [B,T,C,H,W]
        B, T, C, H, W = x.shape
        g = x.mean(dim=(-1, -2))              # [B,T,C]
        logits = self.mlp(g).squeeze(-1)      # [B,T]
        w = torch.softmax(logits, dim=1)      # [B,T]
        y = (x * w.view(B, T, 1, 1, 1)).sum(dim=1)  # [B,C,H,W]
        return self.post(y)