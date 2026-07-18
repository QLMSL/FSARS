import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace
from collections import OrderedDict


class SpatialTransformerBranch(nn.Module):
    """
    Spatial branch based on windowed self-attention.
    Keeps spatial resolution and channel dimension.
    """
    def __init__(self, c, num_heads=4, window_size=8):
        super().__init__()
        self.c = c
        self.window_size = window_size

        self.norm1 = nn.LayerNorm(c)
        self.attn = nn.MultiheadAttention(
            embed_dim=c,
            num_heads=num_heads,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(c)
        self.ffn = nn.Sequential(
            nn.Linear(c, c * 4),
            nn.GELU(),
            nn.Linear(c * 4, c)
        )

    def forward(self, x):
        """
        x: [B, C, H, W]
        """
        B, C, H, W = x.shape
        ws = self.window_size

        # pad to multiple of window size
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        x = F.pad(x, (0, pad_w, 0, pad_h))
        Hp, Wp = x.shape[-2:]

        # reshape to windows
        x = x.view(B, C, Hp // ws, ws, Wp // ws, ws)
        x = x.permute(0, 2, 4, 3, 5, 1)  # [B, nh, nw, ws, ws, C]
        x = x.reshape(-1, ws * ws, C)    # [B*num_windows, N, C]

        # attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # FFN
        x = x + self.ffn(self.norm2(x))

        # restore spatial layout
        x = x.view(B, Hp // ws, Wp // ws, ws, ws, C)
        x = x.permute(0, 5, 1, 3, 2, 4)
        x = x.reshape(B, C, Hp, Wp)

        # crop padding
        return x[:, :, :H, :W]
