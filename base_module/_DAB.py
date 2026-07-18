import torch
import torch.nn as nn
from _wavelet import WTConv2d
from _basic import ARFB
from _time import SpatialTransformerBranch
from _frequency import FFTFreqBranch


class SAHBlock(nn.Module):
    """
    SAH Block with:
    - Frequency branch: FFTFreqBranch
    - Spatial branch: Transformer-based
    - Adaptive branch fusion
    """
    def __init__(self, c, res_scale=0.2,
                 hf_ratio=0.5,
                 num_heads=4,
                 window_size=8):
        super().__init__()
        self.res_scale = res_scale

        # shared preprocessing
        self.pre = ARFB(c)
        self.merge = nn.Conv2d(c, c, 3, 1, 1)

        # branches
        self.freq = FFTFreqBranch(c, hf_ratio=hf_ratio)
        self.spatial = SpatialTransformerBranch(
            c,
            num_heads=num_heads,
            window_size=window_size
        )

        # adaptive fusion
        mid = max(c // 8, 4)
        self.fusion_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c * 2, mid, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 2, 1)
        )
        self.softmax = nn.Softmax(dim=1)

        # post refinement
        self.post = ARFB(c)

    def forward(self, x):
        base = self.pre(x)
        base = self.merge(base)

        feat_freq = self.freq(base)
        feat_spatial = self.spatial(base)

        fusion_in = torch.cat([feat_freq, feat_spatial], dim=1)
        w = self.softmax(self.fusion_gate(fusion_in))  # [B,2,1,1]

        out = w[:, 0:1] * feat_freq + w[:, 1:2] * feat_spatial
        out = self.post(out)

        return x + self.res_scale * out