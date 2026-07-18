import torch
import torch.nn as nn
import torch.nn.functional as F
from _wavelet import WTConv2d
from _basic import ARFB

class LateFreqFusionWavelet(nn.Module):
    """
    Inject explicit high-frequency (wavelet-residual) from shallow feature into deep feature,
    gated by deep feature.
    """
    def __init__(self, c, reduction=8, wt_levels=1, wt_type="db1"):
        super().__init__()
        mid = max(c // reduction, 4)
        self.align_h = nn.Conv2d(c, c, 1)
        self.align_l = nn.Conv2d(c, c, 1)

        # explicit HF enhancement on shallow branch
        self.hf_wavelet = WTConv2d(
            in_channels=c, out_channels=c,
            kernel_size=3, stride=1,
            wt_levels=wt_levels, wt_type=wt_type
        )

        # gate generated from deep feature (structure-aware)
        self.gate = nn.Sequential(
            nn.Conv2d(c, mid, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, c, 3, 1, 1),
            nn.Sigmoid()
        )

        self.refine = ARFB(c)

    def forward(self, fh, fl):
        fh = self.align_h(fh)
        fl = self.align_l(fl)

        hf = self.hf_wavelet(fh)  # explicit wavelet HF residual enhancement
        g = self.gate(fl)

        out = fl + g * hf
        return self.refine(out)