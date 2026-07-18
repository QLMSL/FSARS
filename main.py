import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace
from collections import OrderedDict
from base_module._DAB import SAHBlock
from base_module._basic import ARFB
from base_module._WSCB import LateFreqFusionWavelet

class StageAgg(nn.Module):
    """Aggregate multi-stage features (DenseNet-style)"""
    def __init__(self, c, n_inputs):
        super().__init__()
        self.compress = nn.Conv2d(c * n_inputs, c, 1)
        self.refine = ARFB(c)

    def forward(self, feats):
        x = torch.cat(feats, dim=1)
        x = self.compress(x)
        x = self.refine(x)
        return x


class SA_adapt_v2(nn.Module):
    """Scale-adaptive spatial modulation SAFE for arbitrary H×W"""
    def __init__(self, c):
        super().__init__()
        self.mask_net = nn.Sequential(
            nn.Conv2d(c, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, ceil_mode=True),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x, scale, scale2):
        B, C, H, W = x.shape
        m = self.mask_net(x)
        if m.shape[-2:] != (H, W):
            m = F.interpolate(m, size=(H, W), mode="bilinear", align_corners=False)
        return x + x * m



class SA_upsample(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.refine = nn.Conv2d(c, 1, 3, 1, 1)

    def forward(self, x, scale, scale2):
        x = F.interpolate(x, scale_factor=(scale, scale2), mode="bilinear", align_corners=False)
        return self.refine(x)



class SAHFormer_SR(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_feats = args.n_feats
        self.n_stages = args.n_stages

        self.head = nn.Conv2d(args.n_colors, self.n_feats, 3, 1, 1)

        self.blocks = nn.ModuleList([
            SAHBlock(self.n_feats, res_scale=args.sah_res_scale)
            for _ in range(self.n_stages)
        ])

        self.sa_adapt = nn.ModuleList([SA_adapt_v2(self.n_feats) for _ in range(self.n_stages)])

        # SRAG every 2 stages (assumes n_stages=6; extend as needed)
        self.stage_aggs = nn.ModuleList([
            StageAgg(self.n_feats, 2),
            StageAgg(self.n_feats, 4),
            StageAgg(self.n_feats, 6),
        ])

        # Wavelet-enhanced late fusion (shallow HF injection)
        self.late_fuse = LateFreqFusionWavelet(
            self.n_feats,
            reduction=args.late_reduction,
            wt_levels=args.wt_levels,
            wt_type=args.wt_type
        )

        self.body_tail = nn.Conv2d(self.n_feats, self.n_feats, 3, 1, 1)
        self.sa_upsample = SA_upsample(self.n_feats)
        self.tail = nn.Conv2d(self.n_feats, args.n_colors, 3, 1, 1)

        self.scale = 1.0
        self.scale2 = 1.0
        self.out_size = args.out_size
        self.arm = AlignmentRefinementModule(args.n_colors)

    def set_scale(self, s, s2=None):
        self.scale = float(s)
        self.scale2 = float(s if s2 is None else s2)

    def forward(self, x):
        x = self.head(x)
        res = x

        feats = []
        feat_shallow = res  # shallow feature for HF injection (head output)

        for i in range(self.n_stages):
            res = self.blocks[i](res)
            res = self.sa_adapt[i](res, self.scale, self.scale2)
            feats.append(res)

            if (i + 1) % 2 == 0:
                idx = (i + 1) // 2 - 1
                if idx < len(self.stage_aggs):
                    res = self.stage_aggs[idx](feats)

        res = self.body_tail(res)


        res = self.late_fuse(feat_shallow, res)

        res = self.sa_upsample(res, self.scale, self.scale2)

        out = res + self.arm(res)

        return out



