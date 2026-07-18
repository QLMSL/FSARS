import torch
import torch.nn as nn
import torch.nn.functional as F


class wavelet:
    @staticmethod
    def create_2d_wavelet_filter(wt_type: str, in_ch: int, out_ch: int, dtype=torch.float32):

        if wt_type.lower() not in ["db1", "haar"]:
            raise ValueError("This self-contained version supports only wt_type='db1' (haar).")

        # 1D Haar low/high
        s = 0.5 ** 0.5
        lo = torch.tensor([s, s], dtype=dtype)
        hi = torch.tensor([s, -s], dtype=dtype)

        # 2D separable filters: LL, LH, HL, HH
        ll = torch.ger(lo, lo)  # [2,2]
        lh = torch.ger(lo, hi)
        hl = torch.ger(hi, lo)
        hh = torch.ger(hi, hi)

        base = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)  # [4,1,2,2]

        wt_filter = base.repeat(in_ch, 1, 1, 1)

        iwt_filter = wt_filter.clone()

        return wt_filter, iwt_filter

    @staticmethod
    def wavelet_2d_transform(x_ll: torch.Tensor, wt_filter: torch.Tensor):
        B, C, H, W = x_ll.shape
        y = F.conv2d(x_ll, wt_filter, stride=2, padding=0, groups=C)  # [B, 4C, H/2, W/2]
        y = y.view(B, C, 4, y.shape[-2], y.shape[-1])
        return y

    @staticmethod
    def inverse_2d_wavelet_transform(x: torch.Tensor, iwt_filter: torch.Tensor):
        B, C, _, H, W = x.shape
        x = x.view(B, C * 4, H, W)
        y = F.conv_transpose2d(x, iwt_filter, stride=2, padding=0, groups=C)  # [B, C, 2H, 2W]
        return y


class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)

    def forward(self, x):
        return x * self.weight


class WTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1'):
        super().__init__()
        assert in_channels == out_channels, "WTConv2d expects in_channels == out_channels (depthwise-style)."

        self.in_channels = in_channels
        self.wt_levels = int(wt_levels)
        self.stride = int(stride)

        wt_filter, iwt_filter = wavelet.create_2d_wavelet_filter(wt_type, in_channels, in_channels, torch.float32)
        self.wt_filter = nn.Parameter(wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(iwt_filter, requires_grad=False)

        self.base_conv = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            padding="same", stride=1, dilation=1, groups=in_channels, bias=bias
        )
        self.base_scale = _ScaleModule([1, in_channels, 1, 1], init_scale=1.0)


        self.wavelet_convs = nn.ModuleList([
            nn.Conv2d(
                in_channels * 4, in_channels * 4, kernel_size,
                padding="same", stride=1, dilation=1, groups=in_channels * 4, bias=False
            )
            for _ in range(self.wt_levels)
        ])
        self.wavelet_scale = nn.ModuleList([
            _ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1)
            for _ in range(self.wt_levels)
        ])

        self.do_stride = nn.AvgPool2d(kernel_size=1, stride=self.stride) if self.stride > 1 else None

    def forward(self, x):

        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x
        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)


            if (curr_shape[2] % 2 != 0) or (curr_shape[3] % 2 != 0):
                pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, pads)

            curr_x = wavelet.wavelet_2d_transform(curr_x_ll, self.wt_filter)
            curr_x_ll = curr_x[:, :, 0, :, :]

            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])

        next_x_ll = 0
        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll
            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = wavelet.inverse_2d_wavelet_transform(curr_x, self.iwt_filter)
            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        x_tag = next_x_ll


        base = self.base_scale(self.base_conv(x))


        y = base + x_tag

        if self.do_stride is not None:
            y = self.do_stride(y)

        return y