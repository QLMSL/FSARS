import torch
import torch.nn as nn



class FFTFreqBranch(nn.Module):
    """
    FFT-based frequency branch:
    - Extract high-frequency components via FFT
    - Transform back to spatial domain
    - Light depthwise refinement
    """
    def __init__(self, c, hf_ratio=0.5):
        super().__init__()
        self.hf_ratio = hf_ratio

        # light spatial refinement after iFFT
        self.refine = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1, groups=c, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 1, 1, 0, bias=True)
        )

    def forward(self, x):
        """
        x: [B, C, H, W]
        """
        B, C, H, W = x.shape

        # ---- FFT ----
        fft = torch.fft.fft2(x, norm="ortho")        # [B,C,H,W]
        mag = torch.abs(fft)

        # ---- radial HF mask (no grad, deterministic) ----
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device),
            torch.linspace(-1, 1, W, device=x.device),
            indexing="ij"
        )
        rr = torch.sqrt(xx ** 2 + yy ** 2)
        hf_mask = (rr >= self.hf_ratio).float()      # [H,W]
        hf_mask = hf_mask.unsqueeze(0).unsqueeze(0) # [1,1,H,W]

        # ---- keep phase, suppress LF magnitude ----
        fft_hf = fft * hf_mask

        # ---- inverse FFT ----
        hf_spatial = torch.fft.ifft2(fft_hf, norm="ortho").real

        # ---- spatial refinement ----
        return self.refine(hf_spatial)

