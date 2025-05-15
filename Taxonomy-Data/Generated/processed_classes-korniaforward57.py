import torch
from torch import Tensor
from torch.nn import Module

class AdjustHue(Module):
    def __init__(self, hue_factor: float):
        super().__init__()
        self.hue_factor = hue_factor

    def forward(self, input: Tensor) -> Tensor:
        # Ensure the input is a float tensor
        if not input.is_floating_point():
            raise TypeError('Input tensor should be a float tensor with values in [0, 1]')

        # Check the input shape
        if input.size(-3) != 3:
            raise ValueError('Input tensor should have 3 channels in the last dimension')

        # Convert RGB to HSV
        hsv = self.rgb_to_hsv(input)

        # Adjust the hue
        hsv[..., 0, :, :] = (hsv[..., 0, :, :] + self.hue_factor) % 1.0

        # Convert back to RGB
        return self.hsv_to_rgb(hsv)

    def rgb_to_hsv(self, rgb: Tensor) -> Tensor:
        r, g, b = rgb.unbind(dim=-3)

        maxc = torch.max(rgb, dim=-3).values
        minc = torch.min(rgb, dim=-3).values
        v = maxc
        deltac = maxc - minc

        s = deltac / v
        s[v == 0] = 0

        rc = (maxc - r) / deltac
        gc = (maxc - g) / deltac
        bc = (maxc - b) / deltac

        h = torch.zeros_like(v)
        h[r == maxc] = bc[r == maxc] - gc[r == maxc]
        h[g == maxc] = 2.0 + rc[g == maxc] - bc[g == maxc]
        h[b == maxc] = 4.0 + gc[b == maxc] - rc[b == maxc]

        h = (h / 6.0) % 1.0
        h[deltac == 0] = 0.0

        return torch.stack((h, s, v), dim=-3)

    def hsv_to_rgb(self, hsv: Tensor) -> Tensor:
        h, s, v = hsv.unbind(dim=-3)
        i = (h * 6.0).floor()
        f = (h * 6.0) - i
        i = i.to(torch.int32) % 6

        p = v * (1.0 - s)
        q = v * (1.0 - f * s)
        t = v * (1.0 - (1.0 - f) * s)

        conditions = [
            (i == 0, torch.stack((v, t, p), dim=-3)),
            (i == 1, torch.stack((q, v, p), dim=-3)),
            (i == 2, torch.stack((p, v, t), dim=-3)),
            (i == 3, torch.stack((p, q, v), dim=-3)),
            (i == 4, torch.stack((t, p, v), dim=-3)),
            (i == 5, torch.stack((v, p, q), dim=-3)),
        ]

        rgb = torch.zeros_like(hsv)
        for condition, value in conditions:
            rgb = torch.where(condition.unsqueeze(-3), value, rgb)

        return rgb