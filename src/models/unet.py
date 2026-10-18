"""U-Net backbone shared by flow matching and DDPM.

This follows the DDPM (Ho et al., 2020) CIFAR-10 architecture: residual blocks with GroupNorm,
self-attention at low resolutions, and a sinusoidal time embedding that is added inside every
residual block. The network maps (x_t, t) to a tensor the same shape as x_t: a velocity for
flow matching, a noise estimate for DDPM. Only the training target differs between the two,
so they can share this file and a fair comparison is a matter of using the same config.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Transformer-style sinusoidal embedding of a (B,) tensor of times -> (B, dim).

    emb = [cos(t * w_k), sin(t * w_k)] with frequencies w_k from 1 down to 1/max_period.
    The frequencies are tuned for t on a scale of ~[0, 1000], which is why flow matching
    (t in [0, 1]) sets `time_scale: 1000`; otherwise nearby times would be indistinguishable.
    """
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=1)


def zero_init(module: nn.Module) -> nn.Module:
    """Zero the last layer of a residual branch, so each block starts as the identity."""
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class ResBlock(nn.Module):
    """GN -> SiLU -> conv, add the time embedding, GN -> SiLU -> dropout -> conv, plus a skip."""

    def __init__(self, c_in: int, c_out: int, t_dim: int, dropout: float) -> None:
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, c_in), nn.SiLU(), nn.Conv2d(c_in, c_out, 3, padding=1)
        )
        self.t_proj = nn.Sequential(nn.SiLU(), nn.Linear(t_dim, c_out))
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, c_out),
            nn.SiLU(),
            nn.Dropout(dropout),
            zero_init(nn.Conv2d(c_out, c_out, 3, padding=1)),
        )
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        h = h + self.t_proj(t_emb)[:, :, None, None]  # same time shift at every pixel
        return self.skip(x) + self.block2(h)


class AttnBlock(nn.Module):
    """Single-head self-attention over all pixels, so distant regions can coordinate."""

    def __init__(self, c: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(32, c)
        self.qkv = nn.Conv2d(c, 3 * c, 1)
        self.proj = zero_init(nn.Conv2d(c, c, 1))

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor | None = None) -> torch.Tensor:
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).flatten(2).transpose(1, 2).chunk(3, dim=2)  # (B, HW, C)
        out = F.scaled_dot_product_attention(q, k, v)  # softmax(q k^T / sqrt(C)) v
        return x + self.proj(out.transpose(1, 2).reshape(b, c, h, w))


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        resolution: int,
        base_channels: int,
        channel_mult: list[int],
        num_res_blocks: int,
        attn_resolutions: list[int],
        dropout: float,
        time_scale: float,
    ) -> None:
        super().__init__()
        self.in_channels, self.resolution, self.time_scale = in_channels, resolution, time_scale
        self.base_channels = base_channels
        t_dim = 4 * base_channels
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim)
        )
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Down path. Every block's output is saved as a skip connection for the up path,
        # so we track the skip channel counts to size the up-path blocks.
        self.down = nn.ModuleList()
        skips, c, res = [base_channels], base_channels, resolution
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                self.down.append(ResBlock(c, base_channels * mult, t_dim, dropout))
                c = base_channels * mult
                if res in attn_resolutions:
                    self.down.append(AttnBlock(c))
                skips.append(c)
            if level != len(channel_mult) - 1:
                self.down.append(nn.Conv2d(c, c, 3, stride=2, padding=1))  # downsample 2x
                res //= 2
                skips.append(c)

        self.mid = nn.ModuleList(
            [ResBlock(c, c, t_dim, dropout), AttnBlock(c), ResBlock(c, c, t_dim, dropout)]
        )

        # Up path: one extra block per level, so every saved skip is consumed exactly once.
        self.up = nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_mult))):
            for _ in range(num_res_blocks + 1):
                self.up.append(ResBlock(c + skips.pop(), base_channels * mult, t_dim, dropout))
                c = base_channels * mult
                if res in attn_resolutions:
                    self.up.append(AttnBlock(c))
            if level != 0:
                self.up.append(nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(c, c, 3, 1, 1)))
                res *= 2

        self.out = nn.Sequential(
            nn.GroupNorm(32, c), nn.SiLU(), zero_init(nn.Conv2d(c, in_channels, 3, padding=1))
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) noisy images, t: (B,) times on the model's own scale."""
        t_emb = self.time_mlp(timestep_embedding(t * self.time_scale, self.base_channels))
        h = self.conv_in(x)
        hs = [h]
        for layer in self.down:
            if isinstance(layer, AttnBlock):
                hs[-1] = h = layer(h)  # attention refines the skip its ResBlock just saved
            else:
                h = layer(h, t_emb) if isinstance(layer, ResBlock) else layer(h)
                hs.append(h)
        for layer in self.mid:
            h = layer(h, t_emb)
        for layer in self.up:
            if isinstance(layer, ResBlock):
                h = layer(torch.cat([h, hs.pop()], dim=1), t_emb)
            else:
                h = layer(h)
        assert not hs, "every skip connection should be consumed"
        return self.out(h)
