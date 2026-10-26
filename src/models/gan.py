"""DCGAN-style generator and discriminator.

The generator maps z ~ N(0, I) straight to an image in one pass (like the VAE decoder), but it
is never shown a real image: its only training signal is the discriminator's gradient.
The discriminator is a binary classifier, real vs. generated, that outputs a logit.

The layout follows DCGAN (Radford et al., 2015) as adapted to 32x32 in the spectral norm paper
(Miyato et al., 2018, "standard CNN"): strided transposed convs + BatchNorm + ReLU in G;
3x3 / strided 4x4 conv pairs + LeakyReLU in D. Stabilizers are switches, so the ablation in
the training config changes one thing at a time:
- `norm`: BatchNorm in D (the DCGAN baseline) or none.
- `spectral_norm`: divide every D weight by its largest singular value, which makes D
  1-Lipschitz per layer and keeps its gradients (G's training signal) from exploding.
"""

from itertools import pairwise

import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm as sn

LEAKY_SLOPE = 0.2  # DCGAN's LeakyReLU slope in D: keeps gradients alive for negative inputs
INIT_STD = 0.02  # DCGAN's weight init N(0, 0.02)


def dcgan_init(module: nn.Module) -> None:
    """DCGAN init: conv/linear weights ~ N(0, 0.02), BatchNorm scale ~ N(1, 0.02)."""
    if isinstance(module, nn.Conv2d | nn.ConvTranspose2d | nn.Linear):
        nn.init.normal_(module.weight, 0.0, INIT_STD)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, 1.0, INIT_STD)
        nn.init.zeros_(module.bias)


class Generator(nn.Module):
    """z -> 4x4 feature map -> (upsample 2x, halve channels) per level -> image in [-1, 1]."""

    def __init__(
        self, img_channels: int, resolution: int, latent_dim: int, channels: list[int]
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.bottom = resolution // 2 ** (len(channels) - 1)  # 4 for 32px with 4 levels
        c0 = channels[0]
        layers: list[nn.Module] = [
            nn.Linear(latent_dim, c0 * self.bottom * self.bottom),
            nn.Unflatten(1, (c0, self.bottom, self.bottom)),
            nn.BatchNorm2d(c0),
            nn.ReLU(),
        ]
        for c_in, c_out in pairwise(channels):
            # kernel 4, stride 2, padding 1: exactly doubles the resolution
            layers += [nn.ConvTranspose2d(c_in, c_out, 4, 2, 1), nn.BatchNorm2d(c_out), nn.ReLU()]
        # tanh squashes to [-1, 1], the range of the normalized data
        layers += [nn.Conv2d(channels[-1], img_channels, 3, padding=1), nn.Tanh()]
        self.net = nn.Sequential(*layers)
        self.apply(dcgan_init)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Discriminator(nn.Module):
    """image -> (3x3 conv, strided 4x4 conv) per level -> 4x4 map -> linear -> real/fake logit.

    Returns raw logits; the losses apply the sigmoid inside softplus for numerical stability.
    """

    def __init__(
        self,
        img_channels: int,
        resolution: int,
        channels: list[int],
        norm: str,
        spectral_norm: bool,
    ) -> None:
        super().__init__()
        if norm not in ("batchnorm", "none"):
            raise ValueError(f"unknown norm: {norm}")
        wrap = sn if spectral_norm else (lambda m: m)

        def block(c_in: int, c_out: int, kernel: int, stride: int, use_norm: bool) -> list:
            layers = [wrap(nn.Conv2d(c_in, c_out, kernel, stride, padding=(kernel - 1) // 2))]
            if use_norm and norm == "batchnorm":
                layers.append(nn.BatchNorm2d(c_out))
            return [*layers, nn.LeakyReLU(LEAKY_SLOPE)]

        # DCGAN: no BatchNorm on the first layer, which sees raw pixels.
        layers = block(img_channels, channels[0], 3, 1, use_norm=False)
        for c_in, c_out in pairwise(channels):
            layers += block(c_in, c_in, 4, 2, use_norm=True)  # halve the resolution
            layers += block(c_in, c_out, 3, 1, use_norm=True)  # widen
        bottom = resolution // 2 ** (len(channels) - 1)
        layers += [nn.Flatten(), wrap(nn.Linear(channels[-1] * bottom * bottom, 1))]
        self.net = nn.Sequential(*layers)
        self.apply(dcgan_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)
