"""Convolutional VAE.

The encoder maps an image x to a Gaussian q(z|x) = N(mu, diag(exp(logvar))).
The decoder maps a latent z back to the mean of p(x|z). Sampling new images is just
decoding z ~ N(0, I), the prior that the KL term pulls q(z|x) toward.
"""

from itertools import pairwise

import torch
from torch import nn


def conv_block(c_in: int, c_out: int, stride: int = 1) -> nn.Sequential:
    """3x3 conv -> GroupNorm -> SiLU. stride=2 halves the resolution."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1),
        nn.GroupNorm(32, c_out),
        nn.SiLU(),
    )


class VAE(nn.Module):
    def __init__(
        self, in_channels: int, resolution: int, channels: list[int], latent_dim: int
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        c_last = channels[-1]
        self.bottom = resolution // 2 ** (len(channels) - 1)  # spatial size at the bottleneck
        flat = c_last * self.bottom * self.bottom

        # Encoder: each level refines, then downsamples 2x while widening channels.
        enc = [conv_block(in_channels, channels[0])]
        for c_prev, c in pairwise(channels):
            enc += [conv_block(c_prev, c_prev), conv_block(c_prev, c, stride=2)]
        enc += [conv_block(c_last, c_last), nn.Flatten()]
        self.encoder = nn.Sequential(*enc)
        self.to_stats = nn.Linear(flat, 2 * latent_dim)  # -> (mu, logvar)

        # Decoder: mirror image of the encoder, with nearest upsampling + conv.
        self.from_latent = nn.Linear(latent_dim, flat)
        dec = [nn.Unflatten(1, (c_last, self.bottom, self.bottom)), conv_block(c_last, c_last)]
        for c, c_prev in zip(reversed(channels[1:]), reversed(channels[:-1])):
            dec += [nn.Upsample(scale_factor=2), conv_block(c, c_prev), conv_block(c_prev, c_prev)]
        # No output activation: this is the mean of a Gaussian over pixels in [-1, 1].
        dec += [nn.Conv2d(channels[0], in_channels, 3, padding=1)]
        self.decoder = nn.Sequential(*dec)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = self.to_stats(self.encoder(x)).chunk(2, dim=1)
        return mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.from_latent(z))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        # Reparameterization trick: z = mu + sigma * eps keeps sampling differentiable
        # w.r.t. mu and logvar, because the randomness lives in eps ~ N(0, I).
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return self.decode(z), mu, logvar
