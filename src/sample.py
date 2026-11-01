"""Sampling for all models: load any checkpoint and draw images from it.

Each model kind registers one sampler that maps (model, n, generator) to images in [0, 1],
plus its NFE (network function evaluations per image), the unit of sampling cost:
a VAE needs 1 decoder pass, an ODE/SDE sampler needs one U-Net pass per step.

Run:  uv run python -m src.sample --ckpt runs/vae_cifar10/last.pt --n 64
      uv run python -m src.sample --ckpt runs/fm_cifar10/last.pt --n 64 --steps 50
      uv run python -m src.sample --ckpt runs/gan_cifar10/last.pt --n 64
      uv run python -m src.sample --ckpt runs/ddpm_cifar10/last.pt --n 64  # ancestral, 1000 steps
      uv run python -m src.sample --ckpt runs/ddpm_cifar10/last.pt --n 64 --sampler ddim --steps 50
"""

import argparse
import math
from collections.abc import Callable, Iterator
from functools import partial
from itertools import pairwise
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

from src.models.gan import Generator
from src.models.unet import UNet
from src.models.vae import VAE
from src.utils import load_checkpoint, save_grid, to_unit

# kind -> model class; the checkpoint's "kind" key picks the entry.
MODELS: dict[str, type[nn.Module]] = {"vae": VAE, "fm": UNet, "gan": Generator, "ddpm": UNet}

FM_DEFAULT_STEPS = 50  # Euler steps when none are given (the spec's default)
DDIM_DEFAULT_STEPS = 50


@torch.no_grad()
def sample_vae(
    model: VAE, n: int, generator: torch.Generator, steps: int | None
) -> tuple[torch.Tensor, int]:
    """Decode z ~ N(0, I), the prior. One forward pass, so NFE = 1 (`steps` is unused)."""
    z = torch.randn(n, model.latent_dim, generator=generator)
    device = next(model.parameters()).device
    return to_unit(model.decode(z.to(device))), 1


@torch.no_grad()
def sample_gan(
    model: Generator, n: int, generator: torch.Generator, steps: int | None
) -> tuple[torch.Tensor, int]:
    """Map z ~ N(0, I) through the generator. One forward pass, so NFE = 1 (`steps` is unused)."""
    z = torch.randn(n, model.latent_dim, generator=generator)
    device = next(model.parameters()).device
    return to_unit(model(z.to(device))), 1


@torch.no_grad()
def euler_ode(model: UNet, x1: torch.Tensor, steps: int) -> torch.Tensor:
    """Integrate the learned flow dx/dt = v(x, t) from noise (t = 1) back to data (t = 0).

    Euler's method takes `steps` equal steps of size dt = -1/steps:
        x_{t + dt} = x_t + dt * v(x_t, t)
    If the learned paths were perfectly straight, a single step would be exact; curvature is
    what makes few-step sampling lose quality.
    """
    ts = torch.linspace(1.0, 0.0, steps + 1, device=x1.device)
    x = x1
    for t, t_next in pairwise(ts):
        x = x + (t_next - t) * model(x, t.expand(x.shape[0]))
    return x


@torch.no_grad()
def sample_fm(
    model: UNet, n: int, generator: torch.Generator, steps: int | None
) -> tuple[torch.Tensor, int]:
    """Euler ODE sampling from x1 ~ N(0, I). One U-Net pass per step, so NFE = steps."""
    steps = steps or FM_DEFAULT_STEPS
    shape = (n, model.in_channels, model.resolution, model.resolution)
    x1 = torch.randn(shape, generator=generator).to(next(model.parameters()).device)
    return to_unit(euler_ode(model, x1, steps)), steps


def make_betas(
    schedule: str, num_steps: int, beta_start: float, beta_end: float, cosine_s: float
) -> torch.Tensor:
    """DDPM noise schedule beta_1..beta_T as a (T,) float64 tensor (index t = 0..T-1).

    Everything else follows from the betas:
        alpha_t = 1 - beta_t,   alpha_bar_t = prod_{s <= t} alpha_s
        q(x_t | x_0) = N(sqrt(alpha_bar_t) x_0, (1 - alpha_bar_t) I)
    "linear" is the DDPM paper's choice. "cosine" (Nichol & Dhariwal, 2021) defines alpha_bar
    directly as a squared cosine, so it destroys information more evenly over t; the linear
    schedule is almost pure noise for the last ~third of the steps.
    """
    if schedule == "linear":
        return torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float64)
    if schedule == "cosine":
        s = torch.arange(num_steps + 1, dtype=torch.float64) / num_steps
        f = torch.cos((s + cosine_s) / (1 + cosine_s) * math.pi / 2) ** 2
        alpha_bar = f / f[0]
        # beta_t = 1 - alpha_bar_t / alpha_bar_{t-1}, clipped so the last steps stay invertible.
        return (1 - alpha_bar[1:] / alpha_bar[:-1]).clamp(max=0.999)
    raise ValueError(f"unknown schedule {schedule!r}")


@torch.no_grad()
def ddpm_ancestral(
    model: UNet, x_T: torch.Tensor, betas: torch.Tensor, generator: torch.Generator
) -> torch.Tensor:
    """Ancestral sampling (DDPM paper, Algorithm 2): T stochastic denoising steps.

    Each step removes the predicted noise and then adds back a little fresh noise:
        x_{t-1} = (x_t - beta_t / sqrt(1 - alpha_bar_t) * eps_theta(x_t, t)) / sqrt(alpha_t)
                  + sigma_t * z,    z ~ N(0, I),   sigma_t^2 = beta_t,   no noise at t = 0
    sigma_t^2 = beta_t is the paper's "fixed large" variance used for CIFAR-10. Fresh noise
    comes from the CPU `generator`, so samples are reproducible for a given seed.
    """
    alpha_bar = torch.cumprod(1 - betas, dim=0)
    x = x_T
    for t in reversed(range(len(betas))):
        beta, ab = betas[t].item(), alpha_bar[t].item()
        eps = model(x, torch.full((x.shape[0],), float(t), device=x.device))
        x = (x - beta / math.sqrt(1 - ab) * eps) / math.sqrt(1 - beta)
        if t > 0:
            x = x + math.sqrt(beta) * torch.randn(x.shape, generator=generator).to(x.device)
    return x


@torch.no_grad()
def ddim(model: UNet, x_T: torch.Tensor, betas: torch.Tensor, steps: int) -> torch.Tensor:
    """Deterministic DDIM sampling (eta = 0) on `steps` evenly spaced timesteps.

    The same eps-network, read differently: estimate the clean image, then re-noise it to the
    previous (less noisy) level using the *same* predicted noise instead of fresh noise:
        x0_hat = (x_t - sqrt(1 - alpha_bar_t) * eps) / sqrt(alpha_bar_t)
        x_prev = sqrt(alpha_bar_prev) * x0_hat + sqrt(1 - alpha_bar_prev) * eps
    With no randomness this is an ODE solver (like the FM Euler sampler), which is why it can
    skip most of the 1000 training timesteps. After the last step (t = 0), alpha_bar_prev = 1.
    """
    num_steps = len(betas)
    if not 1 <= steps <= num_steps:
        raise ValueError(f"DDIM steps must be in [1, {num_steps}], got {steps}")
    alpha_bar = torch.cumprod(1 - betas, dim=0).tolist() + [1.0]  # alpha_bar[-1] = 1: clean data
    ts = torch.linspace(num_steps - 1, 0, steps).round().long().tolist()
    x = x_T
    for t, t_prev in zip(ts, ts[1:] + [-1]):
        ab, ab_prev = alpha_bar[t], alpha_bar[t_prev]
        eps = model(x, torch.full((x.shape[0],), float(t), device=x.device))
        x0_hat = (x - math.sqrt(1 - ab) * eps) / math.sqrt(ab)
        x = math.sqrt(ab_prev) * x0_hat + math.sqrt(1 - ab_prev) * eps
    return x


@torch.no_grad()
def sample_ddpm(
    model: UNet,
    n: int,
    generator: torch.Generator,
    steps: int | None,
    betas: torch.Tensor,
    sampler: str,
) -> tuple[torch.Tensor, int]:
    """x_T ~ N(0, I), then "ancestral" (NFE = T, `steps` must be None or T) or "ddim"."""
    shape = (n, model.in_channels, model.resolution, model.resolution)
    x_T = torch.randn(shape, generator=generator).to(next(model.parameters()).device)
    if sampler == "ancestral":
        if steps not in (None, len(betas)):
            raise ValueError(f"ancestral sampling always uses all {len(betas)} steps")
        return to_unit(ddpm_ancestral(model, x_T, betas, generator)), len(betas)
    if sampler == "ddim":
        steps = steps or DDIM_DEFAULT_STEPS
        return to_unit(ddim(model, x_T, betas, steps)), steps
    raise ValueError(f"unknown DDPM sampler {sampler!r}")


Sampler = Callable[[nn.Module, int, torch.Generator, int | None], tuple[torch.Tensor, int]]
SAMPLERS: dict[str, Sampler] = {"vae": sample_vae, "fm": sample_fm, "gan": sample_gan}
DDPM_SAMPLERS = ("ancestral", "ddim")


def get_sampler(kind: str, cfg: DictConfig, sampler: str | None = None) -> Sampler:
    """The sampler for a model kind. DDPM needs its noise schedule from the config, and offers
    a choice of sampler (default: ancestral); the other kinds have exactly one."""
    if kind == "ddpm":
        betas = make_betas(**cfg.diffusion)
        return partial(sample_ddpm, betas=betas, sampler=sampler or "ancestral")
    if sampler is not None:
        raise ValueError(f"--sampler only applies to ddpm checkpoints, not {kind}")
    return SAMPLERS[kind]


def load_model(ckpt_path: Path, device: torch.device) -> tuple[str, nn.Module, DictConfig, int]:
    """Rebuild the model from the config stored in the checkpoint.

    EMA weights are used when present: they average out the noise of the last SGD steps
    and give noticeably better samples for diffusion / flow models.
    """
    ckpt = load_checkpoint(ckpt_path)
    kind = ckpt["kind"]
    cfg = OmegaConf.create(ckpt["config"])
    model = MODELS[kind](**cfg.model)
    model.load_state_dict(ckpt["ema"] if ckpt["ema"] is not None else ckpt["model"])
    print(f"loaded {kind} from {ckpt_path} (step {ckpt['step']}, ema={ckpt['ema'] is not None})")
    return kind, model.to(device).eval(), cfg, ckpt["step"]


def generate(
    sampler: Sampler,
    model: nn.Module,
    n: int,
    batch_size: int,
    seed: int,
    steps: int | None = None,
) -> Iterator[tuple[torch.Tensor, int]]:
    """Yield (images in [0, 1], NFE) batches. A CPU generator makes the noise, and so the
    samples, identical for a given seed regardless of GPU. (Samplers that draw noise every
    step, like DDPM ancestral, also depend on the batch size.)"""
    gen = torch.Generator().manual_seed(seed)
    for start in range(0, n, batch_size):
        yield sampler(model, min(batch_size, n - start), gen, steps)


def sample_suffix(sampler: str | None, steps: int | None) -> str:
    """Output-file suffix naming the sampler settings, e.g. "_ddim_steps50"."""
    return ("" if sampler is None else f"_{sampler}") + ("" if steps is None else f"_steps{steps}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Save a fixed-seed sample grid")
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--steps", type=int, default=None, help="sampler steps (FM / DDIM)")
    parser.add_argument("--sampler", choices=DDPM_SAMPLERS, default=None, help="DDPM only")
    parser.add_argument("--seed", type=int, default=None, help="default: the config's sample.seed")
    args = parser.parse_args()

    kind, model, cfg, _ = load_model(args.ckpt, torch.device("cuda"))
    if kind == "ddpm" and args.sampler is None:
        args.sampler = "ancestral"  # explicit, so output names and logs say which sampler
    sampler = get_sampler(kind, cfg, args.sampler)
    seed = cfg.sample.seed if args.seed is None else args.seed
    images = torch.cat(
        [x.cpu() for x, _ in generate(sampler, model, args.n, args.n, seed, args.steps)]
    )
    suffix = sample_suffix(args.sampler, args.steps)
    out = args.ckpt.parent / f"grid_seed{seed}{suffix}.png"
    save_grid(images, out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
