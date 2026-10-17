"""Sampling for all models: load any checkpoint and draw images from it.

Each model kind registers one sampler that maps (model, n, generator) to images in [0, 1],
plus its NFE (network function evaluations per image), the unit of sampling cost:
a VAE needs 1 decoder pass, an ODE/SDE sampler needs one U-Net pass per step.

Run:  uv run python -m src.sample --ckpt runs/vae_cifar10/last.pt --n 64
"""

import argparse
from collections.abc import Callable, Iterator
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

from src.models.vae import VAE
from src.utils import load_checkpoint, save_grid, to_unit

# kind -> model class; the checkpoint's "kind" key picks the entry.
MODELS: dict[str, type[nn.Module]] = {"vae": VAE}


@torch.no_grad()
def sample_vae(
    model: VAE, n: int, generator: torch.Generator, steps: int | None
) -> tuple[torch.Tensor, int]:
    """Decode z ~ N(0, I), the prior. One forward pass, so NFE = 1 (`steps` is unused)."""
    z = torch.randn(n, model.latent_dim, generator=generator)
    device = next(model.parameters()).device
    return to_unit(model.decode(z.to(device))), 1


Sampler = Callable[[nn.Module, int, torch.Generator, int | None], tuple[torch.Tensor, int]]
SAMPLERS: dict[str, Sampler] = {"vae": sample_vae}


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
    kind: str, model: nn.Module, n: int, batch_size: int, seed: int, steps: int | None = None
) -> Iterator[tuple[torch.Tensor, int]]:
    """Yield (images in [0, 1], NFE) batches. A CPU generator makes the noise, and so the
    samples, identical for a given seed regardless of batch size or GPU."""
    gen = torch.Generator().manual_seed(seed)
    for start in range(0, n, batch_size):
        yield SAMPLERS[kind](model, min(batch_size, n - start), gen, steps)


def main() -> None:
    parser = argparse.ArgumentParser(description="Save a fixed-seed sample grid")
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--steps", type=int, default=None, help="sampler steps (FM / DDPM)")
    parser.add_argument("--seed", type=int, default=None, help="default: the config's sample.seed")
    args = parser.parse_args()

    kind, model, cfg, _ = load_model(args.ckpt, torch.device("cuda"))
    seed = cfg.sample.seed if args.seed is None else args.seed
    images = torch.cat(
        [x.cpu() for x, _ in generate(kind, model, args.n, args.n, seed, args.steps)]
    )
    suffix = "" if args.steps is None else f"_steps{args.steps}"
    out = args.ckpt.parent / f"grid_seed{seed}{suffix}.png"
    save_grid(images, out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
