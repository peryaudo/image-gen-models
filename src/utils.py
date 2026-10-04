"""Shared helpers: config loading, seeding, checkpointing, image grids."""

import argparse
import random
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torchvision.utils import make_grid, save_image


def load_config(description: str) -> DictConfig:
    """Load a YAML config, then apply `key=value` overrides from the command line.

    Struct mode makes an override with a mistyped key raise instead of silently adding it.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*", help="e.g. train.max_steps=200")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    OmegaConf.set_struct(cfg, True)
    return OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds all CUDA devices
    print(f"seed: {seed}")


def infinite(loader: Iterable) -> Iterator:
    """Loop over a DataLoader forever, so training is counted in steps rather than epochs."""
    while True:
        yield from loader


def to_unit(x: torch.Tensor) -> torch.Tensor:
    """Map model outputs from [-1, 1] back to [0, 1] for saving and FID."""
    return ((x.clamp(-1, 1) + 1) / 2).float()


def save_grid(images: torch.Tensor, path: Path, nrow: int = 8) -> torch.Tensor:
    """Save images already in [0, 1] as a grid PNG, and return the grid (e.g. for wandb)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    grid = make_grid(images, nrow=nrow)
    save_image(grid, path)
    return grid


def save_checkpoint(path: Path, state: dict) -> None:
    """Write to a temp file, then rename, so a crash mid-save never corrupts last.pt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def load_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)
