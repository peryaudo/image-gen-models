"""Unified Hugging Face dataset loader shared by every model.

Images are normalized to [-1, 1]; random horizontal flip is the only augmentation.
Labels are dropped: all training is unconditional.

Preview a dataset:  uv run python -m src.data --dataset cifar10
"""

import argparse
import hashlib
import os
from pathlib import Path

import torch
from datasets import Dataset, Image, load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

# name -> (HF repo, image column, drop exact-duplicate images)
DATASETS: dict[str, tuple[str, str, bool]] = {
    "cifar10": ("uoft-cs/cifar10", "img", False),
    # The repo ships every image twice (images/*.png and data.zip), and the originals already
    # contain duplicates: 86,204 rows hold only 17,029 distinct images.
    "anime": ("huggan/anime-faces", "image", True),
    "flowers": ("huggan/flowers-102-categories", "image", False),
}


def _pixel_hashes(batch: dict, column: str) -> dict:
    return {
        "hash": [hashlib.md5(img.convert("RGB").tobytes()).hexdigest() for img in batch[column]]
    }


def drop_zip_rows(ds: Dataset, column: str) -> Dataset:
    """Drop rows stored inside a zip archive (only reads file paths, no image decoding).

    For anime-faces these rows repeat images/*.png exactly. Each one is also very slow to
    decode, because every read re-scans the archive's 43k-entry index.
    """
    paths = [img["path"] or "" for img in ds.cast_column(column, Image(decode=False))[column]]
    keep = [i for i, p in enumerate(paths) if not p.startswith("zip://")]
    return ds.select(keep)


def drop_duplicates(ds: Dataset, column: str) -> Dataset:
    """Keep the first copy of each pixel-identical image.

    Duplicates would make some images count several times in training and in the FID
    reference set. The hashes go through `map`, so HF caches them after the first run.
    """
    n_rows = len(ds)
    ds = drop_zip_rows(ds, column)
    hashes = ds.map(
        _pixel_hashes,
        fn_kwargs={"column": column},
        batched=True,
        remove_columns=ds.column_names,
        num_proc=os.cpu_count(),
        desc="hashing images",
    )["hash"]
    seen: set[str] = set()
    keep = []
    for i, h in enumerate(hashes):
        if h not in seen:
            seen.add(h)
            keep.append(i)
    print(f"dedup: kept {len(keep)} of {n_rows} images")
    return ds.select(keep)


def build_transform(resolution: int, hflip: bool) -> transforms.Compose:
    """Resize + center crop (needed for flowers' varying sizes), optional flip, map to [-1, 1]."""
    steps = [transforms.Resize(resolution), transforms.CenterCrop(resolution)]
    if hflip:
        steps.append(transforms.RandomHorizontalFlip())
    steps += [transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)]
    return transforms.Compose(steps)


def get_dataset(name: str, resolution: int, hflip: bool = True) -> Dataset:
    """Training split with lazy decoding: each item becomes {"image": float tensor in [-1, 1]}."""
    repo, column, dedup = DATASETS[name]
    ds = load_dataset(repo, split="train")
    if dedup:
        ds = drop_duplicates(ds, column)
    tf = build_transform(resolution, hflip)

    def apply(batch: dict) -> dict:
        return {"image": [tf(img.convert("RGB")) for img in batch[column]]}

    return ds.with_transform(apply)


def get_dataloader(
    name: str, resolution: int, batch_size: int, num_workers: int, hflip: bool = True
) -> DataLoader:
    ds = get_dataset(name, resolution, hflip)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )


def fixed_batch(name: str, resolution: int, n: int) -> torch.Tensor:
    """The first n training images, unflipped: a stable batch for reconstruction grids."""
    ds = get_dataset(name, resolution, hflip=False)
    return torch.stack(ds[:n]["image"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, default="cifar10")
    parser.add_argument("--resolution", type=int, default=None)
    args = parser.parse_args()
    res = args.resolution or (32 if args.dataset == "cifar10" else 64)

    ds = get_dataset(args.dataset, res)
    x = torch.stack(ds[:64]["image"])
    print(
        f"{args.dataset}: {len(ds)} images, batch {tuple(x.shape)}, "
        f"range [{x.min():.2f}, {x.max():.2f}]"
    )
    out = Path("runs/data_preview") / f"{args.dataset}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    save_image((x + 1) / 2, out, nrow=8)
    print(f"saved {out}")
