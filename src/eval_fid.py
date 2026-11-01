"""FID evaluation: one protocol shared by every model.

FID embeds real and generated images with an Inception-v3 network (2048-d pool features),
fits a Gaussian to each set, and measures the Frechet distance between the two Gaussians:

    FID = ||mu_r - mu_g||^2 + Tr(S_r + S_g - 2 (S_r S_g)^(1/2))
          \\_ mean shift _/    \\_ mismatch in spread / diversity _/

Lower is better; 0 means identical feature statistics.

Protocol (keep it identical everywhere, or numbers are not comparable):
- Implementation: torch-fidelity, feature extractor "inception-v3-compat" (the original
  TF FID Inception weights; images are resized to 299x299 inside the extractor).
- Generated: `--n` samples (10k by default) from a fixed seed, using EMA weights if present.
- Reference: the full training split at the training resolution, no flip (CIFAR-10: 50k).
  Its features are cached by torch-fidelity under $TORCH_HOME/fidelity_cache.
- Both sides are quantized to uint8, exactly as if the images were saved as PNGs.
- FID is biased upward for small n, so 10k-sample FIDs read a few points higher than the
  50k-sample FIDs reported in most papers. Compare only numbers computed with this script.

Run:  uv run python -m src.eval_fid --ckpt runs/vae_cifar10/last.pt --n 10000
      uv run python -m src.eval_fid --ckpt runs/ddpm_cifar10/last.pt --sampler ddim --steps 50
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch_fidelity
from torch.utils.data import Dataset

from src.data import get_dataset
from src.sample import DDPM_SAMPLERS, Sampler, generate, get_sampler, load_model, sample_suffix
from src.utils import seed_everything, to_unit

FEATURE_EXTRACTOR = "inception-v3-compat"


def to_uint8(x: torch.Tensor) -> torch.Tensor:
    """[0, 1] floats -> uint8 in [0, 255], the input format torch-fidelity expects."""
    return (x * 255).round().clamp(0, 255).to(torch.uint8)


class UInt8Images(Dataset):
    """Wraps either a uint8 tensor (generated) or our HF training set (reference)."""

    def __init__(self, source: torch.Tensor | Dataset) -> None:
        self.source = source

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, i: int) -> torch.Tensor:
        if isinstance(self.source, torch.Tensor):
            return self.source[i]
        return to_uint8(to_unit(self.source[i]["image"]))


def compute_fid(generated: torch.Tensor, dataset: str, resolution: int) -> float:
    """FID of uint8 images (N, 3, H, W) against the training split of `dataset`."""
    ref = get_dataset(dataset, resolution, hflip=False)
    metrics = torch_fidelity.calculate_metrics(
        input1=UInt8Images(generated),
        input2=UInt8Images(ref),
        # The size is in the name so a changed dataset (e.g. new dedup) never hits a stale cache.
        input2_cache_name=f"{dataset}-{resolution}px-train-{len(ref)}",
        feature_extractor=FEATURE_EXTRACTOR,
        fid=True,
        cuda=True,
        verbose=False,
    )
    return metrics["frechet_inception_distance"]


@torch.no_grad()
def sample_uint8(
    sampler: Sampler, model: torch.nn.Module, n: int, batch_size: int, seed: int, steps: int | None
) -> tuple[torch.Tensor, int, float, float]:
    """Generate n images. Returns (uint8 images, NFE, seconds, peak GPU GB) for the cost table."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.time()
    chunks, nfe = [], 0
    for x, nfe in generate(sampler, model, n, batch_size, seed, steps):
        chunks.append(to_uint8(x).cpu())
    torch.cuda.synchronize()
    seconds = time.time() - start
    return torch.cat(chunks), nfe, seconds, torch.cuda.max_memory_allocated() / 1e9


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute FID for a checkpoint")
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--steps", type=int, default=None, help="sampler steps (FM / DDIM)")
    parser.add_argument("--sampler", choices=DDPM_SAMPLERS, default=None, help="DDPM only")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    seed_everything(args.seed)
    kind, model, cfg, step = load_model(args.ckpt, torch.device("cuda"))
    if kind == "ddpm" and args.sampler is None:
        args.sampler = "ancestral"  # explicit, so output names and logs say which sampler
    sampler = get_sampler(kind, cfg, args.sampler)
    images, nfe, seconds, peak_gb = sample_uint8(
        sampler, model, args.n, args.batch_size, args.seed, args.steps
    )
    fid = compute_fid(images, cfg.data.dataset, cfg.data.resolution)

    result = {
        "fid": fid,
        "ckpt": str(args.ckpt),
        "step": step,
        "dataset": cfg.data.dataset,
        "resolution": cfg.data.resolution,
        "n": args.n,
        "seed": args.seed,
        "sampler": args.sampler,
        "steps": args.steps,
        "nfe": nfe,
        "batch_size": args.batch_size,
        "sec_per_1k": seconds / args.n * 1000,
        "sample_peak_gpu_gb": peak_gb,
        "fid_impl": f"torch-fidelity {torch_fidelity.__version__} ({FEATURE_EXTRACTOR})",
    }
    print(json.dumps(result, indent=2))
    suffix = sample_suffix(args.sampler, args.steps)
    out = args.ckpt.parent / f"fid_n{args.n}{suffix}.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
