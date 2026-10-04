"""Train a VAE.

We maximize the ELBO, a lower bound on log p(x):

    log p(x) >= E_q(z|x)[log p(x|z)]  -  KL(q(z|x) || p(z))
                \\_ reconstruction _/     \\_ keep codes near N(0, I) _/

so the loss to minimize is  recon + beta * KL  (beta = 1 is the true ELBO).

Run:  uv run python -m src.train_vae --config configs/vae_cifar10.yaml [key=value ...]
"""

import time
from pathlib import Path

import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.data import fixed_batch, get_dataloader
from src.models.vae import VAE
from src.utils import (
    infinite,
    load_checkpoint,
    load_config,
    save_checkpoint,
    save_grid,
    seed_everything,
    to_unit,
)


def vae_loss(
    x: torch.Tensor, x_hat: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-image ELBO terms, averaged over the batch.

    recon: -log p(x|z) for a Gaussian decoder with fixed variance is, up to a constant,
           the squared error summed over pixels (summed, not averaged, so it is weighed
           correctly against the KL, which is summed over latent dims).
    kl:    closed form for q = N(mu, sigma^2) vs. p = N(0, 1), per latent dim:
           KL = -0.5 * (1 + log sigma^2 - mu^2 - sigma^2)
    """
    recon = (x_hat - x).pow(2).flatten(1).sum(dim=1).mean()
    kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(dim=1).mean()
    return recon, kl


@torch.no_grad()
def save_samples(
    model: VAE, z_fixed: torch.Tensor, x_fixed: torch.Tensor, step: int, out_dir: Path
) -> dict:
    """Fixed-seed prior samples, plus reconstructions (rows alternate real / reconstructed)."""
    model.eval()
    samples = to_unit(model.decode(z_fixed))
    mu, _ = model.encode(x_fixed)
    recon = to_unit(model.decode(mu))  # decode the mean: the model's "best guess" for x
    real = to_unit(x_fixed)
    n_rows, n_cols = real.shape[0] // 8, 8
    paired = torch.stack(
        [real.view(n_rows, n_cols, *real.shape[1:]), recon.view(n_rows, n_cols, *real.shape[1:])],
        dim=1,
    ).flatten(0, 2)
    model.train()

    samples_grid = save_grid(samples, out_dir / f"samples_{step:07d}.png")
    recon_grid = save_grid(paired, out_dir / f"recon_{step:07d}.png")
    return {"samples": wandb.Image(samples_grid), "reconstructions": wandb.Image(recon_grid)}


def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    device = torch.device("cuda")
    run_dir = Path(cfg.out_dir) / cfg.run_name
    ckpt_path = run_dir / "last.pt"

    loader = get_dataloader(
        cfg.data.dataset,
        cfg.data.resolution,
        cfg.train.batch_size,
        cfg.data.num_workers,
        cfg.data.hflip,
    )
    model = VAE(**cfg.model).to(device)
    print(f"VAE parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, betas=tuple(cfg.train.betas))

    step, wandb_id = 0, None
    if cfg.train.resume and ckpt_path.exists():
        ckpt = load_checkpoint(ckpt_path)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        step, wandb_id = ckpt["step"], ckpt["wandb_id"]
        print(f"resumed from {ckpt_path} at step {step}")

    wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        name=cfg.run_name,
        id=wandb_id,
        resume="allow",
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    wandb_id = wandb.run.id

    # Same latents and images every time, so grids are comparable across steps and runs.
    gen = torch.Generator().manual_seed(cfg.sample.seed)
    z_fixed = torch.randn(cfg.sample.n, cfg.model.latent_dim, generator=gen).to(device)
    x_fixed = fixed_batch(cfg.data.dataset, cfg.data.resolution, cfg.sample.n_recon).to(device)

    data = infinite(loader)
    if cfg.train.overfit_batch:  # smoke test: the loss on one repeated batch should go ~0
        batch = next(data)
        data = iter(lambda: batch, None)
    max_norm = cfg.train.grad_clip or float("inf")  # inf = just measure the grad norm

    torch.cuda.reset_peak_memory_stats()
    start, last_time, last_step = time.time(), time.time(), step
    pbar = tqdm(total=cfg.train.max_steps, initial=step, dynamic_ncols=True)
    while step < cfg.train.max_steps:
        x = next(data)["image"].to(device, non_blocking=True)
        x_hat, mu, logvar = model(x)
        recon, kl = vae_loss(x, x_hat, mu, logvar)
        loss = recon + cfg.loss.beta * kl

        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        opt.step()
        step += 1
        pbar.update(1)

        if step % cfg.train.log_every == 0:
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}: {loss.item()}")
            now = time.time()
            imgs_per_s = (step - last_step) * cfg.train.batch_size / (now - last_time)
            last_time, last_step = now, step
            metrics = {
                "loss": loss.item(),
                "recon": recon.item(),
                "kl": kl.item(),
                "mse_per_pixel": recon.item() / x[0].numel(),
                "lr": opt.param_groups[0]["lr"],
                "grad_norm": grad_norm.item(),
                "imgs_per_s": imgs_per_s,
            }
            wandb.log(metrics, step=step)
            pbar.set_postfix(
                loss=f"{loss.item():.1f}", kl=f"{kl.item():.1f}", ips=f"{imgs_per_s:.0f}"
            )

        if step % cfg.train.sample_every == 0 or step == cfg.train.max_steps:
            wandb.log(save_samples(model, z_fixed, x_fixed, step, run_dir / "samples"), step=step)

        if step % cfg.train.ckpt_every == 0 or step == cfg.train.max_steps:
            state = {
                "model": model.state_dict(),
                "ema": None,  # the VAE spec uses no EMA; key kept for a uniform checkpoint format
                "optimizer": opt.state_dict(),
                "step": step,
                "config": OmegaConf.to_container(cfg, resolve=True),
                "wandb_id": wandb_id,
            }
            save_checkpoint(ckpt_path, state)
    pbar.close()

    elapsed = time.time() - start
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    wandb.run.summary.update({"train_seconds": elapsed, "peak_gpu_gb": peak_gb})
    print(f"done: step {step}, {elapsed / 60:.1f} min (this session), peak GPU {peak_gb:.2f} GB")
    wandb.finish()


if __name__ == "__main__":
    main(load_config("Train a VAE"))
