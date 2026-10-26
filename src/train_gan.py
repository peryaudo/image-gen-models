"""Train a GAN.

Two networks play a game. The discriminator D(x) = sigmoid(logit) estimates the probability
that x is real; the generator G(z) tries to make D call its samples real:

    min_G max_D  E_x[log D(x)] + E_z[log(1 - D(G(z)))]

D's side is ordinary binary cross-entropy (real -> 1, fake -> 0). If D were optimal, G's side
would minimize the Jensen-Shannon divergence between data and samples. In practice the two
are trained by alternating gradient steps, and nothing guarantees that converges: it can
oscillate, diverge, or collapse onto a few modes. Hence the stabilizers in the config:

1. Non-saturating G loss. Early on D easily rejects fakes, so D(G(z)) ~ 0, where
   log(1 - D) is flat and G gets almost no gradient. Instead G maximizes log D(G(z)):
   same fixed point, but the gradient is strongest exactly when G is losing.
2. Spectral norm in D (src/models/gan.py): bounds how sharply D can change, so its
   gradients, which are G's only training signal, stay well behaved.
3. R1 penalty: (gamma / 2) * E_x ||grad_x D(x)||^2 on real data. Pushes D to be flat
   around the data, which removes the oscillation around the equilibrium.

Run:  uv run python -m src.train_gan --config configs/gan_cifar10.yaml [key=value ...]
"""

import time
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.data import get_dataloader
from src.models.gan import Discriminator, Generator
from src.utils import (
    infinite,
    load_checkpoint,
    load_config,
    save_checkpoint,
    save_grid,
    seed_everything,
    to_unit,
)


def d_loss_fn(logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
    """BCE with real -> 1, fake -> 0, written in terms of logits l, with D = sigmoid(l):

    -log D(x)         = -log sigmoid(l)     = softplus(-l)
    -log(1 - D(G(z))) = -log sigmoid(-l)    = softplus(l)
    """
    return F.softplus(-logits_real).mean() + F.softplus(logits_fake).mean()


def g_loss_fn(logits_fake: torch.Tensor, kind: str) -> torch.Tensor:
    """saturating:     minimize  log(1 - D(G(z))) = -softplus(l)   (the original minimax)
    nonsaturating:  minimize -log D(G(z))       =  softplus(-l)
    """
    if kind == "saturating":
        return -F.softplus(logits_fake).mean()
    if kind == "nonsaturating":
        return F.softplus(-logits_fake).mean()
    raise ValueError(f"unknown generator loss: {kind}")


def r1_penalty(logits_real: torch.Tensor, x_real: torch.Tensor) -> torch.Tensor:
    """E_x ||grad_x l(x)||^2 on real images (x_real must have requires_grad=True).

    create_graph=True so the penalty itself can be differentiated w.r.t. D's weights.
    """
    (grad,) = torch.autograd.grad(logits_real.sum(), x_real, create_graph=True)
    return grad.pow(2).flatten(1).sum(dim=1).mean()


@torch.no_grad()
def save_samples(G: Generator, z_fixed: torch.Tensor, step: int, out_dir: Path) -> dict:
    """Fixed-z samples, so grids compare across steps and runs. eval(): BatchNorm uses its
    running statistics, as in src/sample.py and FID."""
    G.eval()
    grid = save_grid(to_unit(G(z_fixed)), out_dir / f"samples_{step:07d}.png")
    G.train()
    return {"samples": wandb.Image(grid)}


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
    G = Generator(**cfg.model).to(device)
    D = Discriminator(**cfg.discriminator).to(device)
    print(f"G parameters: {sum(p.numel() for p in G.parameters()) / 1e6:.2f}M")
    print(f"D parameters: {sum(p.numel() for p in D.parameters()) / 1e6:.2f}M")
    betas = tuple(cfg.train.betas)
    opt_g = torch.optim.Adam(G.parameters(), lr=cfg.train.lr_g, betas=betas)
    opt_d = torch.optim.Adam(D.parameters(), lr=cfg.train.lr_d, betas=betas)

    step, wandb_id = 0, None
    if cfg.train.resume and ckpt_path.exists():
        ckpt = load_checkpoint(ckpt_path)
        G.load_state_dict(ckpt["model"])
        D.load_state_dict(ckpt["discriminator"])
        opt_g.load_state_dict(ckpt["optimizer"])
        opt_d.load_state_dict(ckpt["optimizer_d"])
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

    gen = torch.Generator().manual_seed(cfg.sample.seed)
    z_fixed = torch.randn(cfg.sample.n, cfg.model.latent_dim, generator=gen).to(device)

    data = infinite(loader)
    if cfg.train.overfit_batch:  # smoke test: G should learn to reproduce one batch
        batch = next(data)
        data = iter(lambda: batch, None)
    use_r1 = cfg.loss.r1_gamma > 0

    torch.cuda.reset_peak_memory_stats()
    start, last_time, last_step = time.time(), time.time(), step
    pbar = tqdm(total=cfg.train.max_steps, initial=step, dynamic_ncols=True)
    while step < cfg.train.max_steps:
        x = next(data)["image"].to(device, non_blocking=True)
        z = torch.randn(x.shape[0], cfg.model.latent_dim, device=device)
        fake = G(z)

        # --- D step: classify real vs. fake. detach() so this step does not update G.
        x.requires_grad_(use_r1)
        logits_real = D(x)
        logits_fake = D(fake.detach())
        d_loss = d_loss_fn(logits_real, logits_fake)
        r1 = r1_penalty(logits_real, x) if use_r1 else torch.zeros((), device=device)
        opt_d.zero_grad(set_to_none=True)
        (d_loss + cfg.loss.r1_gamma / 2 * r1).backward()
        d_grad_norm = torch.nn.utils.clip_grad_norm_(D.parameters(), float("inf"))  # measure
        opt_d.step()

        # --- G step: push the (just updated) D's verdict on the same fakes toward "real".
        # Backprop runs through D into G; D's grads from here are cleared by the next D step.
        g_loss = g_loss_fn(D(fake), cfg.loss.generator)
        opt_g.zero_grad(set_to_none=True)
        g_loss.backward()
        g_grad_norm = torch.nn.utils.clip_grad_norm_(G.parameters(), float("inf"))
        opt_g.step()
        step += 1
        pbar.update(1)

        if step % cfg.train.log_every == 0:
            if not (torch.isfinite(d_loss) and torch.isfinite(g_loss)):
                raise RuntimeError(f"non-finite loss at step {step}: D {d_loss}, G {g_loss}")
            now = time.time()
            imgs_per_s = (step - last_step) * cfg.train.batch_size / (now - last_time)
            last_time, last_step = now, step
            metrics = {
                "d_loss": d_loss.item(),
                "g_loss": g_loss.item(),
                "r1": r1.item(),
                # D's probabilities on this batch: ~0.5 / 0.5 is a balanced game; D(real) ~ 1
                # with D(fake) ~ 0 means D is winning and G's gradients are vanishing.
                "d_real": torch.sigmoid(logits_real).mean().item(),
                "d_fake": torch.sigmoid(logits_fake).mean().item(),
                "lr_g": opt_g.param_groups[0]["lr"],
                "lr_d": opt_d.param_groups[0]["lr"],
                "g_grad_norm": g_grad_norm.item(),
                "d_grad_norm": d_grad_norm.item(),
                "imgs_per_s": imgs_per_s,
            }
            wandb.log(metrics, step=step)
            pbar.set_postfix(
                d=f"{d_loss.item():.3f}", g=f"{g_loss.item():.3f}", ips=f"{imgs_per_s:.0f}"
            )

        if step % cfg.train.sample_every == 0 or step == cfg.train.max_steps:
            wandb.log(save_samples(G, z_fixed, step, run_dir / "samples"), step=step)

        if step % cfg.train.ckpt_every == 0 or step == cfg.train.max_steps:
            state = {
                "kind": "gan",  # tells src/sample.py which model class and sampler to use
                "model": G.state_dict(),
                "ema": None,  # the GAN spec uses no EMA; key kept for a uniform checkpoint format
                "discriminator": D.state_dict(),
                "optimizer": opt_g.state_dict(),
                "optimizer_d": opt_d.state_dict(),
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
    main(load_config("Train a GAN"))
