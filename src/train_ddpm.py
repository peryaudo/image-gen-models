"""Train a DDPM (denoising diffusion probabilistic model, Ho et al., 2020).

The forward process corrupts a data image x_0 a little at a time with Gaussian noise, using a
fixed schedule beta_1..beta_T (src/sample.py: `make_betas`). Composing the T small steps gives
a closed form for any step, so training never has to simulate the chain:

    x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps,    eps ~ N(0, I)

where alpha_bar_t = prod_{s <= t} (1 - beta_s) falls from ~1 (clean) to ~0 (pure noise).
The network looks at (x_t, t) and predicts the noise that was added:

    loss = E_{t, x_0, eps} || eps_theta(x_t, t) - eps ||^2

This "simple" loss is the variational bound (ELBO) on log p(x_0) with its per-t weights
dropped. Knowing eps is enough to reverse each small step, since the reverse step's mean is
x_t with the predicted noise partly removed. Sampling starts from x_T ~ N(0, I) and walks back
to t = 0 (ancestral DDPM or deterministic DDIM, see src/sample.py).

Compared with flow matching (src/train_fm.py): the U-Net, optimizer, and loop are identical.
Only the path (variance-preserving, not a straight line) and the target (eps, not x1 - x0)
differ.

Run:  uv run python -m src.train_ddpm --config configs/ddpm_cifar10.yaml [key=value ...]
"""

import copy
import time
from pathlib import Path

import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.data import get_dataloader
from src.models.unet import UNet
from src.sample import ddim, ddpm_ancestral, make_betas
from src.utils import (
    ema_update,
    infinite,
    load_checkpoint,
    load_config,
    save_checkpoint,
    save_grid,
    seed_everything,
    to_unit,
)


def ddpm_loss(model: UNet, x0: torch.Tensor, alpha_bar: torch.Tensor) -> torch.Tensor:
    """Epsilon-prediction MSE at a uniformly random timestep t in {0, ..., T-1}."""
    t = torch.randint(0, len(alpha_bar), (x0.shape[0],), device=x0.device)
    eps = torch.randn_like(x0)
    ab = alpha_bar[t][:, None, None, None]  # broadcast over (C, H, W)
    x_t = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
    return (model(x_t, t.float()) - eps).pow(2).mean()


@torch.no_grad()
def save_samples(
    ema_model: UNet, betas: torch.Tensor, cfg: DictConfig, step: int, out_dir: Path
) -> dict:
    """Fixed-seed EMA samples (same x_T for every sampler), so grids compare across runs."""
    gen = torch.Generator().manual_seed(cfg.sample.seed)
    shape = (cfg.sample.n, cfg.model.in_channels, cfg.data.resolution, cfg.data.resolution)
    x_T = torch.randn(shape, generator=gen).to(next(ema_model.parameters()).device)
    logs = {}
    for steps in cfg.sample.ddim_steps:
        samples = to_unit(ddim(ema_model, x_T, betas, steps))
        grid = save_grid(samples, out_dir / f"samples_{step:07d}_ddim{steps}.png")
        logs[f"samples/ddim_{steps}"] = wandb.Image(grid)
    if cfg.sample.ancestral:
        samples = to_unit(ddpm_ancestral(ema_model, x_T, betas, gen))
        grid = save_grid(samples, out_dir / f"samples_{step:07d}_ancestral.png")
        logs["samples/ancestral"] = wandb.Image(grid)
    return logs


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
    betas = make_betas(**cfg.diffusion)
    # float64 cumprod, then float32 on the GPU for the training loss.
    alpha_bar = torch.cumprod(1 - betas, dim=0).float().to(device)
    print(
        f"alpha_bar: t=0 {alpha_bar[0]:.5f}, t=T/2 {alpha_bar[len(betas) // 2]:.5f}, "
        f"t=T-1 {alpha_bar[-1]:.2e}"
    )
    model = UNet(**cfg.model).to(device)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)  # eval: no dropout
    print(f"U-Net parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, betas=tuple(cfg.train.betas))
    # Linear warmup: the zero-initialized U-Net gets large, noisy gradients at the start.
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / cfg.train.warmup_steps)
    )

    step, wandb_id = 0, None
    if cfg.train.resume and ckpt_path.exists():
        ckpt = load_checkpoint(ckpt_path)
        model.load_state_dict(ckpt["model"])
        ema_model.load_state_dict(ckpt["ema"])
        opt.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
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

    data = infinite(loader)
    if cfg.train.overfit_batch:  # smoke test: the loss on one repeated batch should drop fast
        batch = next(data)
        data = iter(lambda: batch, None)
    max_norm = cfg.train.grad_clip or float("inf")  # inf = just measure the grad norm

    torch.cuda.reset_peak_memory_stats()
    start, last_time, last_step = time.time(), time.time(), step
    pbar = tqdm(total=cfg.train.max_steps, initial=step, dynamic_ncols=True)
    while step < cfg.train.max_steps:
        x0 = next(data)["image"].to(device, non_blocking=True)
        loss = ddpm_loss(model, x0, alpha_bar)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        opt.step()
        sched.step()
        ema_decay = ema_update(ema_model, model, cfg.train.ema_decay, step)
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
                "lr": opt.param_groups[0]["lr"],
                "grad_norm": grad_norm.item(),
                "ema_decay": ema_decay,
                "imgs_per_s": imgs_per_s,
            }
            wandb.log(metrics, step=step)
            pbar.set_postfix(loss=f"{loss.item():.4f}", ips=f"{imgs_per_s:.0f}")

        if step % cfg.train.sample_every == 0 or step == cfg.train.max_steps:
            logs = save_samples(ema_model, betas, cfg, step, run_dir / "samples")
            wandb.log(logs, step=step)
            last_time = time.time()  # keep sampling time out of the training throughput

        if step % cfg.train.ckpt_every == 0 or step == cfg.train.max_steps:
            state = {
                "kind": "ddpm",  # tells src/sample.py which model class and sampler to use
                "model": model.state_dict(),
                "ema": ema_model.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(),
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
    main(load_config("Train a DDPM"))
