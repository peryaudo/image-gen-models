"""Train a flow matching model (rectified flow).

Pick a data image x0 and a noise image x1 ~ N(0, I), and connect them with a straight line:

    x_t = (1 - t) * x0 + t * x1,    t ~ U(0, 1)

Moving along that line has constant velocity  dx_t/dt = x1 - x0.  We train a network to
predict this velocity from (x_t, t) alone:

    loss = E || v_theta(x_t, t) - (x1 - x0) ||^2

A given x_t lies on many such lines (many (x0, x1) pairs pass through it), so the best the
network can do is predict their average direction, E[x1 - x0 | x_t]. That average velocity
field defines an ODE whose flow carries N(0, I) exactly onto the data distribution, so
sampling = start from noise at t = 1 and integrate dx/dt = v_theta back to t = 0
(see `euler_ode` in src/sample.py).

Run:  uv run python -m src.train_fm --config configs/fm_cifar10.yaml [key=value ...]
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
from src.sample import euler_ode
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


def fm_loss(model: UNet, x0: torch.Tensor) -> torch.Tensor:
    """Rectified flow loss: MSE between predicted and true velocity on the straight path."""
    x1 = torch.randn_like(x0)
    t = torch.rand(x0.shape[0], device=x0.device)
    t_ = t[:, None, None, None]  # broadcast over (C, H, W)
    x_t = (1 - t_) * x0 + t_ * x1
    v_target = x1 - x0
    return (model(x_t, t) - v_target).pow(2).mean()


@torch.no_grad()
def save_samples(
    ema_model: UNet, x1_fixed: torch.Tensor, steps_list: list[int], step: int, out_dir: Path
) -> dict:
    """Fixed-noise EMA samples for each Euler step count, so grids compare across steps/runs."""
    logs = {}
    for steps in steps_list:
        samples = to_unit(euler_ode(ema_model, x1_fixed, steps))
        grid = save_grid(samples, out_dir / f"samples_{step:07d}_steps{steps}.png")
        logs[f"samples/euler_{steps}"] = wandb.Image(grid)
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

    gen = torch.Generator().manual_seed(cfg.sample.seed)
    shape = (cfg.sample.n, cfg.model.in_channels, cfg.data.resolution, cfg.data.resolution)
    x1_fixed = torch.randn(shape, generator=gen).to(device)

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
        loss = fm_loss(model, x0)

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
            logs = save_samples(ema_model, x1_fixed, cfg.sample.steps, step, run_dir / "samples")
            wandb.log(logs, step=step)

        if step % cfg.train.ckpt_every == 0 or step == cfg.train.max_steps:
            state = {
                "kind": "fm",  # tells src/sample.py which model class and sampler to use
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
    main(load_config("Train a flow matching model"))
