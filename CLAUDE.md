# CLAUDE.md

## Project overview

A hands-on comparison of four generative model families, implemented from scratch in PyTorch:

1. **VAE**
2. **Flow matching** (rectified flow)
3. **GAN**
4. **DDPM**

The goal is learning, not SOTA. The author is comfortable with PyTorch but has not yet worked through the math of these models, so the code should double as a learning resource. Prefer clarity over cleverness, and keep every model small enough to train in hours, not days. The code is also meant as a foundation for a later mini text-to-image project (latent diffusion / flow matching).

Implementation order is fixed: **VAE -> flow matching -> GAN -> DDPM**. Do not start a later model before the earlier one trains and produces samples.

## Hardware and environment

- Single GPU: NVIDIA RTX 3090 (24 GB). No multi-GPU, no distributed training.
- Python 3.10+, PyTorch 2.x, CUDA.
- Train in **fp32 by default**. bf16 (`torch.autocast("cuda", dtype=torch.bfloat16)`) may be added later as an optional config flag (`mixed_precision: bf16`), off by default, to speed up training.
- Do not implement `torch.compile` support.
- Project is managed with **uv** and `pyproject.toml` (no `requirements.txt`). Add dependencies with `uv add <pkg>` (dev tools with `uv add --dev <pkg>`), commit `uv.lock`, and run everything via `uv run`.
- Dependencies: `torch`, `torchvision`, `datasets`, `torch-fidelity` (or `torchmetrics`) for FID, `pyyaml`, `omegaconf`, `tqdm`, `matplotlib`, `wandb`. Dev: `ruff`.

## Experiment tracking (wandb)

- Use **wandb** for loss curves and metrics. One wandb run per training run; the run name matches the `runs/<name>/` directory, and the full config is passed as `wandb.init(config=...)`.
- Log scalars (losses, lr, grad norm, throughput, model-specific terms like KL or D(real)/D(fake)) and periodic sample grids as `wandb.Image`.
- wandb entity/project: `peryaudo/image-gen-models` (set in config; `WANDB_ENTITY` / `WANDB_PROJECT` / `WANDB_MODE` env vars override). Smoke tests should run with `WANDB_MODE=disabled` (or `offline`) so they do not pollute the dashboard.
- wandb is for viewing; checkpoints and sample grids are still saved locally under `runs/` so results stay reproducible without it.

## Datasets (all via Hugging Face `datasets`)

| Dataset | HF repo | Resolution used | Notes |
|---|---|---|---|
| CIFAR-10 | `uoft-cs/cifar10` | 32x32 | Image column is `img`, plus `label`. Main benchmark. |
| Anime faces | `huggan/anime-faces` | 64x64 | Image column is `image`. 17,029 unique images after dedup (the repo loads as 86,204 rows: `images/` and `data.zip` are copies, and the originals contain duplicates). `src/data.py` drops the zip rows and pixel-identical duplicates. |
| Flowers-102 | `huggan/flowers-102-categories` | 64x64 | Image column is `image`, plus `label`. Varying native sizes, so resize + center crop. |

Rules:
- All data goes through one loader in `src/data.py`. Do not write per-model data code.
- Normalize images to `[-1, 1]`. Generated samples are mapped back to `[0, 1]` for saving and FID.
- Random horizontal flip is the only augmentation.
- Training is **unconditional** unless a config explicitly says otherwise. Labels are ignored by default.
- Decode lazily (`with_transform`), use `num_workers >= 4` and `pin_memory=True`.
- Dataset cache location is controlled by `HF_HOME`. Never hardcode paths.
- If a repo or column name has changed and loading fails, check the dataset page and fix it in `src/data.py` only.

## Repository layout

```
.
├── CLAUDE.md
├── README.md
├── pyproject.toml       # deps, ruff config
├── uv.lock
├── configs/
│   ├── vae_cifar10.yaml
│   ├── fm_cifar10.yaml
│   ├── gan_cifar10.yaml
│   ├── ddpm_cifar10.yaml
│   └── ... (same pattern for anime, flowers)
├── src/
│   ├── data.py          # unified HF dataset loader
│   ├── models/
│   │   ├── unet.py      # shared U-Net (flow matching and DDPM)
│   │   ├── vae.py
│   │   └── gan.py       # generator + discriminator
│   ├── train_vae.py
│   ├── train_fm.py
│   ├── train_gan.py
│   ├── train_ddpm.py
│   ├── sample.py        # sampling for all models
│   ├── eval_fid.py      # FID computation (shared protocol)
│   └── utils.py         # seeding, EMA, checkpointing, image grids
├── runs/                # checkpoints, logs, sample grids (gitignored)
├── wandb/               # local wandb files (gitignored)
└── results/             # final tables and figures
```

Each `train_*.py` should be a single readable file: model creation, loss, loop, periodic sampling, checkpointing. Avoid deep abstraction hierarchies and framework-style trainers.

## Model specifications

### 1. VAE
- Conv encoder -> `mu`, `logvar`; conv decoder.
- Loss = reconstruction (MSE by default) + `beta * KL`. Start with `beta = 1` and log both terms separately.
- Reparameterization: `z = mu + exp(0.5 * logvar) * eps`.
- Latent dim: 128 for CIFAR-10, 256 for 64px datasets.
- Blurry samples are expected and are part of the comparison. Do not add perceptual or adversarial losses unless asked.

### 2. Flow matching (rectified flow)
- Path: `x_t = (1 - t) * x0 + t * x1`, with `x0` = data, `x1 ~ N(0, I)`, `t ~ U(0, 1)`.
- Target velocity: `v = x1 - x0`. Loss: MSE between `model(x_t, t)` and `v`.
- Sampling: Euler ODE solver from `t = 1` to `t = 0`, with a configurable number of steps (default 50; also test 10 and 20).
- Backbone: the shared U-Net with sinusoidal time embedding.
- Use EMA of weights (decay 0.999 to 0.9999) for sampling.

### 3. GAN
- DCGAN-style baseline first, then add stabilizers one at a time and log the effect of each:
  1. Non-saturating generator loss
  2. Spectral normalization in the discriminator
  3. R1 gradient penalty (optional)
- Adam, learning rate around `2e-4`, `betas=(0.0, 0.99)` or `(0.5, 0.999)`.
- Log D loss, G loss, and mean D(real) / D(fake). Save sample grids frequently: mode collapse and divergence show up in the grids before the loss.
- Instability is expected and is part of what the project teaches. Document failures rather than hiding them.

### 4. DDPM
- Linear beta schedule (1e-4 to 0.02, T = 1000) as the baseline; cosine schedule as an optional ablation.
- Epsilon-prediction with simple MSE loss.
- Same U-Net backbone and EMA settings as flow matching, so the comparison is fair.
- Sampling: ancestral DDPM (1000 steps) and DDIM (deterministic; 50 steps and fewer). Keep sampler code separate from training code.

## Fair comparison protocol

- Flow matching and DDPM must use the **same U-Net architecture, parameter count, optimizer, batch size, and number of training steps**.
- VAE and GAN use their own architectures but should be in a roughly comparable parameter range (about 5M to 40M).
- Same seeds, same data, same resolution per dataset.
- **Metrics** for every model x dataset:
  - FID with 10k generated samples vs. the training set (same implementation and settings everywhere; document which).
  - Training wall-clock time and number of steps.
  - Sampling cost: NFE (network function evaluations) and seconds per 1k images.
  - Peak GPU memory.
- Always save a fixed-seed 8x8 sample grid for qualitative comparison.
- Final output: one results table per dataset in `results/`, plus a short written comparison in `README.md`.

## Coding conventions

- Type hints on public functions. Short docstrings explaining **why**, with the relevant equation in a comment where it helps (e.g., the flow matching target, the ELBO terms).
- All hyperparameters live in YAML configs. Command-line overrides (`key=value`) may only change values already in the config. No magic numbers in training scripts.
- Seed `random`, `numpy`, and `torch` at the start of every script, and log the seed.
- Checkpoints include model, EMA model, optimizer, step, and config, so training resumes exactly.
- Every N steps log loss, learning rate, grad norm, and throughput (images/s) to wandb. Every M steps save a sample grid (locally and to wandb).
- Keep functions short. Prefer plain PyTorch over third-party training frameworks.
- Lint and format with **ruff** only (no black): `uv run ruff check --fix .` and `uv run ruff format .`. Configure it in `pyproject.toml` with `line-length = 100`.

## Working agreements for Claude

- **Explain as you go.** When implementing a model, briefly state the math being implemented (the loss, the sampling rule). The author is learning this material, so keep explanations intuitive first, formal second.
- **Start small.** Before any long run, smoke-test the pipeline: overfit one batch, run about 100 steps, and check shapes, decreasing loss, and that a sample grid is saved.
- **Do not launch multi-hour training runs without asking.** Estimate the time from a short run's throughput and report it first.
- **Do not add features beyond the spec** (no perceptual loss, no classifier-free guidance, no latent diffusion) unless asked. List extension ideas at the end of a response instead.
- **Surface problems instead of silently working around them.** If a result looks wrong (NaN loss, all-gray samples, suspiciously low FID), say so and investigate before moving on.
- **Keep results reproducible.** Any number reported in the README must be traceable to a config and a checkpoint in `runs/`.

## Common commands

```bash
# Setup
uv sync

# Lint / format
uv run ruff check --fix .
uv run ruff format .

# Smoke test (a few hundred steps, wandb off)
WANDB_MODE=disabled uv run python -m src.train_fm --config configs/fm_cifar10.yaml train.max_steps=200

# Full training runs
uv run python -m src.train_vae  --config configs/vae_cifar10.yaml
uv run python -m src.train_fm   --config configs/fm_cifar10.yaml
uv run python -m src.train_gan  --config configs/gan_cifar10.yaml
uv run python -m src.train_ddpm --config configs/ddpm_cifar10.yaml

# Sampling
uv run python -m src.sample --ckpt runs/fm_cifar10/last.pt --n 64 --steps 50

# FID
uv run python -m src.eval_fid --ckpt runs/fm_cifar10/last.pt --n 10000
```

## Milestones

- [ ] `src/data.py` loads all three datasets and shows a sample grid
- [ ] VAE trains on CIFAR-10 and produces samples + reconstructions
- [ ] Flow matching trains on CIFAR-10 and produces samples (Euler, 50 steps)
- [ ] GAN trains on CIFAR-10 (document the stabilization steps)
- [ ] DDPM trains on CIFAR-10 with DDPM and DDIM sampling
- [ ] FID evaluation script shared by all models
- [ ] Repeat on anime-faces and flowers-102 at 64x64
- [ ] Results tables and written comparison in `README.md`