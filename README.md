# DiffusionNet — Conditional Generative Modeling on FashionMNIST

This repo implements and compares **four class-conditional generative models** on **FashionMNIST (32×32 padded)**:

- **DDPM** (Denoising Diffusion Probabilistic Model)  
- **DDIM** (Denoising Diffusion Implicit Model)  
- **VAE** (Conditional Variational Autoencoder)  
- **LDDPM** (Latent Diffusion: DDPM in VAE latent space)

The code is written from scratch in **PyTorch**: custom UNet, variance/alpha schedules, training loops, and sampling pipelines. It trains on GPU or CPU and logs simple quantitative checks using a small CNN classifier supplied in the repo.

---

## Highlights (what a reviewer should know)

- **Conditional generation**: all models condition on the FashionMNIST class label (0–9).  
- **Minimal, clear code**: models and trainers are short and readable; no external training frameworks.  
- **End-to-end**: training → checkpointing → batched sampling → quick accuracy sanity-check.  
- **Results snapshot** (examples from my runs):
  - **VAE**: classifier accuracy on generated samples ~**88%** (score 1.00)  
  - **DDPM**: generated samples reached ~**80%** (score 0.50) in a short run  
  - **DDIM**: prototype run (very few steps) reached ~**18%** — kept to show work-in-progress tuning  
  - All samplers save grids: `DDPM_generated_samples.png`, `DDIM_generated_samples.png`, `VAE_generated_samples.png`, `LDDPM_generated_samples.png`

---

## Implementation Details

### Dataset
- **FashionMNIST**, grayscale, padded to **32×32** for clean ×2 down/upsampling.
- Normalization to `[-1, 1]` where required (diffusion).

### Conditioning
- Class labels are passed to the UNet/decoders (embedding + injection).
- Sampling APIs accept a `labels` tensor to generate specific classes in batch.

### UNet (diffusion backbones)
- Input channels: `1` for pixel-space diffusion; equals latent dim for LDDPM.
- Example DDPM/DDIM config:
  ```python
  UNet(
    in_channels=1,
    down_channels=(64, 128, 256, 512),
    up_channels=(512, 256, 128, 64),
    time_emb_dim=128,
    num_classes=10
  )
