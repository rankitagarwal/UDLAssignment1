# Image Generation using (β-)VAE, VQ-VAE and WGAN — CIFAR-10

**Group Number: 123**

| # | Student Name | Email |
|---|---|---|
| 1 | Agarwal Rankit Atulkumar | 2025ab05296@wilp.bits-pilani.ac.in |
| 2 | Nishant Verma | 2025aa05051@wilp.bits-pilani.ac.in |
| 3 | Aryasomayajula Subrahmanya Srinivas | 2025aa05539@wilp.bits-pilani.ac.in |
| 4 | Wali Haider | 2025ab05295@wilp.bits-pilani.ac.in |
| 5 | Amit Mallikarjun Masuti | 2024ac05742@wilp.bits-pilani.ac.in |

## Deliverables

| file | contents |
|---|---|
| `PartA_VAE_VQVAE.ipynb` / `.html` | β-VAE (β ∈ {1,2,4,10}); VQ-VAE (K ∈ {512,256,128}) + Gated PixelCNN prior |
| `PartB_WGAN.ipynb` / `.html` | WGAN (weight clipping) and WGAN-GP (gradient penalty) |
| `PartC_Comparative_Analysis.pdf` | 100 samples per model, comparative discussion of reconstruction / sampling quality / speed |

**All outputs are already embedded in these files** — every figure is stored as inline base64, so
nothing needs to be executed to read them.

## Results

FID uses 10,000 generated images against the 10,000 CIFAR-10 test images (torchmetrics, 2048-d
InceptionV3 features). The identical real-side statistics are reused for every model. PSNR is the
mean per-image value over the test split at `data_range=1.0`.

| model | FID ↓ | PSNR ↑ | codebook | train | s / 1000 samples |
|---|---|---|---|---|---|
| β-VAE (β=1) | 108.4 | 23.97 dB | — | 3 min | 0.01 |
| β-VAE (β=2) | 106.5 | 24.00 dB | — | 3 min | 0.01 |
| β-VAE (β=4) | 107.4 | 23.86 dB | — | 3 min | 0.01 |
| β-VAE (β=10) | 118.3 | 22.39 dB | — | 3 min | 0.01 |
| VQ-VAE (K=512) | **88.9** | **25.45 dB** | 512/512, perplexity 435 | 31 min | 4.73 |
| VQ-VAE (K=256) | 92.8 | 24.89 dB | 256/256, perplexity 222 | 30 min | 4.60 |
| VQ-VAE (K=128) | 91.6 | 24.43 dB | 128/128, perplexity 114 | 29 min | 4.48 |
| WGAN | 55.2 | — | — | 31 min | 0.01 |
| **WGAN-GP** | **51.6** | — | — | 97 min | 0.01 |

Sanity floor: FID between two disjoint 5,000-image halves of real CIFAR-10 is 10.3.

Trained on one RTX 4060 Ti (16 GB). β-VAE and VQ-VAE ran 100 epochs, PixelCNN 60 epochs, and both
GANs exactly 70,000 generator iterations each (`n_critic = 5`).

## Two implementation details that decide whether this works

**1. The VQ-VAE codebook needs dead-code restarts.** An EMA codebook initialised from Gaussian
noise collapses to ~2 live codes within one epoch: a code that is never selected is never updated,
its smoothed cluster size decays toward zero, and dividing the accumulator by that vanishing size
pushes the code further from the data, so it can never be selected again. Seeding the codebook from
real encoder outputs and periodically restarting dead codes onto live encoder outputs takes
utilisation from 2/512 to 512/512 (perplexity 2.0 → 435).

**2. The VAE decoder variance must be calibrated.** Writing reconstruction as a bare
`MSE.sum()` implicitly asserts σ² = 0.5, but the achieved per-pixel error is ~0.004 — so
reconstruction is under-weighted against the KL by roughly two orders of magnitude and the
posterior partially collapses. That version scored **FID 200.5** at β=1 with only 23 nats in a
128-dim latent. Learning a single global σ (Rybkin et al., *Simple and Effective VAE Training with
Calibrated Decoders*, 2021) gives **FID 108.4** and 287 nats.

A consequence worth noting: with the calibrated decoder the β trend is monotone in the **KL**
(287 → 233 → 176 → 89 nats) but *flat* in PSNR and FID for β ≤ 4, breaking only at β = 10. The
latent has capacity to spare at 32×32, so moderate β removes information the decoder was not
relying on. The tidy monotone staircase the uncalibrated version produced was partly an artefact of
every model being starved from the outset.

## Reproducing

```bash
pip install -r requirements.txt        # torch-fidelity is required and easy to miss
jupyter nbconvert --to notebook --execute --inplace PartA_VAE_VQVAE.ipynb --ExecutePreprocessor.timeout=-1
jupyter nbconvert --to notebook --execute --inplace PartB_WGAN.ipynb      --ExecutePreprocessor.timeout=-1
python3 make_partc.py                  # rebuilds the Part C PDF from results/
python3 make_submission.py             # builds submission/ + the zip, under the portal's 10 MB cap
```

`make_submission.py` exists because an executed notebook here is ~12 MB, almost entirely base64 PNG
figures, and the portal caps uploads at 10 MB per file. It re-encodes the stored image bytes only —
charts stay lossless PNG, sample grids become JPEG q92 at full resolution — and asserts that code
cells, execution counts and text outputs come through byte-identical. **Re-executing a notebook
regenerates plain PNG figures, so re-run this script after any re-execution, before uploading.**

CIFAR-10 downloads automatically into `data/` (`download=True`). `data/`, `checkpoints/` and
`results/` are gitignored.

Notes:

* A full run from scratch takes **~4 hours** on an RTX 4060 Ti. The two notebooks are independent
  and can be run concurrently — they write separate metrics files precisely so that they can.
* Every training cell skips itself when its checkpoint already exists, so re-running with
  `checkpoints/` present only regenerates the figures (a few minutes). Set `FORCE_RETRAIN=1` to
  retrain regardless.
* `SMOKE=1` runs the whole pipeline with 2 epochs and reduced FID sample counts (~10 min) to
  validate every code path before committing to a full run. It writes to `checkpoints_smoke/` and
  `results_smoke/` so a smoke checkpoint can never be mistaken for a real one.
* `make_partc.py` reads `results/metrics_part{A,B}.json`, so run the notebooks before it.
