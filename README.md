# Universal Encoder for Electronic Nose Data

> **Status:** PROVEN for known substances. Mean chemoprint R² = 0.892 on held-out sessions of 44 training substances. NOT PROVEN for novel substances (leave-substance-out R² < 0 — see below).

## What this does

Maps 6-sensor e-nose data (SmellNet, 100 time steps × 6 MOX sensors) to a 256-dim latent space and a 29-dim chemoprint. Trained simultaneously on reconstruction (MAE), contrastive (SupCon), and chemoprediction (MSE) losses.

## What this does NOT do

- **Novel substance identification.** Leave-11-substances-out cross-validation gives mean R² = -7.79 (worse than guessing the mean). The encoder can re-identify substances it saw during training (R² = 0.892) but cannot generalise to unseen substances. The 44-food dataset is too small for genuine chemistry learning.
- **Device-invariance.** All data comes from a single sensor board. Cross-device generalisation requires multi-device data and domain-adversarial training — not yet done.
- **Chirality or vibrational mode prediction.** 29 structural dimensions only. The 6 broad-spectrum MOX sensors cannot distinguish chiral enantiomers or trace compounds below detection limits.
- **Environmental or industrial samples.** Trained on 44 food substances from the SmellNet + FooDB overlap only.

## Status

| Claim | Result | Honest statement |
|-------|--------|-----------------|
| Session invariance | R² = 0.892 | Works when the substance was seen during training |
| Substance generalisation | R² = -7.79 | Does NOT work. Current architecture memorises substances |
| Device invariance | Not tested | Requires multi-device data |

## Leave-substance-out cross-validation

4-fold CV by substance (33 train, 11 test per fold):

| Fold | Test substances | Mean R² |
|------|----------------|---------|
| 1 | apple, asparagus, banana, broccoli, cauliflower, lemon, mandarin_orange, pili_nut, potato, radish, strawberry | -0.87 |
| 2 | brussel_sprouts, chervil, cinnamon, cloves, coriander, dill, mango, mint, nutmeg, oregano, sweet_potato | -1.81 |
| 3 | allspice, almond, angelica, avocado, cashew, ginger, mustard, peach, pear, star_anise, tomato | -0.10 |
| 4 | brazil_nut, cabbage, chives, cumin, garlic, hazelnut, kiwi, mugwort, pineapple, saffron, turnip | -28.39 |

All folds negative. Fold 4 is particularly bad (cabbage-family and spices), suggesting dataset bias. The encoder memorises substance-specific sensor signatures rather than learning generalisable chemistry.

## Why substance generalisation fails

1. **44 substances is insufficient.** The contrastive loss pulls same-substance latents together, which encourages substance-specific memorisation. With only 33 training substances, the latent space is shaped by a tiny fraction of chemical space.
2. **6 MOX sensors lack chemical resolution.** These are broad-spectrum sensors (NO2, ethanol, VOC, CO, alcohol, LPG). They respond to overall volatile concentration, not specific functional groups.
3. **FooDB chemoprints are computed, not measured.** They come from SMILES structures using RDKit descriptors. A food's true volatile profile may differ significantly.

## Training approach

1. **MAE reconstruction** — Full-window autoencoder. Encoder → latent → decoder reconstructs 100 × 6 sensor window.
2. **Contrastive (SupCon)** — Pulls same-substance latents together, pushes different-substance latents apart. Temperature = 0.5.
3. **Chemoprint head** — 2-layer MLP (256 → 64 → 29) maps latent to chemoprint. Trained with MSE.
4. **KL divergence** — VAE-style with linear annealing (0 → 0.1 over 50 epochs) to regularise latent space.

## Reproduce

```bash
conda activate odor
python src/train_encoder.py
```

Expected: R² = 0.892 ± 0.05 on held-out sessions. If it doesn't reproduce, file an issue.

```bash
python src/cv_leave_substance_out.py
```

Expected: negative R² across all 4 folds.

## Next steps

1. Collect 100s–1000s of additional substances for genuine chemistry learning
2. Multi-device data collection for device-invariance
3. Replace contrastive loss with chemistry-predictive loss that forces generalisation
4. Weighted loss on weak chemoprint dimensions (dim 16 R² = 0.43)

## Relationship to other repos

| Repo | Role |
|------|------|
| `opensmell` | Pip-installable SDK wrapping this encoder |
| `chemoprint` | Ground-truth 29-dim physicochemical descriptor |
| `data-commons` | Standard format for contributed e-nose datasets |
| `session-invariance` | Proof that latent spaces can be session-invariant |
| `affine-calibration-failed` | Documented dead end (affine calibration insufficient) |
