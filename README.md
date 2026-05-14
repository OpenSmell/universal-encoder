# Universal Encoder for Electronic Nose Data

> **Status:** IN DEVELOPMENT. Skeleton only.
> Proof of concept at [`opensmell/session-invariance`](https://github.com/opensmell/session-invariance)

## Overview

The universal encoder is the core of the OpenSmell standard. It maps raw e-nose data to a **256-dimensional latent space** that is:

- **Task-agnostic** — captures all information in the sensor waveform, not just classification targets
- **Session-invariant** — same substance on different days maps to the same region
- **Device-invariant** — different hardware (MQ series, BME, MiCS) maps to the same region
- **Versioned** — once trained, the encoder is frozen and never changed

## Training approach

1. **MAE pretraining** — Masked autoencoder on raw sensor waveforms. The encoder learns to reconstruct masked time steps from unmasked context, capturing chemical information without labels.
2. **Contrastive fine-tuning** — Pull same-substance latents together, push different-substance latents apart. Uses the SmellNet dataset.
3. **Domain-adversarial training** — A domain discriminator tries to predict session/device from latents; the encoder is trained to fool it. This is the key to invariance.

## Usage (once trained)

```python
from universal_encoder import load_encoder

encoder = load_encoder("models/encoder.pth")
latent = encoder.encode(raw_sensor_csv)  # → (256,) numpy vector
```

Downstream heads (chemoprint decoder, classifiers, apps) are trained on top of the **frozen** encoder. They never modify it.

## Directory structure

```
universal-encoder/
├── README.md
├── data/           → training data references / symlinks
├── models/         → trained encoder checkpoints
├── src/
│   ├── mae_pretrain.py         → MAE pretraining (placeholder)
│   ├── contrastive_finetune.py → contrastive + domain-adversarial (placeholder)
│   └── train_chemoprint_head.py → chemoprint decoder on frozen encoder (placeholder)
├── tests/
└── notebooks/
```

## Relationship to other repos

| Repo | Role |
|------|------|
| `session-invariance` | Proof that latent spaces can be session-invariant (81.8% held-out accuracy) |
| `chemoprint` | Ground-truth 29-dim physicochemical descriptor (from SMILES) |
| `smell-pipeline` | Data pipeline utilities (FooDB extraction, etc.) |
| `data-commons` | Standard format for contributed e-nose datasets |
