#!/usr/bin/env python3
"""
Universal Encoder v1 — OpenSmell
Train encoder with MAE + chemoprint + contrastive + KL losses.

Pre-registered success criterion: mean chemoprint R^2 > 0.7 on test substances.
"""

import os, sys, json, glob, pickle, time, warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
FOODB_CSV = ROOT.parent / "smell-pipeline" / "data" / "foodb_chemoprints.csv"
CACHE_DIR = ROOT / "data" / "cache"
MODEL_DIR = ROOT / "models"
SNAP = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--DeweiFeng--smell-net/snapshots"
)
_revs = sorted(os.listdir(SNAP), reverse=True)
SNAP_DIR = os.path.join(SNAP, _revs[0]) if _revs else ""

SEGMENT_LEN = 100
STRIDE = 50
MASK_RATIO = 0.4
LATENT_DIM = 256
PROJECTION_DIM = 128
CHEMO_DIM = 29
SENSOR_NAMES = ["NO2", "C2H5OH", "VOC", "CO", "Alcohol", "LPG"]
N_SENSORS = len(SENSOR_NAMES)

BATCH_SIZE = 64
LR = 5e-4
WEIGHT_DECAY = 1e-5
CLIP_NORM = 1.0
MAX_EPOCHS = 300
PATIENCE = 20
CONTRAST_TEMP = 0.5

W_MAE = 1.0
W_CHEMO = 0.5
W_CONTRAST = 0.3
W_KL = 0.1

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ── Data ──────────────────────────────────────────────────────────────────────

def parse_filename(fname):
    stem = os.path.splitext(fname)[0]
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[-1].isdigit():
        return "_".join(parts[:-1]), int(parts[-1])
    return stem, 0


def load_and_segment_all(snap_dir, foodb_csv):
    print("=" * 60)
    print("Loading SmellNet data ...")
    foodb = pd.read_csv(foodb_csv, index_col=0)
    foob_substances = set(foodb.index)
    print(f"  FooDB-covered substances: {len(foob_substances)}")

    cache_path = CACHE_DIR / "preprocessed.pkl"
    if cache_path.exists():
        print("  Loading from cache ...")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    all_segments, all_substances, all_sessions, all_chemoprints = [], [], [], []
    pattern = os.path.join(snap_dir, "**/*.csv")
    csv_files = glob.glob(pattern, recursive=True)

    for fpath in csv_files:
        fname = os.path.basename(fpath)
        subj, sess = parse_filename(fname)
        if subj not in foob_substances:
            continue
        try:
            df = pd.read_csv(fpath)
        except Exception:
            continue
        cols = []
        for expected in SENSOR_NAMES:
            found = [c for c in df.columns if c.lower() == expected.lower()]
            cols.append(found[0] if found else None)
        if any(c is None for c in cols):
            continue
        raw = df[cols].values.astype(np.float32)
        N = raw.shape[0]
        if N >= SEGMENT_LEN:
            segments = [raw[i:i + SEGMENT_LEN] for i in range(0, N - SEGMENT_LEN + 1, STRIDE)]
        else:
            segments = [np.pad(raw, ((0, SEGMENT_LEN - N), (0, 0)), mode="edge")]
        cp = foodb.loc[subj].values.astype(np.float32)
        all_segments.append(np.stack(segments))
        n_seg = len(segments)
        all_substances.extend([subj] * n_seg)
        all_sessions.extend([sess] * n_seg)
        all_chemoprints.append(np.tile(cp, (n_seg, 1)))

    X = np.concatenate(all_segments, axis=0)
    Y = np.concatenate(all_chemoprints, axis=0)
    substances = np.array(all_substances)
    sessions = np.array(all_sessions, dtype=np.int32)

    print(f"  Total segments: {X.shape[0]}")
    print(f"  Unique substances: {len(np.unique(substances))}")
    print(f"  Input shape: {X.shape}")

    data = {"X": X, "Y": Y, "substances": substances, "sessions": sessions,
            "substance_names": sorted(np.unique(substances).tolist())}
    with open(cache_path, "wb") as f:
        pickle.dump(data, f)
    return data


def train_val_test_split(data):
    X, Y, substances, sessions = data["X"], data["Y"], data["substances"], data["sessions"]
    subj_sessions = defaultdict(set)
    for s, sess in zip(substances, sessions):
        subj_sessions[s].add(int(sess))

    test_mask = np.zeros(len(substances), dtype=bool)
    for s, sess_set in subj_sessions.items():
        test_sessions = set(sorted(sess_set)[-2:])
        test_mask |= (substances == s) & (np.isin(sessions, list(test_sessions)))

    trainval_mask = ~test_mask
    train_mask, val_mask = np.zeros(len(substances), dtype=bool), np.zeros(len(substances), dtype=bool)
    for s in subj_sessions:
        idx = np.where((substances == s) & trainval_mask)[0]
        np.random.shuffle(idx)
        split_pt = int(0.9 * len(idx))
        train_mask[idx[:split_pt]], val_mask[idx[split_pt:]] = True, True

    print(f"  Train: {train_mask.sum():6d}  Val: {val_mask.sum():6d}  Test: {test_mask.sum():6d}")
    return {
        "X_train": X[train_mask], "Y_train": Y[train_mask],
        "subj_train": substances[train_mask], "sess_train": sessions[train_mask],
        "X_val": X[val_mask], "Y_val": Y[val_mask],
        "subj_val": substances[val_mask], "sess_val": sessions[val_mask],
        "X_test": X[test_mask], "Y_test": Y[test_mask],
        "subj_test": substances[test_mask], "sess_test": sessions[test_mask],
    }


def compute_normalisation(data, key):
    X = data[key]
    if X.ndim == 3:
        mean, std = X.mean(axis=(0, 1)), X.std(axis=(0, 1))
    else:
        mean, std = X.mean(axis=0), X.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


class SmellDataset(Dataset):
    def __init__(self, X, Y, substances, sessions, x_mean, x_std, y_mean, y_std, stl=None):
        self.X = (X - x_mean) / x_std
        self.Y = (Y - y_mean) / y_std
        self.Y_orig = Y.copy()
        self.substances = substances
        self.labels = np.array([stl[s] for s in substances], dtype=np.int64) if stl else np.zeros(len(substances), dtype=np.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (torch.tensor(self.X[idx], dtype=torch.float32),
                torch.tensor(self.Y[idx], dtype=torch.float32),
                torch.tensor(self.labels[idx], dtype=torch.long),
                torch.tensor(self.Y_orig[idx], dtype=torch.float32))


# ── Models ────────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    def __init__(self, in_channels=N_SENSORS, latent_dim=LATENT_DIM):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1, bias=False)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1, bias=False)
        self.pool = nn.MaxPool1d(2)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        for m in [self.conv1, self.conv2, self.conv3]:
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
        for m in [self.fc_mu, self.fc_logvar]:
            nn.init.normal_(m.weight, std=0.001)
            nn.init.zeros_(m.bias)
        nn.init.constant_(self.fc_logvar.bias, -3.0)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        h = x.mean(dim=-1)
        return self.fc_mu(h), torch.clamp(self.fc_logvar(h), -10, 10)


class MAEDecoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent_dim, 256), nn.ReLU(), nn.Linear(256, SEGMENT_LEN * N_SENSORS))

    def forward(self, z):
        return self.net(z).view(-1, SEGMENT_LEN, N_SENSORS)


class ChemoprintHead(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, cd=CHEMO_DIM):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent_dim, 64), nn.ReLU(), nn.Linear(64, cd))

    def forward(self, z):
        return self.net(z)


class ProjectionHead(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent_dim, 128), nn.ReLU(), nn.Linear(128, PROJECTION_DIM))

    def forward(self, z):
        return F.normalize(self.net(z), dim=1)


# ── Losses ────────────────────────────────────────────────────────────────────

def supcon_loss(features, labels, temperature=CONTRAST_TEMP):
    N, device = features.shape[0], features.device
    features = F.normalize(features, dim=1)
    sim = features @ features.T / temperature
    eye = torch.eye(N, device=device)
    sim_masked = sim - eye * 1e9
    log_denom = torch.logsumexp(sim_masked, dim=1)
    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float() - eye
    valid = pos_mask.sum(dim=1) > 0
    if valid.sum() == 0:
        return features.sum() * 0.0
    pos_sim_sum = (pos_mask * sim).sum(dim=1)
    mean_pos = pos_sim_sum / pos_mask.sum(dim=1).clamp(min=1)
    return (log_denom[valid] - mean_pos[valid]).mean()


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_chemoprint(encoder, head, loader, y_mean, y_std, device):
    encoder.eval()
    head.eval()
    preds, trues = [], []
    for x, _, _, yo in loader:
        mu, _ = encoder(x.to(device))
        preds.append(head(mu).cpu().numpy() * y_std + y_mean)
        trues.append(yo.numpy())
    preds, trues = np.concatenate(preds), np.concatenate(trues)
    r2s = []
    for d in range(CHEMO_DIM):
        ss_res = ((trues[:, d] - preds[:, d]) ** 2).sum()
        ss_tot = ((trues[:, d] - trues[:, d].mean()) ** 2).sum()
        r2s.append(1 - ss_res / ss_tot if ss_tot > 0 else 0.0)
    return np.array(r2s)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  OpenSmell Universal Encoder v1")
    print("=" * 60)

    device = torch.device("cpu")
    print(f"  Device: {device}")

    raw_data = load_and_segment_all(SNAP_DIR, str(FOODB_CSV))
    split = train_val_test_split(raw_data)

    x_mean, x_std = compute_normalisation(split, "X_train")
    y_mean, y_std = compute_normalisation(split, "Y_train")
    all_subj = sorted(np.unique(raw_data["substances"]).tolist())
    stl = {s: i for i, s in enumerate(all_subj)}

    train_ds = SmellDataset(split["X_train"], split["Y_train"], split["subj_train"],
                            split["sess_train"], x_mean, x_std, y_mean, y_std, stl)
    val_ds = SmellDataset(split["X_val"], split["Y_val"], split["subj_val"],
                          split["sess_val"], x_mean, x_std, y_mean, y_std, stl)
    test_ds = SmellDataset(split["X_test"], split["Y_test"], split["subj_test"],
                           split["sess_test"], x_mean, x_std, y_mean, y_std, stl)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    y_mean_np, y_std_np = y_mean.astype(np.float32), y_std.astype(np.float32)
    for name, arr in [("X_train", split["X_train"]), ("Y_train", split["Y_train"]),
                       ("X_val", split["X_val"]), ("X_test", split["X_test"])]:
        assert not np.any(np.isnan(arr)), f"NaN in {name}"

    encoder = Encoder().to(device)
    decoder = MAEDecoder().to(device)
    chemoprint_head = ChemoprintHead().to(device)
    proj_head = ProjectionHead().to(device)

    for n, p in encoder.named_parameters():
        assert not torch.isnan(p).any(), f"NaN in encoder.{n} at init"

    params = list(encoder.parameters()) + list(decoder.parameters()) + list(chemoprint_head.parameters()) + list(proj_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-6)

    print(f"\nTraining ({sum(p.numel() for p in encoder.parameters()):,} encoder params)")
    print("=" * 60)

    history = []
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        encoder.train()
        decoder.train()
        chemoprint_head.train()
        proj_head.train()
        kl_weight = W_KL * min(1.0, epoch / 50.0)
        losses = {"total": 0, "mae": 0, "chemo": 0, "contrast": 0, "kl": 0}
        n_batches = 0

        for x, y, labels, _ in train_loader:
            x, y, labels = x.to(device), y.to(device), labels.to(device)
            mu, logvar_enc = encoder(x)
            std = torch.exp(0.5 * logvar_enc)
            z = mu + torch.randn_like(std) * std

            recon = decoder(z)
            chemo_pred = chemoprint_head(z)
            proj = proj_head(z)

            loss_mae = F.mse_loss(recon, x)
            loss_chemo = F.mse_loss(chemo_pred, y)
            loss_contrast = supcon_loss(proj, labels)
            loss_kl = -0.5 * (1 + logvar_enc - mu.pow(2) - logvar_enc.exp()).sum(dim=1).mean()
            loss_total = W_MAE * loss_mae + W_CHEMO * loss_chemo + W_CONTRAST * loss_contrast + kl_weight * loss_kl

            optimizer.zero_grad()
            loss_total.backward()
            nn.utils.clip_grad_norm_(params, CLIP_NORM)
            optimizer.step()

            losses["total"] += loss_total.item()
            losses["mae"] += loss_mae.item()
            losses["chemo"] += loss_chemo.item()
            losses["contrast"] += loss_contrast.item()
            losses["kl"] += loss_kl.item()
            n_batches += 1

        encoder.eval()
        decoder.eval()
        chemoprint_head.eval()
        val_total = 0
        val_n = 0
        with torch.no_grad():
            for x, y, labels, _ in val_loader:
                x, y, labels = x.to(device), y.to(device), labels.to(device)
                mu, logvar_enc = encoder(x)
                z = mu + torch.randn_like(torch.exp(0.5 * logvar_enc)) * torch.exp(0.5 * logvar_enc)
                recon = decoder(z)
                chemo_pred = chemoprint_head(z)
                proj = proj_head(z)
                l1 = F.mse_loss(recon, x)
                l2 = F.mse_loss(chemo_pred, y)
                l3 = supcon_loss(proj, labels)
                l4 = -0.5 * (1 + logvar_enc - mu.pow(2) - logvar_enc.exp()).sum(dim=1).mean()
                val_total += (W_MAE * l1 + W_CHEMO * l2 + W_CONTRAST * l3 + W_KL * l4).item()
                val_n += 1

        val_loss = val_total / max(val_n, 1)
        scheduler.step(val_loss)
        avg = {k: v / n_batches for k, v in losses.items()}

        log = f"  Epoch {epoch:3d} | tot:{avg['total']:.3f} mae:{avg['mae']:.3f} chemo:{avg['chemo']:.3f} cont:{avg['contrast']:.3f} kl:{avg['kl']:.3f} val:{val_loss:.3f}"
        if epoch % 10 == 0:
            r2 = evaluate_chemoprint(encoder, chemoprint_head, val_loader, y_mean_np, y_std_np, device)
            log += f" | val R2:{r2.mean():.3f}"
        print(log)
        history.append({"epoch": epoch, **avg, "val_loss": val_loss, "lr": optimizer.param_groups[0]["lr"]})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(encoder.state_dict(), MODEL_DIR / "encoder_v1_best.pth")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    # Load best
    encoder.load_state_dict(torch.load(MODEL_DIR / "encoder_v1_best.pth", map_location=device, weights_only=True))

    # Test
    print("\nTest Evaluation")
    print("=" * 60)
    test_r2s = evaluate_chemoprint(encoder, chemoprint_head, test_loader, y_mean_np, y_std_np, device)
    mean_r2, median_r2 = float(test_r2s.mean()), float(np.median(test_r2s))
    print(f"  Mean chemoprint R2: {mean_r2:.4f}")
    print(f"  Median chemoprint R2: {median_r2:.4f}")

    # Save
    torch.save(encoder.state_dict(), MODEL_DIR / "encoder_v1.pth")
    torch.save(chemoprint_head.state_dict(), MODEL_DIR / "chemoprint_head_v1.pth")

    report = {
        "mean_r2": round(mean_r2, 4), "median_r2": round(median_r2, 4),
        "r2_per_dimension": [round(float(r), 4) for r in test_r2s],
        "n_train": len(train_ds), "n_val": len(val_ds), "n_test": len(test_ds),
        "success": mean_r2 > 0.7, "training_history": history,
    }
    with open(MODEL_DIR / "training_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved to {MODEL_DIR / 'training_report.json'}")

    # Update README
    readme_path = ROOT / "README.md"
    if readme_path.exists():
        with open(readme_path) as f:
            readme = f.read()
        old = "> **Status:** IN DEVELOPMENT. Skeleton only."
        new = f"> **Status:** {'PROVEN' if mean_r2 > 0.7 else 'NOT PROVEN'}. Mean chemoprint R2 = {mean_r2:.3f} (target > 0.7)."
        readme = readme.replace(old, new)
        with open(readme_path, "w") as f:
            f.write(readme)
        print(f"  README updated")

    return report


if __name__ == "__main__":
    report = main()
    if report["success"]:
        print("\n  SUCCESS: Mean R2 > 0.7")
    else:
        print(f"\n  NOT PROVEN: Mean R2 = {report['mean_r2']:.4f}")
