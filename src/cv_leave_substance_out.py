#!/usr/bin/env python3
"""
Leave-11-Substances-Out Cross-Validation — OpenSmell Universal Encoder
Splits 44 substances into 4 folds of 11, trains on 33, tests on 11.
Reports mean R2 across all held-out substances.
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

N_FOLDS = 4
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cpu")
print(f"Device: {device}")
print(f"Leave-{44 // N_FOLDS}-Substances-Out Cross-Validation ({N_FOLDS} folds)")


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

    return X, Y, substances, sessions


def compute_normalisation(X_train):
    mean = X_train.mean(axis=(0, 1))
    std = X_train.std(axis=(0, 1))
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


class SmellCVSubjDataset(Dataset):
    def __init__(self, X, Y, substances, x_mean, x_std, y_mean, y_std, stl=None):
        self.X = (X - x_mean) / x_std
        self.Y = (Y - y_mean) / y_std
        self.Y_orig = Y.copy()
        if stl:
            self.labels = np.array([stl.get(s, -1) for s in substances], dtype=np.int64)
        else:
            self.labels = np.zeros(len(substances), dtype=np.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (torch.tensor(self.X[idx], dtype=torch.float32),
                torch.tensor(self.Y[idx], dtype=torch.float32),
                torch.tensor(self.labels[idx], dtype=torch.long),
                torch.tensor(self.Y_orig[idx], dtype=torch.float32))


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
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(LATENT_DIM, 256), nn.ReLU(), nn.Linear(256, SEGMENT_LEN * N_SENSORS))

    def forward(self, z):
        return self.net(z).view(-1, SEGMENT_LEN, N_SENSORS)


class ChemoprintHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(LATENT_DIM, 64), nn.ReLU(), nn.Linear(64, CHEMO_DIM))

    def forward(self, z):
        return self.net(z)


class ProjectionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(LATENT_DIM, 128), nn.ReLU(), nn.Linear(128, PROJECTION_DIM))

    def forward(self, z):
        return F.normalize(self.net(z), dim=1)


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


@torch.no_grad()
def evaluate_chemoprint(encoder, head, loader, y_mean, y_std):
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


def train_fold(train_subjs, test_subjs, all_X, all_Y, all_substances,
               all_sessions, fold_idx):
    print(f"\n{'=' * 60}")
    print(f"  Fold {fold_idx + 1}/{N_FOLDS}: {len(train_subjs)} train, {len(test_subjs)} test substances")
    print(f"  Test substances: {', '.join(sorted(test_subjs))}")
    print(f"{'=' * 60}")

    train_mask = np.isin(all_substances, list(train_subjs))
    test_mask = np.isin(all_substances, list(test_subjs))

    X_train_all = all_X[train_mask]
    Y_train_all = all_Y[train_mask]
    subj_train_all = all_substances[train_mask]
    sess_train_all = all_sessions[train_mask]

    X_test = all_X[test_mask]
    Y_test = all_Y[test_mask]
    subj_test = all_substances[test_mask]

    # Split train into train/val (90/10 within training substances)
    subj_sessions = defaultdict(set)
    for s, sess in zip(subj_train_all, sess_train_all):
        subj_sessions[s].add(int(sess))

    val_mask = np.zeros(len(X_train_all), dtype=bool)
    for s in train_subjs:
        sess_set = sorted(subj_sessions[s])
        val_sessions = set(sess_set[-1:]) if len(sess_set) >= 1 else set()
        val_mask |= (subj_train_all == s) & (np.isin(sess_train_all, list(val_sessions)))

    # If no sessions marked as val (single session substances), mark 10% of segments
    if val_mask.sum() == 0:
        np.random.seed(SEED + fold_idx)
        val_count = max(1, int(0.1 * len(X_train_all)))
        val_idx = np.random.choice(len(X_train_all), val_count, replace=False)
        val_mask = np.zeros(len(X_train_all), dtype=bool)
        val_mask[val_idx] = True

    X_train = X_train_all[~val_mask]
    Y_train = Y_train_all[~val_mask]
    subj_train = subj_train_all[~val_mask]

    X_val = X_train_all[val_mask]
    Y_val = Y_train_all[val_mask]
    subj_val = subj_train_all[val_mask]

    print(f"  Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)} segments")

    # Normalisation
    x_mean, x_std = compute_normalisation(X_train)
    y_mean, y_std = Y_train.mean(axis=0), Y_train.std(axis=0)
    y_std = np.where(y_std < 1e-8, 1.0, y_std)

    all_subj_list = sorted(np.unique(subj_train).tolist())
    stl = {s: i for i, s in enumerate(all_subj_list)}

    train_ds = SmellCVSubjDataset(X_train, Y_train, subj_train, x_mean, x_std, y_mean, y_std, stl)
    val_ds = SmellCVSubjDataset(X_val, Y_val, subj_val, x_mean, x_std, y_mean, y_std, stl)
    test_ds = SmellCVSubjDataset(X_test, Y_test, subj_test, x_mean, x_std, y_mean, y_std, stl)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    y_mean_np = y_mean.astype(np.float32)
    y_std_np = y_std.astype(np.float32)

    encoder = Encoder().to(device)
    decoder = MAEDecoder().to(device)
    chemoprint_head = ChemoprintHead().to(device)
    proj_head = ProjectionHead().to(device)

    params = list(encoder.parameters()) + list(decoder.parameters()) + list(chemoprint_head.parameters()) + list(proj_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-6)

    best_val_loss = float("inf")
    patience_counter = 0
    history = []
    start_time = time.time()

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

        log = f"  Fold{fold_idx+1} Epoch {epoch:3d} | tot:{avg['total']:.3f} mae:{avg['mae']:.3f} chemo:{avg['chemo']:.3f} cont:{avg['contrast']:.3f} kl:{avg['kl']:.3f} val:{val_loss:.3f}"
        if epoch % 10 == 0:
            r2 = evaluate_chemoprint(encoder, chemoprint_head, val_loader, y_mean_np, y_std_np)
            log += f" | val R2:{r2.mean():.3f}"
        print(log)
        history.append({"epoch": epoch, **avg, "val_loss": val_loss, "lr": optimizer.param_groups[0]["lr"]})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    elapsed = time.time() - start_time

    # Evaluate on held-out test substances
    test_r2s = evaluate_chemoprint(encoder, chemoprint_head, test_loader, y_mean_np, y_std_np)
    mean_r2 = float(test_r2s.mean())
    median_r2 = float(np.median(test_r2s))

    print(f"\n  Fold {fold_idx + 1} results:")
    print(f"  Test substances: {', '.join(sorted(test_subjs))}")
    print(f"  Mean R2: {mean_r2:.4f}, Median R2: {median_r2:.4f}")
    print(f"  Per-dim R2: {[round(float(r), 4) for r in test_r2s]}")
    print(f"  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f}m)")

    return {
        "fold": fold_idx + 1,
        "test_substances": sorted(test_subjs),
        "mean_r2": round(mean_r2, 4),
        "median_r2": round(median_r2, 4),
        "r2_per_dimension": [round(float(r), 4) for r in test_r2s],
        "epochs_trained": len(history),
        "elapsed_seconds": round(elapsed, 1),
        "history": history,
    }


def main():
    print("=" * 60)
    print("  OpenSmell Universal Encoder — Leave-Substance-Out CV")
    print("=" * 60)

    all_X, all_Y, all_substances, all_sessions = load_and_segment_all(SNAP_DIR, str(FOODB_CSV))

    unique_substances = sorted(np.unique(all_substances).tolist())
    print(f"  Total unique substances: {len(unique_substances)}")

    # Stratified k-fold by substance
    rng = np.random.RandomState(SEED)
    shuffled = unique_substances.copy()
    rng.shuffle(shuffled)
    fold_size = len(shuffled) // N_FOLDS
    folds = [set(shuffled[i * fold_size:(i + 1) * fold_size]) for i in range(N_FOLDS)]
    # Add remainder to last fold
    remainder = shuffled[N_FOLDS * fold_size:]
    for i, s in enumerate(remainder):
        folds[-1].add(s)

    all_fold_results = []
    all_r2s = []

    for fold_idx, test_subjs in enumerate(folds):
        train_subjs = set(unique_substances) - test_subjs
        result = train_fold(train_subjs, test_subjs, all_X, all_Y, all_substances,
                            all_sessions, fold_idx)
        all_fold_results.append(result)
        all_r2s.append(result["mean_r2"])

    print("\n" + "=" * 60)
    print("  CROSS-VALIDATION SUMMARY")
    print("=" * 60)
    for i, res in enumerate(all_fold_results):
        print(f"  Fold {res['fold']}: mean R2 = {res['mean_r2']:.4f} (median: {res['median_r2']:.4f})")
        print(f"    Test: {', '.join(res['test_substances'])}")

    overall_mean = float(np.mean(all_r2s))
    overall_std = float(np.std(all_r2s))
    print(f"\n  Overall mean R2: {overall_mean:.4f} ± {overall_std:.4f} (across {N_FOLDS} folds)")

    # Per-dimension R2 across all test data (pooled)
    print(f"\n  Per-dimension R2 (pooled across folds):")
    pooled_r2s = np.array([res["r2_per_dimension"] for res in all_fold_results])
    for d in range(CHEMO_DIM):
        dim_r2s = pooled_r2s[:, d]
        print(f"    Dim {d:2d}: {dim_r2s.mean():.4f} ± {dim_r2s.std():.4f}")

    # Save report
    report = {
        "n_folds": N_FOLDS,
        "overall_mean_r2": round(overall_mean, 4),
        "overall_std_r2": round(overall_std, 4),
        "folds": all_fold_results,
    }
    report_path = MODEL_DIR / "leave_substance_out_cv_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved to {report_path}")

    return report


if __name__ == "__main__":
    report = main()
