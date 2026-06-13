"""
hparam_search.py

Dedicated hyperparameter tuning for individual models.
Run AFTER `train_section_lstm_full.py --mode embed` (seqs cache must exist).

Supported models:
  --model m1   Structured only (LightGBM on structured features)
  --model m2   Struct + mean BERT note (LightGBM on struct + ClinicalBERT mean-pool)
  --model m3   Struct + last BERT note (LightGBM on struct + ClinicalBERT last note)
  --model m5   MoE residual (LSTM encoder + Expert A/B + gate — full pipeline)

Each model gets its own Optuna study with a model-specific search space.
Best params are saved to <output>/best_params_<model>.json.

Usage:
  # Tune M1 (fast, ~5 min for 40 trials)
  python mimic3_dataset/hparam_search.py --model m1 --trials 40

  # Tune M2 / M3 (moderate, ~10 min each)
  python mimic3_dataset/hparam_search.py --model m2 --trials 40
  python mimic3_dataset/hparam_search.py --model m3 --trials 40

  # Tune M5 (expensive — trains LSTM per trial, ~2-4 h for 30 trials)
  python mimic3_dataset/hparam_search.py --model m5 --trials 30 --sample 3000

  # Custom output dir / seqs cache
  python mimic3_dataset/hparam_search.py --model m1 \\
      --output outputs_mimic3_lstm_full_gated_residual \\
      --seqs-cache processed/section_trajectory_long/seqs_clinicalbert_24h.pkl
"""

import os
import json
import pickle
import argparse
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import lightgbm as lgb
import joblib
import optuna

from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder

from utils import (
    LSTMWithAttention, pad_seq, set_seed,
    NURSING_SECTIONS, RADIOLOGY_SECTIONS, NURSING_OTHER_SECTIONS, SECTIONS_TO_USE,
)

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# =====================================================
# Args
# =====================================================

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["m1", "m2", "m3", "m5"], required=True,
                    help="Which model to tune")
parser.add_argument("--trials", type=int, default=40, help="Optuna trials")
parser.add_argument("--sample", type=int, default=3000, help="patients for hparam search")
parser.add_argument("--output", type=str, default="outputs_mimic3_lstm_full_gated_residual")
parser.add_argument("--seqs-cache", type=str,
                    default="processed/section_trajectory_long/seqs_clinicalbert_24h.pkl")
parser.add_argument("--max-notes", type=int, default=8)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

os.makedirs(args.output, exist_ok=True)
set_seed(args.seed)

SEED = args.seed
STRUCT_PATH = "processed/mimic3_structured_24h_clean.csv"
NURSING_PARQUET = "processed/section_trajectory_long/nursing_sections_24h_long.parquet"
NURSING_OTHER_PARQUET = "processed/section_trajectory_long/nursing_other_sections_24h_long.parquet"
RADIOLOGY_PARQUET = "processed/section_trajectory_long/radiology_24h_long.parquet"

device = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)
print(f"Device: {device}")


# =====================================================
# Data helpers (self-contained; no argparse side effects)
# =====================================================

def load_structured(hadm_ids=None):
    df = pd.read_csv(STRUCT_PATH)
    df["hadm_id"] = df["hadm_id"].astype(int)
    df["admittime"] = pd.to_datetime(df["admittime"])
    df = df[(df["los_days"] >= 0.5) & (df["los_days"] <= 365)].copy()

    if hadm_ids is not None:
        df = df[df["hadm_id"].isin(hadm_ids)].copy()

    feat_cols = [c for c in df.columns if c not in ("hadm_id", "admittime", "los_days")]

    for c in df[feat_cols].select_dtypes(include="object").columns:
        df[c] = LabelEncoder().fit_transform(df[c].fillna("missing").astype(str))

    df[feat_cols] = df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    return df, feat_cols


def temporal_split(df, train_frac=0.70, val_frac=0.10):
    df_s = df.sort_values("admittime").reset_index(drop=True)
    n = len(df_s)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_hadm = set(df_s.iloc[:n_train]["hadm_id"])
    val_hadm = set(df_s.iloc[n_train:n_train + n_val]["hadm_id"])
    test_hadm = set(df_s.iloc[n_train + n_val:]["hadm_id"])
    print(f"  Split — train:{len(train_hadm)}  val:{len(val_hadm)}  test:{len(test_hadm)}")
    return train_hadm, val_hadm, test_hadm


def load_seqs_cache(cache_path, hadm_set):
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"Seqs cache not found: {cache_path}\n"
            "Run `python mimic3_dataset/train_section_lstm_full.py --mode embed` first."
        )
    print(f"Loading seqs cache: {cache_path}")
    with open(cache_path, "rb") as f:
        seqs_all = pickle.load(f)
    seqs = {
        sec: {hid: arr for hid, arr in seqs_all[sec].items() if hid in hadm_set}
        for sec in SECTIONS_TO_USE
    }
    return seqs


def struct_matrix(hadm_ids, hadm_to_feat, feat_cols):
    return np.array(
        [[hadm_to_feat[h][c] for c in feat_cols] for h in hadm_ids],
        dtype=np.float32,
    )


# =====================================================
# LSTM helpers (needed for M5)
# =====================================================

class SeqDataset(Dataset):
    def __init__(self, seqs, sec_idx, labels):
        self.X = torch.tensor(np.stack(seqs), dtype=torch.float32)
        self.sec = torch.tensor(sec_idx, dtype=torch.long)
        self.y = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.sec[i], self.y[i]


def build_lstm_samples(seqs, hadm_set, hadm_to_los, max_notes):
    xs, secs, ys = [], [], []
    for j, sec in enumerate(SECTIONS_TO_USE):
        for hid, arr in seqs[sec].items():
            if hid not in hadm_set:
                continue
            xs.append(pad_seq(arr, max_notes))
            secs.append(j)
            ys.append(hadm_to_los[hid])
    return xs, secs, ys


def train_lstm(seqs, train_hadm, val_hadm, hadm_to_los, params, max_notes):
    hidden_dim   = params["hidden_dim"]
    sec_emb_dim  = params["sec_emb_dim"]
    dropout      = params["dropout"]
    lr           = params["lr"]
    epochs       = params["epochs"]
    batch_size   = params["batch_size"]

    xs, scs, ys = build_lstm_samples(seqs, train_hadm, hadm_to_los, max_notes)
    if len(xs) == 0:
        raise ValueError("No LSTM training samples — check seqs cache and section mapping.")

    loader = DataLoader(SeqDataset(xs, scs, ys), batch_size=batch_size, shuffle=True)

    model = LSTMWithAttention(
        input_dim=768,
        n_sections=len(SECTIONS_TO_USE),
        sec_emb_dim=sec_emb_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best_val_loss, best_state = float("inf"), None

    for epoch in range(epochs):
        model.train()
        for xb, sb, yb in loader:
            xb, sb, yb = xb.to(device), sb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(model(xb, sb), yb).backward()
            opt.step()

        model.eval()
        val_xs, val_scs, val_ys = build_lstm_samples(seqs, val_hadm, hadm_to_los, max_notes)
        if val_xs:
            with torch.no_grad():
                vl = loss_fn(
                    model(
                        torch.tensor(np.stack(val_xs), dtype=torch.float32).to(device),
                        torch.tensor(val_scs, dtype=torch.long).to(device),
                    ),
                    torch.tensor(val_ys, dtype=torch.float32).to(device),
                ).item()
            if vl < best_val_loss:
                best_val_loss = vl
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def extract_lstm_features(seqs, hadm_ids, model, max_notes, batch_size=512):
    model.eval()
    n_sec = len(SECTIONS_TO_USE)
    hidden_size = model.lstm.hidden_size
    feats = np.zeros((len(hadm_ids), n_sec * hidden_size), dtype=np.float32)

    items = []
    for i, hid in enumerate(hadm_ids):
        for j, sec in enumerate(SECTIONS_TO_USE):
            arr = seqs[sec].get(int(hid))
            if arr is not None:
                items.append((i, j, pad_seq(arr, max_notes)))

    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        row_ids = [b[0] for b in batch]
        sec_ids = [b[1] for b in batch]
        xb = torch.tensor(np.stack([b[2] for b in batch]), dtype=torch.float32).to(device)
        sb = torch.tensor(sec_ids, dtype=torch.long).to(device)
        enc = model.encode(xb, sb).cpu().numpy()
        for k, (row_i, sec_j) in enumerate(zip(row_ids, sec_ids)):
            feats[row_i, sec_j * hidden_size:(sec_j + 1) * hidden_size] = enc[k]
    return feats


@torch.no_grad()
def extract_static_note_features(seqs, hadm_ids, mode="last"):
    n_sec = len(SECTIONS_TO_USE)
    feats = np.zeros((len(hadm_ids), n_sec * 768), dtype=np.float32)
    for i, hid in enumerate(hadm_ids):
        for j, sec in enumerate(SECTIONS_TO_USE):
            arr = seqs[sec].get(int(hid))
            if arr is None or len(arr) == 0:
                continue
            vec = arr[-1] if mode == "last" else arr.mean(axis=0)
            feats[i, j * 768:(j + 1) * 768] = vec.astype(np.float32)
    return feats


def train_lgbm(X_tr, y_tr, params, X_val=None, y_val=None):
    m = lgb.LGBMRegressor(
        n_estimators=params.get("lgbm_iters", 500),
        learning_rate=params.get("lgbm_lr", 0.05),
        num_leaves=params.get("lgbm_leaves", 63),
        min_child_samples=params.get("lgbm_min_child_samples", 20),
        reg_lambda=params.get("lgbm_l2", 1.0),
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    )
    if X_val is not None:
        m.fit(X_tr, y_tr,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(50, verbose=False)])
    else:
        m.fit(X_tr, y_tr)
    return m


# =====================================================
# Load data once (shared across all trials)
# =====================================================

print(f"\nTuning model: {args.model.upper()}  |  trials={args.trials}  |  sample={args.sample}")
print(f"Loading {args.sample}-patient sample...")

df_full, feat_cols = load_structured()
df_full = df_full.sort_values("admittime")

df_full["los_q"] = pd.qcut(df_full["los_days"], q=4, labels=False, duplicates="drop")
sample = (
    df_full.groupby("los_q", group_keys=False)
    .apply(lambda g: g.sample(min(len(g), args.sample // 4), random_state=SEED))
    .head(args.sample)
    .copy()
)
print(f"  Sample size: {len(sample)}")

train_h, val_h, _ = temporal_split(sample)
all_h = set(sample["hadm_id"])

hadm_to_los = dict(zip(sample["hadm_id"], sample["los_days"]))
hadm_to_feat = sample.set_index("hadm_id")[feat_cols].to_dict("index")

tr_ids = list(train_h)
va_ids = list(val_h)

Xtr_struct = struct_matrix(tr_ids, hadm_to_feat, feat_cols)
Xva_struct = struct_matrix(va_ids, hadm_to_feat, feat_cols)
y_tr = np.array([hadm_to_los[h] for h in tr_ids], dtype=np.float32)
y_va = np.array([hadm_to_los[h] for h in va_ids], dtype=np.float32)

# Load seqs only when needed (M2, M3, M5)
seqs = None
if args.model in ("m2", "m3", "m5"):
    print("Loading seqs cache...")
    seqs_all = load_seqs_cache(args.seqs_cache, all_h)
    seqs = {sec: {hid: arr for hid, arr in seqs_all[sec].items() if hid in all_h}
            for sec in SECTIONS_TO_USE}

# Pre-extract static BERT features for M2/M3 (avoids re-extraction per trial)
Xtr_mean = Xva_mean = Xtr_last = Xva_last = None
if args.model == "m2":
    print("Pre-extracting mean-note features...")
    Xtr_mean = extract_static_note_features(seqs, tr_ids, mode="mean")
    Xva_mean = extract_static_note_features(seqs, va_ids, mode="mean")
    print(f"  Mean-note features: {Xtr_mean.shape}")

if args.model == "m3":
    print("Pre-extracting last-note features...")
    Xtr_last = extract_static_note_features(seqs, tr_ids, mode="last")
    Xva_last = extract_static_note_features(seqs, va_ids, mode="last")
    print(f"  Last-note features: {Xtr_last.shape}")


# =====================================================
# Objectives
# =====================================================

def _lgbm_space(trial, prefix=""):
    return {
        f"{prefix}lgbm_iters":             trial.suggest_int(f"{prefix}lgbm_iters", 200, 1000, step=100),
        f"{prefix}lgbm_lr":                trial.suggest_float(f"{prefix}lgbm_lr", 0.01, 0.1, log=True),
        f"{prefix}lgbm_leaves":            trial.suggest_categorical(f"{prefix}lgbm_leaves", [31, 63, 127, 255]),
        f"{prefix}lgbm_l2":                trial.suggest_float(f"{prefix}lgbm_l2", 0.0, 2.0),
        f"{prefix}lgbm_min_child_samples": trial.suggest_categorical(f"{prefix}lgbm_min_child_samples", [10, 20, 50]),
    }


def objective_m1(trial):
    params = _lgbm_space(trial)
    try:
        m = train_lgbm(Xtr_struct, y_tr, params, X_val=Xva_struct, y_val=y_va)
        return mean_absolute_error(y_va, m.predict(Xva_struct))
    except Exception as e:
        print(f"  Trial failed: {e}")
        return 999.0


def objective_m2(trial):
    params = _lgbm_space(trial)
    try:
        Xtr = np.hstack([Xtr_struct, Xtr_mean])
        Xva = np.hstack([Xva_struct, Xva_mean])
        m = train_lgbm(Xtr, y_tr, params, X_val=Xva, y_val=y_va)
        return mean_absolute_error(y_va, m.predict(Xva))
    except Exception as e:
        print(f"  Trial failed: {e}")
        return 999.0


def objective_m3(trial):
    params = _lgbm_space(trial)
    try:
        Xtr = np.hstack([Xtr_struct, Xtr_last])
        Xva = np.hstack([Xva_struct, Xva_last])
        m = train_lgbm(Xtr, y_tr, params, X_val=Xva, y_val=y_va)
        return mean_absolute_error(y_va, m.predict(Xva))
    except Exception as e:
        print(f"  Trial failed: {e}")
        return 999.0


def objective_m5(trial):
    lstm_params = {
        "hidden_dim":  trial.suggest_categorical("hidden_dim", [32, 64, 128]),
        "sec_emb_dim": trial.suggest_categorical("sec_emb_dim", [16, 32, 64]),
        "dropout":     trial.suggest_float("dropout", 0.0, 0.3, step=0.1),
        "lr":          trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "epochs":      trial.suggest_int("epochs", 5, 20),
        "batch_size":  trial.suggest_categorical("batch_size", [128, 256]),
    }
    tail_pct = trial.suggest_categorical("tail_pct", [80, 85, 90, 95])

    # Expert A/B params (shared search space, independent values)
    expA_params = {
        "lgbm_iters":             trial.suggest_int("expA_iters", 100, 500, step=50),
        "lgbm_lr":                trial.suggest_float("expA_lr", 0.01, 0.1, log=True),
        "lgbm_leaves":            trial.suggest_categorical("expA_leaves", [15, 31, 63]),
        "lgbm_l2":                trial.suggest_float("expA_l2", 0.0, 2.0),
        "lgbm_min_child_samples": 20,
    }
    expB_params = {
        "lgbm_iters":             trial.suggest_int("expB_iters", 100, 500, step=50),
        "lgbm_lr":                trial.suggest_float("expB_lr", 0.01, 0.1, log=True),
        "lgbm_leaves":            trial.suggest_categorical("expB_leaves", [15, 31, 63]),
        "lgbm_l2":                trial.suggest_float("expB_l2", 0.0, 2.0),
        "lgbm_min_child_samples": 5,
    }
    gate_params = {
        "lgbm_iters":             trial.suggest_int("gate_iters", 100, 300, step=50),
        "lgbm_lr":                trial.suggest_float("gate_lr", 0.01, 0.1, log=True),
        "lgbm_leaves":            trial.suggest_categorical("gate_leaves", [15, 31]),
        "lgbm_l2":                0.0,
        "lgbm_min_child_samples": 20,
    }

    try:
        # Train LSTM
        lstm_m = train_lstm(seqs, train_h, val_h, hadm_to_los, lstm_params, max_notes=args.max_notes)

        Xtr_lstm = extract_lstm_features(seqs, tr_ids, lstm_m, max_notes=args.max_notes)
        Xva_lstm = extract_lstm_features(seqs, va_ids, lstm_m, max_notes=args.max_notes)

        # M1 struct predictions (needed for residual)
        lgbm_struct_params = {
            "lgbm_iters": 500, "lgbm_lr": 0.05, "lgbm_leaves": 63,
            "lgbm_l2": 1.0, "lgbm_min_child_samples": 20,
        }
        lgbm_s = train_lgbm(Xtr_struct, y_tr, lgbm_struct_params, X_val=Xva_struct, y_val=y_va)
        pred_s_tr = lgbm_s.predict(Xtr_struct).astype(np.float32)
        pred_s_va = lgbm_s.predict(Xva_struct).astype(np.float32)

        X_exp_tr = np.hstack([Xtr_struct, Xtr_lstm])
        X_exp_va = np.hstack([Xva_struct, Xva_lstm])
        resid_tr = y_tr - pred_s_tr

        tail_mask_tr = y_tr >= np.percentile(y_tr, tail_pct)

        # Expert A — non-tail
        expA_m = lgb.LGBMRegressor(
            n_estimators=expA_params["lgbm_iters"],
            learning_rate=expA_params["lgbm_lr"],
            num_leaves=expA_params["lgbm_leaves"],
            reg_lambda=expA_params["lgbm_l2"],
            min_child_samples=expA_params["lgbm_min_child_samples"],
            random_state=SEED, n_jobs=-1, verbose=-1,
        )
        expA_m.fit(X_exp_tr[~tail_mask_tr], resid_tr[~tail_mask_tr])

        # Expert B — tail
        expB_m = lgb.LGBMRegressor(
            n_estimators=expB_params["lgbm_iters"],
            learning_rate=expB_params["lgbm_lr"],
            num_leaves=expB_params["lgbm_leaves"],
            reg_lambda=expB_params["lgbm_l2"],
            min_child_samples=expB_params["lgbm_min_child_samples"],
            random_state=SEED, n_jobs=-1, verbose=-1,
        )
        expB_m.fit(X_exp_tr[tail_mask_tr], resid_tr[tail_mask_tr])

        # Gate
        gate_m = lgb.LGBMClassifier(
            n_estimators=gate_params["lgbm_iters"],
            learning_rate=gate_params["lgbm_lr"],
            num_leaves=gate_params["lgbm_leaves"],
            random_state=SEED, n_jobs=-1, verbose=-1,
        )
        gate_m.fit(X_exp_tr, tail_mask_tr.astype(int))

        gate_va = gate_m.predict_proba(X_exp_va)[:, 1].astype(np.float32)
        pred_m5_va = (pred_s_va
                      + (1 - gate_va) * expA_m.predict(X_exp_va).astype(np.float32)
                      + gate_va       * expB_m.predict(X_exp_va).astype(np.float32))

        return mean_absolute_error(y_va, pred_m5_va)

    except Exception as e:
        print(f"  Trial failed: {e}")
        return 999.0


# =====================================================
# Run study
# =====================================================

OBJECTIVES = {
    "m1": objective_m1,
    "m2": objective_m2,
    "m3": objective_m3,
    "m5": objective_m5,
}

print(f"\nStarting Optuna study ({args.trials} trials)...")
study = optuna.create_study(direction="minimize",
                            study_name=f"hparam_{args.model}")
study.optimize(OBJECTIVES[args.model], n_trials=args.trials, show_progress_bar=True)

best = study.best_params
print(f"\nBest val MAE: {study.best_value:.4f}")
print(f"Best params: {json.dumps(best, indent=2)}")

# For M5: collect all sub-group params into a flat dict with clear keys
# The JSON is intended to be passed directly to full_train via --params

out_path = f"{args.output}/best_params_{args.model}.json"
with open(out_path, "w") as f:
    json.dump(best, f, indent=2)
print(f"\nSaved: {out_path}")

# Print top-5 trials
print("\nTop-5 trials:")
trials_df = study.trials_dataframe(attrs=("number", "value", "params"))
print(trials_df.sort_values("value").head(5).to_string(index=False))
