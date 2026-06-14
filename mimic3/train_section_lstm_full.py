"""
train_section_lstm_full_gated_residual.py

Full MIMIC-3 LOS prediction with:
  - Temporal split 70/10/20 by admittime
  - 10-section ClinicalBERT section embeddings
  - Section-aware LSTMWithAttention
  - LightGBM fusion
  - Static note baselines: last-note and mean-note
  - Text-only baselines
  - Soft gate based on structured prediction
  - Residual correction: text model predicts structured baseline error
  - Optional Optuna hyperparameter search
  - Optional 5-seed robustness evaluation

Important:
  This is still section-level LSTM, not patient-level timeline LSTM.

Usage:
  # Step 1: precompute embeddings
  python los_section_lstm/train_section_lstm_full_gated_residual.py --mode embed

  # Step 2: hyperparameter search
  python los_section_lstm/train_section_lstm_full_gated_residual.py \
    --mode hparam \
    --sample 3000 \
    --trials 40

  # Step 3: full training
  python los_section_lstm/train_section_lstm_full_gated_residual.py \
    --mode train \
    --params outputs_mimic3_lstm_full_gated_residual/best_params.json

  # Step 4: 5-seed robustness
  python los_section_lstm/train_section_lstm_full_gated_residual.py \
    --mode seeds \
    --params outputs_mimic3_lstm_full_gated_residual/best_params.json \
    --seeds 0,1,2,3,4
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

from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

from utils import (
    LSTMWithAttention, pad_seq, set_seed,
    NURSING_SECTIONS, RADIOLOGY_SECTIONS, NURSING_OTHER_SECTIONS, SECTIONS_TO_USE,
)

warnings.filterwarnings("ignore")


# =====================================================
# Args
# =====================================================

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["embed", "train", "eval", "seeds"], default="train")
parser.add_argument("--model", choices=["m1", "m2", "m3", "m5", "all"], default="all",
                    help="Which model(s) to train: m1=struct only, m2=struct+mean_note, "
                         "m3=struct+last_note, m5=MoE residual [OURS], all=all four")
parser.add_argument("--output", type=str, default="outputs_mimic3_lstm_full_gated_residual")
parser.add_argument("--params", type=str, default=None, help="path to best_params.json")
parser.add_argument("--seeds", type=str, default="0,1,2,3,4", help="comma-separated seeds for --mode seeds")
parser.add_argument("--max-notes", type=int, default=8, help="max notes per section sequence")
parser.add_argument("--seqs-cache", type=str, default="processed/section_trajectory_long/seqs_clinicalbert_24h.pkl")
parser.add_argument("--rebuild-seqs", action="store_true", help="force rebuild ClinicalBERT section embeddings")
args = parser.parse_args()

OUTPUT_DIR = args.output
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =====================================================
# Paths / constants
# =====================================================

STRUCT_PATH = "processed/mimic3_structured_24h_clean.csv"
NURSING_PARQUET = "processed/section_trajectory_long/nursing_sections_24h_long.parquet"
NURSING_OTHER_PARQUET = "processed/section_trajectory_long/nursing_other_sections_24h_long.parquet"
RADIOLOGY_PARQUET = "processed/section_trajectory_long/radiology_24h_long.parquet"

SEED = 42
GATE_PERCENTILE = 90

set_seed(SEED)

device = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)
print(f"Device: {device}")


def sigmoid_gate(preds, threshold, temp):
    return (1.0 / (1.0 + np.exp(-(preds - threshold) / temp))).astype(np.float32)


# =====================================================
# Model
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


# =====================================================
# Structured data
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


# =====================================================
# Temporal split
# =====================================================

def temporal_split(df, train_frac=0.70, val_frac=0.10):
    df_s = df.sort_values("admittime").reset_index(drop=True)
    n = len(df_s)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_hadm = set(df_s.iloc[:n_train]["hadm_id"])
    val_hadm = set(df_s.iloc[n_train:n_train + n_val]["hadm_id"])
    test_hadm = set(df_s.iloc[n_train + n_val:]["hadm_id"])

    print(f"  Temporal split — train:{len(train_hadm)}  val:{len(val_hadm)}  test:{len(test_hadm)}")
    yr = df_s["admittime"].dt.year
    print(f"  Train years: {yr.iloc[:n_train].min()} – {yr.iloc[:n_train].max()}")
    print(f"  Val years:   {yr.iloc[n_train:n_train+n_val].min()} – {yr.iloc[n_train:n_train+n_val].max()}")
    print(f"  Test years:  {yr.iloc[n_train+n_val:].min()} – {yr.iloc[n_train+n_val:].max()}")
    return train_hadm, val_hadm, test_hadm


# =====================================================
# ClinicalBERT embedding
# =====================================================

def load_bert():
    from transformers import AutoTokenizer, AutoModel
    print("Loading ClinicalBERT...")
    tok = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    mdl = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(device)
    mdl.eval()
    return tok, mdl


@torch.no_grad()
def encode_texts(texts, tokenizer, bert, batch_size=128):
    vecs = []
    for i in range(0, len(texts), batch_size):
        tok_out = tokenizer(
            texts[i:i + batch_size],
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(device)
        vecs.append(bert(**tok_out).last_hidden_state[:, 0, :].cpu().numpy())
        if i % 10000 == 0 and i > 0:
            print(f"    Encoded {i}/{len(texts)}")
    return np.vstack(vecs)


# =====================================================
# Build section sequences
# =====================================================

def build_seqs(hadm_set, tokenizer, bert, max_notes=8):
    seqs = {s: {} for s in SECTIONS_TO_USE}

    def process_parquet(path, allowed_sections, source_name):
        print(f"  Loading {source_name}: {path}")
        df_p = pd.read_parquet(path)
        df_p.columns = df_p.columns.str.lower()

        df_p["hadm_id"] = df_p["hadm_id"].astype(int)
        df_p["section_name"] = df_p["section_name"].astype(str).str.strip()
        df_p["text"] = df_p["text"].fillna("").astype(str)

        df_p = df_p[
            df_p["hadm_id"].isin(hadm_set)
            & df_p["section_name"].isin(allowed_sections)
            & (df_p["text"].str.strip().str.len() > 5)
        ].copy()

        if "hours_from_admit" in df_p.columns:
            df_p = df_p.sort_values(["hadm_id", "section_name", "hours_from_admit"])
        else:
            df_p = df_p.sort_values(["hadm_id", "section_name"])

        df_p = df_p.groupby(["hadm_id", "section_name"], group_keys=False).head(max_notes)
        df_p["text_short"] = df_p["text"].str.slice(0, 512)

        unique_texts = df_p["text_short"].drop_duplicates().tolist()
        print(f"    Rows after filter: {len(df_p)}")
        print(f"    Unique texts to encode: {len(unique_texts)}")

        if len(unique_texts) == 0:
            return

        vecs = encode_texts(unique_texts, tokenizer, bert, batch_size=128)
        text_to_vec = {t: v.astype(np.float32) for t, v in zip(unique_texts, vecs)}

        for row in df_p.itertuples(index=False):
            hid = int(row.hadm_id)
            sec = row.section_name
            vec = text_to_vec[row.text_short]
            seqs[sec].setdefault(hid, []).append(vec)

    process_parquet(NURSING_PARQUET, NURSING_SECTIONS, "nursing_sections_24h_long")
    process_parquet(RADIOLOGY_PARQUET, RADIOLOGY_SECTIONS, "radiology_24h_long")
    process_parquet(NURSING_OTHER_PARQUET, NURSING_OTHER_SECTIONS, "nursing_other_sections_24h_long")

    for sec in SECTIONS_TO_USE:
        for hid in list(seqs[sec].keys()):
            seqs[sec][hid] = np.stack(seqs[sec][hid]).astype(np.float32)

    non_empty = sum(len(seqs[sec]) for sec in SECTIONS_TO_USE)
    print(f"  Non-empty patient-section sequences: {non_empty}")
    return seqs


def build_or_load_seqs(hadm_set, max_notes=8):
    cache_path = args.seqs_cache

    if os.path.exists(cache_path) and not args.rebuild_seqs:
        print(f"\nLoading precomputed seqs from: {cache_path}")
        with open(cache_path, "rb") as f:
            seqs = pickle.load(f)
        return seqs

    print("\nNo cached seqs found, or --rebuild-seqs was set.")
    print("Building ClinicalBERT section embeddings...")
    tokenizer, bert = load_bert()
    seqs = build_seqs(hadm_set, tokenizer, bert, max_notes=max_notes)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(seqs, f)
    print(f"Saved precomputed seqs to: {cache_path}")
    return seqs


def precompute_embeddings():
    print("\nPrecomputing ClinicalBERT section embeddings for full structured cohort...")
    df, _ = load_structured()
    all_hadm = set(df["hadm_id"].astype(int).tolist())
    print(f"  Patients in structured cohort: {len(all_hadm)}")
    build_or_load_seqs(all_hadm, max_notes=args.max_notes)
    print("\nEmbedding precomputation complete.")


# =====================================================
# LSTM train / extract
# =====================================================

def build_lstm_samples(seqs, hadm_set, hadm_to_los, max_notes=8):
    xs, secs, ys = [], [], []
    for j, sec in enumerate(SECTIONS_TO_USE):
        for hid, arr in seqs[sec].items():
            if hid not in hadm_set:
                continue
            xs.append(pad_seq(arr, max_notes))
            secs.append(j)
            ys.append(hadm_to_los[hid])
    return xs, secs, ys


def train_lstm(seqs, train_hadm, val_hadm, hadm_to_los, params, max_notes=8):
    hidden_dim = params["hidden_dim"]
    sec_emb_dim = params["sec_emb_dim"]
    dropout = params["dropout"]
    lr = params["lr"]
    epochs = params["epochs"]
    batch_size = params["batch_size"]

    xs, scs, ys = build_lstm_samples(seqs, train_hadm, hadm_to_los, max_notes)
    print(f"  LSTM training samples: {len(xs)}")

    if len(xs) == 0:
        raise ValueError("No LSTM training samples found. Check section files and section_name mapping.")

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
        train_losses = []

        for xb, sb, yb in loader:
            xb, sb, yb = xb.to(device), sb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb, sb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            train_losses.append(loss.item())

        model.eval()
        val_xs, val_scs, val_ys = build_lstm_samples(seqs, val_hadm, hadm_to_los, max_notes)

        if val_xs:
            with torch.no_grad():
                vx = torch.tensor(np.stack(val_xs), dtype=torch.float32).to(device)
                vs = torch.tensor(val_scs, dtype=torch.long).to(device)
                vy = torch.tensor(val_ys, dtype=torch.float32).to(device)
                val_loss = loss_fn(model(vx, vs), vy).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            print(f"  Epoch {epoch+1}/{epochs}: train_loss={np.mean(train_losses):.4f}  val_loss={val_loss:.4f}")
        else:
            print(f"  Epoch {epoch+1}/{epochs}: train_loss={np.mean(train_losses):.4f}  no val samples")

    if best_state:
        model.load_state_dict(best_state)

    return model


@torch.no_grad()
def extract_lstm_features(seqs, hadm_ids, model, max_notes=8, batch_size=512):
    model.eval()
    n = len(SECTIONS_TO_USE)
    hidden_size = model.lstm.hidden_size
    feats = np.zeros((len(hadm_ids), n * hidden_size), dtype=np.float32)

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
    assert mode in {"last", "mean"}, "mode must be 'last' or 'mean'"

    n_sections = len(SECTIONS_TO_USE)
    emb_dim = 768
    feats = np.zeros((len(hadm_ids), n_sections * emb_dim), dtype=np.float32)

    for i, hid in enumerate(hadm_ids):
        hid = int(hid)
        for j, sec in enumerate(SECTIONS_TO_USE):
            arr = seqs[sec].get(hid)
            if arr is None or len(arr) == 0:
                continue
            vec = arr[-1] if mode == "last" else arr.mean(axis=0)
            feats[i, j * emb_dim:(j + 1) * emb_dim] = vec.astype(np.float32)

    return feats


# =====================================================
# LightGBM / metrics
# =====================================================

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

    if X_val is not None and y_val is not None:
        m.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
    else:
        m.fit(X_tr, y_tr)

    return m


def eval_metrics(name, y_true, y_pred, tail_thresholds=None):
    """
    tail_thresholds: dict {label: threshold_days} computed from training LOS,
    e.g. {'P75': 11.9, 'P90': 21.6, 'P95': 30.9, 'P99': 55.0}
    Produces Tail_MAE_P75 / Tail_MAE_P90 / ... columns instead of the
    error-distribution percentiles (P90_err), which were confusingly named.
    """
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    d = {
        "model": name,
        "MAE":   float(mean_absolute_error(y_true, y_pred)),
        "RMSE":  float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2":    float(r2_score(y_true, y_pred)),
    }
    if tail_thresholds:
        for label, thresh in tail_thresholds.items():
            tail = y_true >= thresh
            d[f"Tail_MAE_{label}"] = float(
                mean_absolute_error(y_true[tail], y_pred[tail])
            )
        # NonTail_MAE uses P90 threshold
        if "P90" in tail_thresholds:
            non_tail = y_true < tail_thresholds["P90"]
            d["NonTail_MAE"] = float(
                mean_absolute_error(y_true[non_tail], y_pred[non_tail])
            )
    return d


# =====================================================
# Full train
# =====================================================

def full_train(params, seed=42, output_dir=None):
    global OUTPUT_DIR

    set_seed(seed)

    if output_dir is not None:
        OUTPUT_DIR = output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"FULL TRAINING | seed={seed} | output={OUTPUT_DIR}")
    print("=" * 80)

    print("\nLoading full structured data...")
    df, feat_cols = load_structured()
    print(f"  Total patients: {len(df)}")

    train_h, val_h, test_h = temporal_split(df)

    np.save(f"{OUTPUT_DIR}/train_hadm.npy", np.array(list(train_h)))
    np.save(f"{OUTPUT_DIR}/val_hadm.npy", np.array(list(val_h)))
    np.save(f"{OUTPUT_DIR}/test_hadm.npy", np.array(list(test_h)))

    all_h = train_h | val_h | test_h
    print("\nBuilding/loading section sequences...")
    seqs_all = build_or_load_seqs(all_h, max_notes=args.max_notes)
    seqs = {
        sec: {hid: arr for hid, arr in seqs_all[sec].items() if hid in all_h}
        for sec in SECTIONS_TO_USE
    }

    with open(f"{OUTPUT_DIR}/seqs.pkl", "wb") as f:
        pickle.dump(seqs, f)
    print(f"  Saved run-specific seqs copy: {OUTPUT_DIR}/seqs.pkl")

    hadm_to_los = dict(zip(df["hadm_id"], df["los_days"]))
    hadm_to_feat = df.set_index("hadm_id")[feat_cols].to_dict("index")

    def struct_array(hadm_ids):
        return np.array([[hadm_to_feat[h][c] for c in feat_cols] for h in hadm_ids], dtype=np.float32)

    tr_ids = list(train_h)
    va_ids = list(val_h)
    te_ids = list(test_h)

    y_tr = np.array([hadm_to_los[h] for h in tr_ids], dtype=np.float32)
    y_va = np.array([hadm_to_los[h] for h in va_ids], dtype=np.float32)
    y_te = np.array([hadm_to_los[h] for h in te_ids], dtype=np.float32)

    tail_thresholds = {
        "P75": float(np.percentile(y_tr, 75)),
        "P90": float(np.percentile(y_tr, 90)),
        "P95": float(np.percentile(y_tr, 95)),
        "P99": float(np.percentile(y_tr, 99)),
    }
    print(f"Tail thresholds (train LOS): " +
          ", ".join(f"{k}={v:.1f}d" for k, v in tail_thresholds.items()))

    Xtr_struct = struct_array(tr_ids)
    Xva_struct = struct_array(va_ids)
    Xte_struct = struct_array(te_ids)

    run = args.model  # "m1" | "m2" | "m3" | "m5" | "all"

    # -------------------------
    # M1 — structured only
    # Always trained: M5 needs struct predictions for residual.
    # -------------------------
    print("\nTraining M1 (structured only)...")
    lgbm_struct = train_lgbm(Xtr_struct, y_tr, params, X_val=Xva_struct, y_val=y_va)
    pred_struct_tr = lgbm_struct.predict(Xtr_struct).astype(np.float32)
    pred_struct_te = lgbm_struct.predict(Xte_struct).astype(np.float32)
    joblib.dump(lgbm_struct, f"{OUTPUT_DIR}/lgbm_structured_only.pkl")
    np.save(f"{OUTPUT_DIR}/pred_m1.npy", pred_struct_te)
    print(f"  M1 MAE: {mean_absolute_error(y_te, pred_struct_te):.4f}")

    # -------------------------
    # M2 — struct + mean note (BERT mean pooling)
    # -------------------------
    if run in ("m2", "all"):
        print("\nExtracting static note features (mean pooling)...")
        Xtr_mean = extract_static_note_features(seqs, tr_ids, mode="mean")
        Xva_mean = extract_static_note_features(seqs, va_ids, mode="mean")
        Xte_mean = extract_static_note_features(seqs, te_ids, mode="mean")
        print(f"  Mean-note features: {Xtr_mean.shape}")

        print("Training M2 (struct + mean note)...")
        lgbm_mean = train_lgbm(
            np.hstack([Xtr_struct, Xtr_mean]), y_tr, params,
            X_val=np.hstack([Xva_struct, Xva_mean]), y_val=y_va,
        )
        pred_m2 = lgbm_mean.predict(np.hstack([Xte_struct, Xte_mean])).astype(np.float32)
        joblib.dump(lgbm_mean, f"{OUTPUT_DIR}/lgbm_struct_plus_mean_note.pkl")
        np.save(f"{OUTPUT_DIR}/pred_m2.npy", pred_m2)
        print(f"  M2 MAE: {mean_absolute_error(y_te, pred_m2):.4f}")

    # -------------------------
    # M3 — struct + last note (BERT last note)
    # -------------------------
    if run in ("m3", "all"):
        print("\nExtracting static note features (last note)...")
        Xtr_last = extract_static_note_features(seqs, tr_ids, mode="last")
        Xva_last = extract_static_note_features(seqs, va_ids, mode="last")
        Xte_last = extract_static_note_features(seqs, te_ids, mode="last")
        print(f"  Last-note features: {Xtr_last.shape}")

        print("Training M3 (struct + last note)...")
        lgbm_last = train_lgbm(
            np.hstack([Xtr_struct, Xtr_last]), y_tr, params,
            X_val=np.hstack([Xva_struct, Xva_last]), y_val=y_va,
        )
        pred_m3 = lgbm_last.predict(np.hstack([Xte_struct, Xte_last])).astype(np.float32)
        joblib.dump(lgbm_last, f"{OUTPUT_DIR}/lgbm_struct_plus_last_note.pkl")
        np.save(f"{OUTPUT_DIR}/pred_m3.npy", pred_m3)
        print(f"  M3 MAE: {mean_absolute_error(y_te, pred_m3):.4f}")

    # -------------------------
    # M5 — MoE residual [OURS]
    # Needs LSTM features + M1 struct predictions.
    # -------------------------
    if run in ("m5", "all"):
        print("\nTraining LSTM encoder...")
        lstm_model = train_lstm(seqs, train_h, val_h, hadm_to_los, params, max_notes=args.max_notes)
        torch.save(lstm_model.state_dict(), f"{OUTPUT_DIR}/lstm_model.pt")

        model_cfg = {
            "hidden_dim": params["hidden_dim"],
            "sec_emb_dim": params["sec_emb_dim"],
            "n_sections": len(SECTIONS_TO_USE),
            "max_notes": args.max_notes,
        }
        with open(f"{OUTPUT_DIR}/model_config.json", "w") as f:
            json.dump(model_cfg, f, indent=2)

        print("Extracting LSTM features...")
        Xtr_lstm = extract_lstm_features(seqs, tr_ids, lstm_model, max_notes=args.max_notes)
        Xte_lstm = extract_lstm_features(seqs, te_ids, lstm_model, max_notes=args.max_notes)
        print(f"  LSTM features: {Xtr_lstm.shape}")

        np.save(f"{OUTPUT_DIR}/test_lstm.npy", Xte_lstm)
        np.save(f"{OUTPUT_DIR}/train_lstm.npy", Xtr_lstm)
        np.save(f"{OUTPUT_DIR}/pred_struct_train.npy", pred_struct_tr)

        X_exp_tr     = np.hstack([Xtr_struct, Xtr_lstm])
        X_exp_te     = np.hstack([Xte_struct, Xte_lstm])
        resid_tr_vec = y_tr - pred_struct_tr
        P90_LOS      = tail_thresholds["P90"]
        tail_mask_tr = y_tr >= P90_LOS

        print(f"  Residual train: mean={resid_tr_vec.mean():.3f}  tail ratio={tail_mask_tr.mean():.1%}")

        print("Training M5 (Expert A / Expert B / gate)...")
        expert_A = lgb.LGBMRegressor(
            n_estimators=300, learning_rate=0.03, num_leaves=31,
            colsample_bytree=0.5, random_state=seed, n_jobs=-1, verbose=-1)
        expert_A.fit(X_exp_tr[~tail_mask_tr], resid_tr_vec[~tail_mask_tr])

        expert_B = lgb.LGBMRegressor(
            n_estimators=300, learning_rate=0.03, num_leaves=31,
            colsample_bytree=0.5, random_state=seed, n_jobs=-1, verbose=-1)
        expert_B.fit(X_exp_tr[tail_mask_tr], resid_tr_vec[tail_mask_tr])

        gate_clf = lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            colsample_bytree=0.5, random_state=seed, n_jobs=-1, verbose=-1)
        gate_clf.fit(X_exp_tr, tail_mask_tr.astype(int))

        gate_prob_te = gate_clf.predict_proba(X_exp_te)[:, 1].astype(np.float32)
        pred_m5 = (pred_struct_te
                   + (1 - gate_prob_te) * expert_A.predict(X_exp_te).astype(np.float32)
                   + gate_prob_te       * expert_B.predict(X_exp_te).astype(np.float32))

        joblib.dump(expert_A, f"{OUTPUT_DIR}/expert_A.pkl")
        joblib.dump(expert_B, f"{OUTPUT_DIR}/expert_B.pkl")
        joblib.dump(gate_clf, f"{OUTPUT_DIR}/gate_clf.pkl")
        np.save(f"{OUTPUT_DIR}/pred_m5.npy", pred_m5)
        print(f"  M5 MAE: {mean_absolute_error(y_te, pred_m5):.4f}")

    # -------------------------
    # Shared test artifacts
    # -------------------------
    with open(f"{OUTPUT_DIR}/feat_cols.json", "w") as f:
        json.dump(feat_cols, f)
    np.save(f"{OUTPUT_DIR}/test_struct.npy", Xte_struct)
    np.save(f"{OUTPUT_DIR}/test_y.npy", y_te)
    np.save(f"{OUTPUT_DIR}/train_y.npy", y_tr)
    np.save(f"{OUTPUT_DIR}/train_struct.npy", Xtr_struct)

    print(f"\nAll artifacts saved to: {OUTPUT_DIR}/")


# =====================================================
# Seeds
# =====================================================

def run_seeds(params):
    base_output = OUTPUT_DIR
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip() != ""]
    print(f"\nRunning seeds: {seeds}")
    for seed in seeds:
        full_train(params=params, seed=seed, output_dir=f"{base_output}/seed_{seed}")


# =====================================================
# Eval only
# =====================================================

def eval_only():
    """
    Load saved predictions and report metrics.

    Single-seed:  reads pred_m{1,2,3,5}.npy directly from OUTPUT_DIR.
    Multi-seed:   auto-detects seed_X/ subdirs under OUTPUT_DIR,
                  reports mean ± std across seeds.
    """
    import glob as _glob

    MODELS = [
        ("M1  struct only",          "pred_m1.npy"),
        ("M2  struct+mean_note",      "pred_m2.npy"),
        ("M3  struct+last_note",      "pred_m3.npy"),
        ("M5  MoE residual [OURS]",   "pred_m5.npy"),
    ]

    # Detect seed subdirs
    seed_dirs = sorted(_glob.glob(f"{OUTPUT_DIR}/seed_*/"))

    if seed_dirs:
        print(f"\nFound {len(seed_dirs)} seed dirs: {[os.path.basename(d.rstrip('/')) for d in seed_dirs]}")
        # Use test_y and tail thresholds from first seed dir (split is deterministic)
        y_te = np.load(f"{seed_dirs[0]}/test_y.npy")
        y_tr = np.load(f"{seed_dirs[0]}/train_y.npy")
    else:
        print(f"\nNo seed_X/ dirs found — evaluating single run in {OUTPUT_DIR}/")
        seed_dirs = [OUTPUT_DIR + "/"]
        y_te = np.load(f"{OUTPUT_DIR}/test_y.npy")
        y_tr = np.load(f"{OUTPUT_DIR}/train_y.npy")

    tail_thresholds = {
        "P75": float(np.percentile(y_tr, 75)),
        "P90": float(np.percentile(y_tr, 90)),
        "P95": float(np.percentile(y_tr, 95)),
        "P99": float(np.percentile(y_tr, 99)),
    }

    rows = []
    for label, fname in MODELS:
        # collect per-seed metrics
        seed_metrics = []
        for d in seed_dirs:
            path = os.path.join(d, fname)
            if os.path.exists(path):
                seed_metrics.append(eval_metrics(label, y_te, np.load(path), tail_thresholds))

        if not seed_metrics:
            continue

        if len(seed_metrics) == 1:
            rows.append(seed_metrics[0])
        else:
            # aggregate: mean ± std across seeds
            keys = [k for k in seed_metrics[0] if k != "model"]
            agg = {"model": label}
            for k in keys:
                vals = [m[k] for m in seed_metrics]
                agg[k]             = float(np.mean(vals))
                agg[f"{k}_std"]    = float(np.std(vals))
            rows.append(agg)

    res_df = pd.DataFrame(rows)

    # Delta vs M1
    if "M1  struct only" in res_df["model"].values:
        base = res_df.loc[res_df["model"] == "M1  struct only", "MAE"].iloc[0]
        res_df["ΔMAE"] = res_df["MAE"] - base

    print("\n" + "=" * 80)
    print(f"RESULTS  ({len(seed_dirs)} seed(s))")
    print("=" * 80)
    print(res_df.to_string(index=False, float_format="%.4f"))
    res_df.to_csv(f"{OUTPUT_DIR}/results_main.csv", index=False, float_format="%.4f")
    print(f"\nSaved → {OUTPUT_DIR}/results_main.csv")




# =====================================================
# Params / main
# =====================================================

DEFAULT_PARAMS = {
    "hidden_dim": 64,
    "sec_emb_dim": 32,
    "dropout": 0.1,
    "lr": 1e-3,
    "epochs": 10,
    "batch_size": 256,
    "lgbm_iters": 500,
    "lgbm_lr": 0.05,
    "lgbm_leaves": 63,
    "lgbm_l2": 1.0,
}


def load_params():
    if args.params:
        with open(args.params) as f:
            params = json.load(f)
        print(f"Using params from {args.params}")
        return params

    # Look for any best_params_*.json in output dir
    for model_tag in ("m5", "m1", "m2", "m3"):
        candidate = f"{OUTPUT_DIR}/best_params_{model_tag}.json"
        if os.path.exists(candidate):
            with open(candidate) as f:
                params = json.load(f)
            print(f"Using {candidate}")
            return params

    print("Using default params. Run hparam_search.py first for tuned params.")
    return DEFAULT_PARAMS.copy()


if args.mode == "embed":
    precompute_embeddings()

elif args.mode == "train":
    params = load_params()
    full_train(params=params, seed=SEED, output_dir=OUTPUT_DIR)

elif args.mode == "seeds":
    params = load_params()
    run_seeds(params)

elif args.mode == "eval":
    eval_only()
