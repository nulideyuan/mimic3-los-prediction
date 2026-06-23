"""
predict_m5.py

Run M5 (MoE residual) inference on new patients using pretrained models.
No retraining needed — loads saved .pkl and .pt files.

Pipeline:
  1. Load structured features
  2. Run ClinicalBERT + LSTM → LSTM hidden features
  3. base_pred = lgbm_structured_only.predict(struct_feats)
  4. X_exp = [struct_feats | lstm_feats]
  5. gate_prob = gate_clf.predict_proba(X_exp)[:, 1]
  6. final = base_pred + (1-gate_prob)*expert_A.predict(X_exp)
                       + gate_prob *expert_B.predict(X_exp)

Usage (run from NLP/ directory):
    python mimic3_dataset/mimic3/predict_m5.py \
        --hadm-ids 100001 100002 100003 \
        --output-dir outputs_mimic3_lstm_full_gated_residual
"""

import os
import pickle
import argparse
import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error

import sys
sys.path.insert(0, "mimic3_dataset/mimic3")
from utils import LSTMWithAttention, pad_seq, SECTIONS_TO_USE, set_seed

# ── Args ─────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--hadm-ids", nargs="*", type=int, default=None,
                    help="Specific hadm_ids to predict. If omitted, uses saved test split.")
parser.add_argument("--model-dir", default="mimic3_dataset/mimic3/models")
parser.add_argument("--seqs-cache", default="outputs_mimic3_lstm_full_gated_residual/seqs.pkl",
                    help="path to seqs.pkl (ClinicalBERT section embeddings cache)")
parser.add_argument("--max-notes", type=int, default=8)
args = parser.parse_args()

OUT = args.model_dir
set_seed(42)

device = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)
print(f"Device: {device}")

# ── Load pretrained models ────────────────────────────────────────────────────

print("Loading pretrained models...")
lgbm_struct  = joblib.load(f"{OUT}/lgbm_structured_only.pkl")
expert_A     = joblib.load(f"{OUT}/expert_A.pkl")
expert_B     = joblib.load(f"{OUT}/expert_B.pkl")
gate_clf     = joblib.load(f"{OUT}/gate_clf.pkl")

# Load LSTM architecture params
lstm_hidden_dim  = 64
lstm_sec_emb_dim = 32

lstm_model = LSTMWithAttention(
    input_dim=768,
    n_sections=len(SECTIONS_TO_USE),
    sec_emb_dim=lstm_sec_emb_dim,
    hidden_dim=lstm_hidden_dim,
    dropout=0.0,
).to(device)
lstm_model.load_state_dict(torch.load(f"{OUT}/lstm_model.pt", map_location=device))
lstm_model.eval()
print("  Models loaded.")

# ── Load structured data ──────────────────────────────────────────────────────

print("Loading structured data...")
df_struct = pd.read_csv("mimic3_dataset/processed/mimic3_structured_24h_clean.csv")
df_struct["hadm_id"]   = df_struct["hadm_id"].astype(int)
df_struct["admittime"] = pd.to_datetime(df_struct["admittime"])
df_struct = df_struct[(df_struct["los_days"] >= 0.5) & (df_struct["los_days"] <= 365)].copy()

feat_cols = [c for c in df_struct.columns if c not in ("hadm_id", "admittime", "los_days")]
for c in df_struct[feat_cols].select_dtypes(include="object").columns:
    df_struct[c] = LabelEncoder().fit_transform(df_struct[c].fillna("missing").astype(str))
df_struct[feat_cols] = df_struct[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

hadm_to_feat = df_struct.set_index("hadm_id")[feat_cols].to_dict("index")
hadm_to_los  = dict(zip(df_struct["hadm_id"], df_struct["los_days"]))

# ── Determine which hadm_ids to predict ──────────────────────────────────────

if args.hadm_ids:
    hadm_ids = [h for h in args.hadm_ids if h in hadm_to_feat]
    missing  = [h for h in args.hadm_ids if h not in hadm_to_feat]
    if missing:
        print(f"  Warning: {len(missing)} hadm_ids not found in structured data: {missing}")
else:
    # Use saved test split (same temporal split as training)
    df_sorted = df_struct.sort_values("admittime").reset_index(drop=True)
    n = len(df_sorted)
    n_tr = int(0.70 * n)
    n_va = int(0.10 * n)
    hadm_ids = df_sorted["hadm_id"].values[n_tr + n_va:].tolist()
    print(f"  Using saved test split: {len(hadm_ids)} admissions")

# ── Build structured feature matrix ──────────────────────────────────────────

X_struct = np.array(
    [[hadm_to_feat[h][c] for c in feat_cols] for h in hadm_ids],
    dtype=np.float32
)

# ── Load seqs and extract LSTM features ──────────────────────────────────────

print("Loading section embeddings (seqs)...")
seqs_path = args.seqs_cache
with open(seqs_path, "rb") as f:
    seqs = pickle.load(f)

@torch.no_grad()
def extract_lstm_features(hadm_ids, batch_size=512):
    n_sec = len(SECTIONS_TO_USE)
    hidden_size = lstm_model.lstm.hidden_size
    feats = np.zeros((len(hadm_ids), n_sec * hidden_size), dtype=np.float32)

    items = []
    for i, hid in enumerate(hadm_ids):
        for j, sec in enumerate(SECTIONS_TO_USE):
            arr = seqs[sec].get(int(hid))
            if arr is not None:
                items.append((i, j, pad_seq(arr, args.max_notes)))

    for start in range(0, len(items), batch_size):
        batch  = items[start:start + batch_size]
        row_ids = [b[0] for b in batch]
        sec_ids = [b[1] for b in batch]
        xb = torch.tensor(np.stack([b[2] for b in batch]), dtype=torch.float32).to(device)
        sb = torch.tensor(sec_ids, dtype=torch.long).to(device)
        enc = lstm_model.encode(xb, sb).cpu().numpy()
        for k, (row_i, sec_j) in enumerate(zip(row_ids, sec_ids)):
            feats[row_i, sec_j * hidden_size:(sec_j + 1) * hidden_size] = enc[k]

    return feats

print("Extracting LSTM features...")
X_lstm = extract_lstm_features(hadm_ids)
print(f"  LSTM features shape: {X_lstm.shape}")

# ── M5 inference ──────────────────────────────────────────────────────────────

print("Running M5 inference...")
base_pred  = lgbm_struct.predict(X_struct).astype(np.float32)
X_exp      = np.hstack([X_struct, X_lstm])
gate_prob  = gate_clf.predict_proba(X_exp)[:, 1].astype(np.float32)
resid_pred = ((1 - gate_prob) * expert_A.predict(X_exp).astype(np.float32)
              + gate_prob      * expert_B.predict(X_exp).astype(np.float32))
final_pred = base_pred + resid_pred

# ── Output ────────────────────────────────────────────────────────────────────

y_true = np.array([hadm_to_los.get(h, np.nan) for h in hadm_ids], dtype=np.float32)

df_out = pd.DataFrame({
    "hadm_id":     hadm_ids,
    "true_los":    y_true,
    "base_pred":   base_pred,
    "gate_prob":   gate_prob,
    "resid_pred":  resid_pred,
    "final_pred":  final_pred,
})
out_path = "m5_predictions.csv"
df_out.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")

valid = ~np.isnan(y_true)
if valid.sum() > 0:
    mae = mean_absolute_error(y_true[valid], final_pred[valid])
    print(f"MAE on {valid.sum()} patients with known LOS: {mae:.4f}")

print("\nSample predictions:")
print(df_out.head(10).to_string(index=False, float_format="%.3f"))
