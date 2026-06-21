"""
baseline_text.py

Compute the three additional baselines requested for the paper:
  1. ClinicalBERT only      — section-aware LSTM, no structured features
  2. Early Fusion (mean)    — struct + mean ClinicalBERT note (M2)
  3. Early Fusion (last)    — struct + last ClinicalBERT note (M3)
  4. Tabular + BERT no MoE  — struct + section LSTM, no residual MoE (M4)

All predictions are loaded from saved .npy files (no retraining needed).

Usage (run from NLP/ directory):
    python mimic3_dataset/mimic3/baseline_text.py \
        --output outputs_mimic3_lstm_full_gated_residual
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

parser = argparse.ArgumentParser()
parser.add_argument("--output", default="outputs_mimic3_lstm_full_gated_residual")
args = parser.parse_args()
OUT = args.output

# ── Load ground truth ─────────────────────────────────────────────────────────

y_te = np.load(f"{OUT}/test_y.npy")
y_tr = np.load(f"{OUT}/train_y.npy")

thresholds = {
    "P90": float(np.percentile(y_tr, 90)),
    "P95": float(np.percentile(y_tr, 95)),
    "P99": float(np.percentile(y_tr, 99)),
}
print(f"Test patients: {len(y_te)}")
print("Thresholds:", {k: f"{v:.1f}d" for k, v in thresholds.items()})

# ── Metrics helper ────────────────────────────────────────────────────────────

def compute_metrics(name, pred):
    row = {
        "model": name,
        "MAE":   float(mean_absolute_error(y_te, pred)),
        "RMSE":  float(np.sqrt(mean_squared_error(y_te, pred))),
        "R2":    float(r2_score(y_te, pred)),
    }
    for label, thresh in thresholds.items():
        mask = y_te >= thresh
        row[f"Tail_MAE_{label}"] = float(mean_absolute_error(y_te[mask], pred[mask]))
    return row

# ── Load predictions and compute ──────────────────────────────────────────────

models = [
    ("ClinicalBERT only (section LSTM)",  "pred_lstm_only.npy"),
    ("Early Fusion (struct + mean note)", "pred_struct_plus_mean_note.npy"),
    ("Early Fusion (struct + last note)", "pred_struct_plus_last_note.npy"),
    ("Tabular + BERT no MoE (struct + LSTM)", "pred_struct_plus_lstm_ungated.npy"),
]

rows = []
for name, fname in models:
    pred = np.load(f"{OUT}/{fname}")
    r = compute_metrics(name, pred)
    rows.append(r)
    print(f"\n{name}")
    for k, v in r.items():
        if k != "model":
            print(f"  {k:20s}: {v:.4f}")

# ── Save ──────────────────────────────────────────────────────────────────────

df = pd.DataFrame(rows)
out_path = f"{OUT}/baseline_text_results.csv"
df.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
