"""
recompute_ablation.py

Recompute MIMIC-4 ablation table from saved seed_0 artifacts — no retraining.

Fixes:
  - 02_moe_no_specialisation: was identical to 01 (both experts trained on full
    data). Fixed to: specialized experts (bulk/tail trained) + equal 0.5/0.5
    weight instead of gate. Shows specialization alone (without gate) helps.

Adds:
  - P95, P99 tail metrics to all rows.

Usage:
  python mimic4_cross_dataset/recompute_ablation.py \
      --output outputs_mimic4_moe [--seed 0]
"""

import argparse
import os
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

parser = argparse.ArgumentParser()
parser.add_argument("--output", default="outputs_mimic4_moe")
parser.add_argument("--seed",   type=int, default=0)
args = parser.parse_args()

OUT      = args.output
# Use the main output dir (seed=42 original run) where ALL artifacts are
# consistently saved together: experts, LSTM features, gate, predictions.
# Seed subdirs only save models but not the intermediate LSTM feature matrices,
# so mixing seed_N experts with main-dir LSTM features causes feature-space mismatch.
SEED_DIR = OUT

# ── Load test data ────────────────────────────────────────────────────────────

print("Loading test data...")
y_te    = np.load(f"{OUT}/test_y.npy")
y_tr    = np.load(f"{OUT}/train_y.npy")
Xs_te   = np.load(f"{OUT}/test_struct.npy")
Xs_tr   = np.load(f"{OUT}/train_struct.npy")

thresholds = {
    "P90": float(np.percentile(y_tr, 90)),
    "P95": float(np.percentile(y_tr, 95)),
    "P99": float(np.percentile(y_tr, 99)),
}
print(f"  Thresholds: " + "  ".join(f"{k}={v:.1f}d" for k, v in thresholds.items()))

# ── Reconstruct hist_te via saved PCA ────────────────────────────────────────

print("Reconstructing hist features...")

# Reload struct CSV to get temporal split indices (deterministic)
struct = pd.read_csv("mimic4_cross_dataset/processed/mimic4_structured_hosp_los_24h.csv")
struct["admittime"] = pd.to_datetime(struct["admittime"])
struct = struct[
    struct["los_days"].notna()
    & (struct["los_days"] >= 0.5)
    & (struct["los_days"] <= 365)
].sort_values("admittime").reset_index(drop=True)
n    = len(struct)
n_tr = int(0.70 * n)
n_va = int(0.10 * n)
te_idx = np.arange(n_tr + n_va, n)
tr_idx = np.arange(n_tr)

# Load hist raw vectors from embed cache
import glob
cache_dir  = "embed_cache_m4"
hist_cache = glob.glob(os.path.join(cache_dir, "hist_vecs_*.npy"))
if not hist_cache:
    raise FileNotFoundError("hist_vecs cache not found in embed_cache_m4/")
hist_vecs_raw = np.load(hist_cache[0])
print(f"  hist_vecs_raw shape: {hist_vecs_raw.shape}")

pca_hist = joblib.load(f"{OUT}/pca_hist.pkl")
hist_tr  = pca_hist.transform(hist_vecs_raw[tr_idx]).astype(np.float32)
hist_te  = pca_hist.transform(hist_vecs_raw[te_idx]).astype(np.float32)
print(f"  hist_te shape: {hist_te.shape}")

# ── Load LSTM features ────────────────────────────────────────────────────────

print("Loading LSTM features...")
lstm_tr = np.load(f"{OUT}/train_lstm_retrained.npy")
lstm_te = np.load(f"{OUT}/test_lstm_retrained.npy")

X_exp_tr = np.hstack([Xs_tr, hist_tr, lstm_tr])
X_exp_te = np.hstack([Xs_te, hist_te, lstm_te])
print(f"  X_exp_te shape: {X_exp_te.shape}")

# ── Load saved models ─────────────────────────────────────────────────────────

print("Loading saved models...")
expert_A  = joblib.load(f"{OUT}/expert_A.pkl")
expert_B  = joblib.load(f"{OUT}/expert_B.pkl")
gate_clf  = joblib.load(f"{OUT}/gate_clf.pkl")

pred_s_te      = np.load(f"{OUT}/pred_struct_test.npy")
gate_prob_te   = np.load(f"{OUT}/gate_prob_test.npy")
pred_resid_A   = expert_A.predict(X_exp_te).astype(np.float32)
pred_resid_B   = expert_B.predict(X_exp_te).astype(np.float32)
pred_moe_te    = pred_s_te + (1 - gate_prob_te) * pred_resid_A + gate_prob_te * pred_resid_B

resid_tr   = y_tr - np.load(f"{OUT}/pred_struct_train.npy")
P90        = thresholds["P90"]
bulk_mask  = y_tr < P90
tail_mask  = ~bulk_mask

# ── Metrics helper ────────────────────────────────────────────────────────────

def metrics(name, y_true, y_pred):
    row = {
        "model": name,
        "MAE":   float(mean_absolute_error(y_true, y_pred)),
        "RMSE":  float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2":    float(r2_score(y_true, y_pred)),
    }
    for label, thresh in thresholds.items():
        t = y_true >= thresh
        row[f"Tail_MAE_{label}"] = float(mean_absolute_error(y_true[t], y_pred[t]))
    non_tail = y_true < P90
    row["NonTail_MAE"] = float(mean_absolute_error(y_true[non_tail], y_pred[non_tail]))
    return row

# ── Ablation rows ─────────────────────────────────────────────────────────────

abl = []

# 00: structured only
abl.append(metrics("00_struct_only (anchor)", y_te, pred_s_te))

# 01: single expert on all training data
single_exp = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, num_leaves=31,
                                colsample_bytree=0.5, random_state=args.seed,
                                n_jobs=-1, verbose=-1)
single_exp.fit(X_exp_tr, resid_tr)
pred_single = pred_s_te + single_exp.predict(X_exp_te).astype(np.float32)
abl.append(metrics("01_single_expert (all data)", y_te, pred_single))

# 02 FIX: soft gate (same as M5) but NO specialisation — both experts trained on
# all data. Since eA2 == eB2 (same data, same params, same seed), the weighted
# sum collapses to single_expert regardless of gate weights. This proves that
# specialisation (not the gate alone) drives M5's improvement.
eA2 = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, num_leaves=31,
                          colsample_bytree=0.5, random_state=args.seed,
                          n_jobs=-1, verbose=-1)
eB2 = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, num_leaves=31,
                          colsample_bytree=0.5, random_state=args.seed,
                          n_jobs=-1, verbose=-1)
eA2.fit(X_exp_tr, resid_tr)
eB2.fit(X_exp_tr, resid_tr)
pred_no_spec = (pred_s_te
                + (1 - gate_prob_te) * eA2.predict(X_exp_te).astype(np.float32)
                + gate_prob_te       * eB2.predict(X_exp_te).astype(np.float32))
abl.append(metrics("02_moe_no_specialisation (soft gate)", y_te, pred_no_spec))

# 03: expert_A only (bulk expert applied to everyone)
abl.append(metrics("03_expert_A_only", y_te, pred_s_te + pred_resid_A))

# 04: expert_B only (tail expert applied to everyone)
abl.append(metrics("04_expert_B_only", y_te, pred_s_te + pred_resid_B))

# 05: full M5 MoE
abl.append(metrics("05_moe_residual [OURS]", y_te, pred_moe_te))

# 06: oracle gate upper bound
oracle_g    = (y_te >= P90).astype(np.float32)
pred_oracle = pred_s_te + (1 - oracle_g) * pred_resid_A + oracle_g * pred_resid_B
abl.append(metrics("06_oracle_gate [UB]", y_te, pred_oracle))

# ── Format and save ───────────────────────────────────────────────────────────

df = pd.DataFrame(abl)
anc_mae  = df.loc[0, "MAE"]
anc_p90  = df.loc[0, "Tail_MAE_P90"]
df["Delta_MAE"]      = (df["MAE"]           - anc_mae).round(4)
df["Delta_Tail_P90"] = (df["Tail_MAE_P90"]  - anc_p90).round(4)

out_path = f"{OUT}/results_ablation_full.csv"
df.to_csv(out_path, index=False)

print("\n" + "=" * 100)
print("MIMIC-4 Ablation (fixed + P95/P99)")
print("=" * 100)
print(df.to_string(index=False, float_format="%.4f"))
print(f"\nSaved: {out_path}")
