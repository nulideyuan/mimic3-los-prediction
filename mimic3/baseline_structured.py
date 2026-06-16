"""
baseline_structured.py

Structured-only baseline comparison on MIMIC-III.
Models: LightGBM, CatBoost, XGBoost, MLP, Linear, Ridge
Metrics: MAE, RMSE, R², Tail MAE at P90/P95/P99
5-seed mean ± std evaluation.

Usage:
    python mimic3/baseline_structured.py
"""

import os
import json
import argparse
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--struct-path", default="mimic3_dataset/processed/mimic3_structured_24h_clean.csv")
parser.add_argument("--output", default="outputs_mimic3_baselines")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

os.makedirs(args.output, exist_ok=True)
SEED = args.seed

# ── Load and preprocess ───────────────────────────────────────────────────────

def load_structured():
    df = pd.read_csv(args.struct_path)
    df["hadm_id"]   = df["hadm_id"].astype(int)
    df["admittime"] = pd.to_datetime(df["admittime"])
    df = df[(df["los_days"] >= 0.5) & (df["los_days"] <= 365)].copy()
    feat_cols = [c for c in df.columns if c not in ("hadm_id", "admittime", "los_days")]
    for c in df[feat_cols].select_dtypes(include="object").columns:
        df[c] = LabelEncoder().fit_transform(df[c].fillna("missing").astype(str))
    df[feat_cols] = df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    return df, feat_cols

df, feat_cols = load_structured()
df = df.sort_values("admittime").reset_index(drop=True)
n = len(df)
n_tr = int(0.70 * n)
n_va = int(0.10 * n)

X_tr = df[feat_cols].values[:n_tr].astype(np.float32)
X_va = df[feat_cols].values[n_tr:n_tr+n_va].astype(np.float32)
X_te = df[feat_cols].values[n_tr+n_va:].astype(np.float32)
y_tr = df["los_days"].values[:n_tr].astype(np.float32)
y_va = df["los_days"].values[n_tr:n_tr+n_va].astype(np.float32)
y_te = df["los_days"].values[n_tr+n_va:].astype(np.float32)

thresholds = {
    "P90": float(np.percentile(y_tr, 90)),
    "P95": float(np.percentile(y_tr, 95)),
    "P99": float(np.percentile(y_tr, 99)),
}
print(f"Dataset: train={len(X_tr)}  val={len(X_va)}  test={len(X_te)}")
print("Thresholds:", {k: f"{v:.1f}d" for k, v in thresholds.items()})

# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred):
    row = {
        "MAE":  float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2":   float(r2_score(y_true, y_pred)),
    }
    for label, thresh in thresholds.items():
        mask = y_true >= thresh
        row[f"Tail_MAE_{label}"] = float(mean_absolute_error(y_true[mask], y_pred[mask]))
    return row

# ── Model builders ────────────────────────────────────────────────────────────

def build_lightgbm(seed):
    m = lgb.LGBMRegressor(
        n_estimators=500, learning_rate=0.05, num_leaves=63,
        colsample_bytree=0.8, subsample=0.8,
        random_state=seed, n_jobs=-1, verbose=-1)
    m.fit(X_tr, y_tr,
          eval_set=[(X_va, y_va)],
          callbacks=[lgb.early_stopping(50, verbose=False),
                     lgb.log_evaluation(-1)])
    return m

def build_catboost(seed):
    m = CatBoostRegressor(
        iterations=500, learning_rate=0.05, depth=6,
        random_seed=seed, verbose=0,
        early_stopping_rounds=50,
        eval_metric="MAE")
    m.fit(X_tr, y_tr, eval_set=(X_va, y_va))
    return m

def build_xgboost(seed):
    m = xgb.XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        colsample_bytree=0.8, subsample=0.8,
        random_state=seed, n_jobs=-1, verbosity=0,
        early_stopping_rounds=50, eval_metric="mae")
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    return m

def build_mlp(seed):
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    Xtr_s = x_scaler.fit_transform(X_tr)
    Xte_s = x_scaler.transform(X_te)
    ytr_s = y_scaler.fit_transform(y_tr.reshape(-1, 1)).ravel()
    m = MLPRegressor(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu", max_iter=500,
        early_stopping=True, validation_fraction=0.1,
        random_state=seed, verbose=False)
    m.fit(Xtr_s, ytr_s)
    pred = y_scaler.inverse_transform(m.predict(Xte_s).reshape(-1, 1)).ravel()
    pred = np.clip(pred, 0.5, 365)
    return pred

def build_linear(seed):
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    Xtr_s = x_scaler.fit_transform(X_tr)
    Xte_s = x_scaler.transform(X_te)
    ytr_s = y_scaler.fit_transform(y_tr.reshape(-1, 1)).ravel()
    m = Ridge(alpha=1.0)
    m.fit(Xtr_s, ytr_s)
    pred = y_scaler.inverse_transform(m.predict(Xte_s).reshape(-1, 1)).ravel()
    pred = np.clip(pred, 0.5, 365)
    return pred


# ── Run experiments ───────────────────────────────────────────────────────────

np.random.seed(SEED)
rows = []
metric_cols = ["MAE", "RMSE", "R2", "Tail_MAE_P90", "Tail_MAE_P95", "Tail_MAE_P99"]

print(f"\nSeed={SEED}")

m = build_lightgbm(SEED)
r = compute_metrics(y_te, m.predict(X_te))
rows.append({"model": "LightGBM", **r})
print(f"  LightGBM  MAE={r['MAE']:.4f}")

m = build_catboost(SEED)
r = compute_metrics(y_te, m.predict(X_te))
rows.append({"model": "CatBoost", **r})
print(f"  CatBoost  MAE={r['MAE']:.4f}")

m = build_xgboost(SEED)
r = compute_metrics(y_te, m.predict(X_te))
rows.append({"model": "XGBoost", **r})
print(f"  XGBoost   MAE={r['MAE']:.4f}")

pred_mlp = build_mlp(SEED)
r = compute_metrics(y_te, pred_mlp)
rows.append({"model": "MLP", **r})
print(f"  MLP       MAE={r['MAE']:.4f}")

pred_lin = build_linear(SEED)
r = compute_metrics(y_te, pred_lin)
rows.append({"model": "Linear (Ridge)", **r})
print(f"  Linear    MAE={r['MAE']:.4f}")

# ── Save ──────────────────────────────────────────────────────────────────────

df_out = pd.DataFrame(rows)
df_out.to_csv(f"{args.output}/baseline_results.csv", index=False)

print("\n" + "="*80)
print("MIMIC-III Structured Baseline Comparison")
print("="*80)
print(df_out[["model"] + metric_cols].to_string(index=False, float_format="%.4f"))
print(f"\nSaved: {args.output}/baseline_results.csv")
