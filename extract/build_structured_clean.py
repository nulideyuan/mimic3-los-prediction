"""
build_structured_clean.py

Cleans the output of build_mimic3_los_bigquery.py into a model-ready file.

Source : processed/mimic3_los.csv  (output of BigQuery script, 24h window)

Output : processed/mimic3_structured_24h_clean.csv

Columns (~27, no NaN after median imputation):
  Demographics : anchor_age, gender_bin, is_emergency, is_weekend,
                 admission_dayofweek, admission_month,
                 insurance_cat, admission_location_cat
  Acuity       : icu_within_24h
  History      : prior_admissions, rx_count_24h
  Labs (first) : creatinine, wbc, hemoglobin, sodium, potassium,
                 bicarbonate, bun, glucose, platelets, inr
  Vitals (mean): heart_rate, sbp, spo2, resp_rate
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path("processed")

# ── Load ──────────────────────────────────────────────────────────────────────

df = pd.read_csv(DATA_DIR / "mimic3_los.csv")
df["hadm_id"] = df["hadm_id"].astype(int)
print(f"mimic3_los.csv: {df.shape}")

# ── Select columns ────────────────────────────────────────────────────────────

KEEP = (
    ["hadm_id", "admittime", "los_days", "hospital_expire_flag"]
    + ["anchor_age", "gender_bin", "is_emergency", "is_weekend",
       "admission_dayofweek", "admission_month",
       "insurance_cat", "admission_location_cat"]
    + ["icu_within_24h", "prior_admissions", "rx_count_24h"]
    + [f"lab_{n}" for n in [
        "creatinine", "wbc", "hemoglobin", "sodium", "potassium",
        "bicarbonate", "bun", "glucose", "platelets", "inr",
    ]]
    + [f"vital_{n}" for n in ["heart_rate", "sbp", "spo2", "resp_rate"]]
)

available = [c for c in KEEP if c in df.columns]
missing   = [c for c in KEEP if c not in df.columns]
if missing:
    print(f"[warn] columns not found in mimic3_los.csv: {missing}")

df = df[available].copy()

# ── Null summary ──────────────────────────────────────────────────────────────

null_rates = df.isna().mean().sort_values(ascending=False)
print("\nNull rates before imputation:")
print(null_rates[null_rates > 0].to_string())

# ── Median imputation ─────────────────────────────────────────────────────────

feat_cols = [c for c in df.columns
             if c not in ("hadm_id", "admittime", "los_days", "hospital_expire_flag")]

for c in feat_cols:
    if df[c].isna().any():
        df[c] = df[c].fillna(df[c].median())

print(f"\nAfter imputation — any NaN: {df[feat_cols].isna().any().any()}")
print(f"Final shape: {df.shape}")
print(f"\nFeature columns ({len(feat_cols)}):")
for c in feat_cols:
    print(f"  {c}")

# ── Save ──────────────────────────────────────────────────────────────────────

out = DATA_DIR / "mimic3_structured_24h_clean.csv"
df.to_csv(out, index=False)
print(f"\nSaved: {out}")
