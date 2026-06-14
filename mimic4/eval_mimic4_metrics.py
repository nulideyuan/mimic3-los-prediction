"""
Print per-percentile tail metrics for all saved MIMIC-4 predictions.
Usage: python mimic4_cross_dataset/eval_mimic4_metrics.py --output outputs_mimic4_moe
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

parser = argparse.ArgumentParser()
parser.add_argument("--output", default="outputs_mimic4_moe")
args = parser.parse_args()
OUT = args.output

y_te    = np.load(f"{OUT}/test_y.npy")
y_tr    = np.load(f"{OUT}/train_y.npy")

thresholds = {
    "P75": float(np.percentile(y_tr, 75)),
    "P90": float(np.percentile(y_tr, 90)),
    "P95": float(np.percentile(y_tr, 95)),
}
print(f"Thresholds: " + "  ".join(f"{k}={v:.1f}d" for k, v in thresholds.items()))

PRED_FILES = {
    "structured_only":          f"{OUT}/pred_struct_test.npy",
    "struct+history":           f"{OUT}/pred_hist_test.npy",
    "struct+hist+lstm_direct":  f"{OUT}/pred_all_direct_test.npy",
    "struct+rad_frozen_m3":     f"{OUT}/pred_rad_frozen_direct_test.npy",
    "struct+rad_retrained":     f"{OUT}/pred_rad_retrained_direct_test.npy",
    "moe_frozen_m3_lstm":       f"{OUT}/pred_moe_frozen_test.npy",
    "moe_residual [OURS]":      f"{OUT}/pred_moe_test.npy",
}

def compute(name, y_true, y_pred, thresholds):
    row = {
        "model": name,
        "MAE":   float(mean_absolute_error(y_true, y_pred)),
        "RMSE":  float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2":    float(r2_score(y_true, y_pred)),
    }
    for label, thresh in thresholds.items():
        tail = y_true >= thresh
        row[f"Tail_MAE_{label}"] = float(mean_absolute_error(y_true[tail], y_pred[tail]))
    non_tail = y_true < thresholds["P90"]
    row["NonTail_MAE"] = float(mean_absolute_error(y_true[non_tail], y_pred[non_tail]))
    return row

rows = []
for name, path in PRED_FILES.items():
    try:
        pred = np.load(path).astype(np.float32)
        rows.append(compute(name, y_te, pred, thresholds))
    except FileNotFoundError:
        print(f"  [skip] {name}: {path} not found")

df = pd.DataFrame(rows)
base_mae = df.loc[df["model"] == "structured_only", "MAE"].iloc[0]
df["Delta_MAE"] = (df["MAE"] - base_mae).round(4)

print("\n" + df.to_string(index=False, float_format="%.4f"))
df.to_csv(f"{OUT}/results_main_percentile.csv", index=False)
print(f"\nSaved to {OUT}/results_main_percentile.csv")
