"""
ablation_moe.py  —  multi-seed, clean tables

TABLE 1  Main results (building-block progression)
  M1   struct_only
  M2   struct + LSTM (ungated)           add text, no selective routing
  M3   struct + LSTM (gated)             gate blends M1 and M2 by tail probability
  M4   struct + LSTM + residual (1 exp)  gate-weighted single-expert correction
  M5   MoE residual [OURS]               two experts split by bulk / tail

TABLE 2  Gate ablation (G0 Expert_A_only | G1 Soft | G2 Hard | G3 LightGBM [OURS] | G4 Oracle UB)

Multi-seed usage (LSTM predictions fixed, sklearn reseeded):
  python mimic3_dataset/ablation_moe.py \\
      --output outputs_mimic3_lstm_full_gated_residual \\
      --seeds 0 1 2 3 4

For fully independent runs, train the LSTM with each seed first, save each
run to a separate output dir, then pass --output_dirs dir0 dir1 dir2 dir3 dir4.
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lightgbm import LGBMRegressor
import joblib

# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--output",      default="outputs_mimic3_lstm_full_gated_residual")
parser.add_argument("--output_dirs", nargs="+", default=None,
                    help="Per-seed output dirs (overrides --output + --seeds)")
parser.add_argument("--seeds",       nargs="+", type=int, default=[42],
                    help="sklearn seeds to sweep (LSTM predictions stay fixed)")
parser.add_argument("--n_boot",      type=int, default=2000,
                    help="Bootstrap replicates for significance test")
args = parser.parse_args()

# Resolve output dirs per seed
if args.output_dirs:
    SEED_DIRS = args.output_dirs
    SEEDS     = list(range(len(SEED_DIRS)))
else:
    SEED_DIRS = [args.output] * len(args.seeds)
    SEEDS     = args.seeds

print(f"Seeds: {SEEDS}  |  Output dirs: {SEED_DIRS[:3]}{'...' if len(SEED_DIRS)>3 else ''}")


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def eval_metrics(name, y_true, y_pred, tail_thresholds, P90_LOS):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    d = {
        "model": name,
        "MAE":   float(mean_absolute_error(y_true, y_pred)),
        "RMSE":  float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2":    float(r2_score(y_true, y_pred)),
    }
    for label, thresh in tail_thresholds.items():
        mask = y_true >= thresh
        d[f"Tail_MAE_{label}"] = float(mean_absolute_error(y_true[mask], y_pred[mask]))
    d["NonTail_MAE"] = float(mean_absolute_error(
        y_true[y_true < P90_LOS], y_pred[y_true < P90_LOS]
    ))
    return d


def bootstrap_pvalue(y, pred_a, pred_b, n_boot=2000, seed=0):
    """Two-sided bootstrap test: H0 MAE(a)==MAE(b). Returns (delta, p_value).
    delta > 0 means a is worse (b improves over a)."""
    rng   = np.random.RandomState(seed)
    n     = len(y)
    delta = mean_absolute_error(y, pred_a) - mean_absolute_error(y, pred_b)
    boot  = np.empty(n_boot)
    for i in range(n_boot):
        idx     = rng.choice(n, n, replace=True)
        boot[i] = (mean_absolute_error(y[idx], pred_a[idx])
                   - mean_absolute_error(y[idx], pred_b[idx]))
    p = float((np.abs(boot) >= np.abs(delta)).mean())
    return delta, p


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation helpers
# ─────────────────────────────────────────────────────────────────────────────

def agg_df_list(dfs, model_col="model"):
    """Mean ± std over a list of DataFrames with the same model order."""
    num_cols = [c for c in dfs[0].columns if c != model_col]
    stacked  = np.stack([df[num_cols].values for df in dfs], axis=0)  # (S, rows, cols)
    mean_    = stacked.mean(axis=0)
    std_     = stacked.std(axis=0)
    df_mean  = dfs[0][[model_col]].copy().reset_index(drop=True)
    df_std   = dfs[0][[model_col]].copy().reset_index(drop=True)
    for j, col in enumerate(num_cols):
        df_mean[col] = mean_[:, j]
        df_std[col]  = std_[:, j]
    return df_mean, df_std


def print_table(df_mean, df_std, title, float_fmt=".4f"):
    print(f"\n{'='*100}\n{title}\n{'='*100}")
    num_cols = [c for c in df_mean.columns if c != "model"]
    header   = f"{'model':>45s}" + "".join(f"  {c:>14s}" for c in num_cols)
    print(header)
    for i, row in df_mean.iterrows():
        line = f"{row['model']:>45s}"
        for col in num_cols:
            m = row[col]
            s = df_std.loc[i, col] if len(df_std) > 0 else 0.0
            if pd.isna(m):
                line += f"  {'':>14s}"
            elif s > 0:
                line += f"  {m:{float_fmt}} ±{s:.3f}"
            else:
                line += f"  {m:{float_fmt}}"
        print(line)


# ─────────────────────────────────────────────────────────────────────────────
# Load frozen artifacts (shared across seeds unless output_dirs differ)
# ─────────────────────────────────────────────────────────────────────────────

def load_artifacts(out_dir):
    a = {}
    a["y_train"]           = np.load(f"{out_dir}/train_y.npy")
    a["y_test"]            = np.load(f"{out_dir}/test_y.npy")
    a["X_struct_train"]    = np.load(f"{out_dir}/train_struct.npy")
    a["X_struct_test"]     = np.load(f"{out_dir}/test_struct.npy")
    a["lstm_feat_train"]   = np.load(f"{out_dir}/train_lstm.npy")
    a["lstm_feat_test"]    = np.load(f"{out_dir}/test_lstm.npy")
    a["pred_struct_train"] = np.load(f"{out_dir}/pred_struct_train.npy")
    a["pred_m1"]           = np.load(f"{out_dir}/pred_m1.npy")
    a["pred_m5"]           = np.load(f"{out_dir}/pred_m5.npy")
    a["expert_A"]          = joblib.load(f"{out_dir}/expert_A.pkl")
    a["expert_B"]          = joblib.load(f"{out_dir}/expert_B.pkl")
    a["gate_clf"]          = joblib.load(f"{out_dir}/gate_clf.pkl")
    return a


# ─────────────────────────────────────────────────────────────────────────────
# Per-seed run
# ─────────────────────────────────────────────────────────────────────────────

def run_seed(seed, arts):
    """Train all sklearn components with `seed`, return {df_t1, df_t2, df_t4, pred_m5}."""
    np.random.seed(seed)

    # Unpack frozen artifacts
    y_train           = arts["y_train"]
    y_test            = arts["y_test"]
    X_struct_train    = arts["X_struct_train"]
    X_struct_test     = arts["X_struct_test"]
    lstm_feat_train   = arts["lstm_feat_train"]
    lstm_feat_test    = arts["lstm_feat_test"]
    pred_struct_train = arts["pred_struct_train"]
    pred_m1           = arts["pred_m1"]   # struct only = pred_m1

    # Tail thresholds (from training LOS)
    tail_thresholds = {
        "P75": float(np.percentile(y_train, 75)),
        "P90": float(np.percentile(y_train, 90)),
        "P95": float(np.percentile(y_train, 95)),
        "P99": float(np.percentile(y_train, 99)),
    }
    P90_LOS     = tail_thresholds["P90"]
    tail_mask   = y_train >= P90_LOS
    resid_train = y_train - pred_struct_train
    _true       = y_test >= P90_LOS

    anchor_mae  = float(mean_absolute_error(y_test, pred_m1))
    anchor_tail = float(mean_absolute_error(y_test[_true], pred_m1[_true]))

    X_exp_train = np.hstack([X_struct_train, lstm_feat_train])
    X_exp_test  = np.hstack([X_struct_test,  lstm_feat_test])

    # Sigmoid gate (TABLE 2 G1): threshold = P90, temperature = P90/4
    def sigmoid_gate(x, thr, temp):
        return (1.0 / (1.0 + np.exp(-(x - thr) / temp))).astype(np.float32)

    soft_gate_test = sigmoid_gate(pred_m1, P90_LOS, P90_LOS / 4.0)

    em   = lambda name, y_pred: eval_metrics(name, y_test, y_pred, tail_thresholds, P90_LOS)
    hgbr = lambda **kw: HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.03, max_leaf_nodes=31,
        random_state=seed, **kw)

    # ── Load frozen M5 experts and gate from training artifacts ───────────────
    pred_resid_A = arts["expert_A"].predict(X_exp_test).astype(np.float32)
    pred_resid_B = arts["expert_B"].predict(X_exp_test).astype(np.float32)
    tail_prob    = arts["gate_clf"].predict_proba(X_exp_test)[:, 1].astype(np.float32)

    def moe_pred(gate):
        g = np.asarray(gate, dtype=np.float32)
        return pred_m1 + (1 - g) * pred_resid_A + g * pred_resid_B

    def gate_metrics(name, gate, clf_tp=None):
        row = em(name, moe_pred(gate))
        if clf_tp is not None:
            _p  = np.asarray(clf_tp) > 0.5
            pr  = float((_p & _true).sum() / max(_p.sum(), 1))
            re  = float((_p & _true).sum() / max(_true.sum(), 1))
            row["clf_precision"] = round(pr, 3)
            row["clf_recall"]    = round(re, 3)
            row["clf_F1"]        = round(2*pr*re / max(pr+re, 1e-8), 3)
        return row

    # ── Oracle + misc fixed gates ─────────────────────────────────────────────
    hard_gate   = (pred_m1 >= P90_LOS).astype(np.float32)
    oracle_gate = (y_test >= P90_LOS).astype(np.float32)

    # =========================================================================
    # TABLE 2  Gate ablation
    # =========================================================================

    results_t2 = [
        gate_metrics("G0  Expert_A_only  (g=0)",        np.zeros(len(y_test))),
        gate_metrics("G1  Soft_gate      (sigmoid)",     soft_gate_test),
        gate_metrics("G2  Hard_gate      (struct≥P90)",  hard_gate,   hard_gate),
        gate_metrics("G3  LightGBM classifier [OURS]",   tail_prob,   tail_prob),
        gate_metrics("G4  Oracle [UB]",                  oracle_gate, oracle_gate),
    ]

    df_t2 = pd.DataFrame(results_t2)
    df_t2["ΔMAE"]          = (df_t2["MAE"]          - anchor_mae ).round(4)
    df_t2["ΔTail_MAE_P90"] = (df_t2["Tail_MAE_P90"] - anchor_tail).round(4)

    best_gate_vec = tail_prob  # G3 is always the chosen gate

    # =========================================================================
    # TABLE 1  Main results  (building-block progression)
    # =========================================================================

    pred_m5 = arts["pred_m5"]

    # M2: struct + LSTM ungated — LightGBM on [struct | lstm], no gate
    lgbm_m2 = LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=63,
                              random_state=seed, verbose=-1, n_jobs=-1)
    lgbm_m2.fit(X_exp_train, y_train)
    pred_m2 = lgbm_m2.predict(X_exp_test).astype(np.float32)

    # M3: struct + LSTM gated — gate blends M1 (no text) and M2 (with text)
    pred_m3 = ((1 - best_gate_vec) * pred_m1 + best_gate_vec * pred_m2)

    # M4: struct + LSTM + single-expert residual, applied via gate
    single_expert = hgbr()
    single_expert.fit(X_exp_train, resid_train)
    pred_m4 = (pred_m1 + best_gate_vec * single_expert.predict(X_exp_test).astype(np.float32))

    t1_rows = [
        em("M1  struct only",                         pred_m1),
        em("M2  struct + LSTM (ungated)",              pred_m2),
        em("M3  struct + LSTM (gated)",                pred_m3),
        em("M4  struct + LSTM + residual (1 expert)",  pred_m4),
        em("M5  MoE residual [OURS]",                  pred_m5),
    ]

    df_t1 = pd.DataFrame(t1_rows)
    df_t1["ΔMAE"]          = (df_t1["MAE"]          - anchor_mae ).round(4)
    df_t1["ΔTail_MAE_P90"] = (df_t1["Tail_MAE_P90"] - anchor_tail).round(4)

    return {
        "df_t1": df_t1, "df_t2": df_t2,
        "pred_m1": pred_m1, "pred_m5": pred_m5, "y_test": y_test,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

all_runs = []
for seed, out_dir in zip(SEEDS, SEED_DIRS):
    print(f"\n{'─'*60}\nSeed {seed}  |  {out_dir}\n{'─'*60}")
    arts = load_artifacts(out_dir)
    result = run_seed(seed, arts)
    all_runs.append(result)
    print(f"  Seed {seed} done")

# ── Aggregate ─────────────────────────────────────────────────────────────────
single_seed = len(SEEDS) == 1

for table_key, title in [
    ("df_t2", "TABLE 2: Gate Ablation for MoE Residual"),
    ("df_t1", "TABLE 1: Main Results"),
]:
    dfs = [r[table_key] for r in all_runs]
    df_mean, df_std = agg_df_list(dfs)
    if single_seed:
        df_std_show = pd.DataFrame(columns=df_mean.columns)   # hide ± when n=1
    else:
        df_std_show = df_std
    csv_key = {"df_t1": "table1", "df_t2": "table2"}[table_key]
    print_table(df_mean, df_std_show, title)
    df_mean.to_csv(f"{SEED_DIRS[0]}/{csv_key}_mean.csv", index=False)
    if not single_seed:
        df_std.to_csv(f"{SEED_DIRS[0]}/{csv_key}_std.csv", index=False)

# ── Bootstrap significance: M1 vs M5 ─────────────────────────────────────────
print(f"\n{'='*60}\nBootstrap significance test: M1 vs M5 ({args.n_boot} replicates)\n{'='*60}")
for i, r in enumerate(all_runs):
    y    = r["y_test"]
    d, p = bootstrap_pvalue(y, r["pred_m1"], r["pred_m5"], n_boot=args.n_boot, seed=SEEDS[i])
    print(f"  Seed {SEEDS[i]}: ΔMAE(M1-M5)={d:+.4f}  p={p:.4f}"
          + ("  *" if p < 0.05 else ""))

if not single_seed:
    # Pooled test on concatenated predictions
    y_all     = np.concatenate([r["y_test"]   for r in all_runs])
    pred_m1_all = np.concatenate([r["pred_m1"] for r in all_runs])
    pred_m5_all = np.concatenate([r["pred_m5"] for r in all_runs])
    d, p = bootstrap_pvalue(y_all, pred_m1_all, pred_m5_all, n_boot=args.n_boot)
    print(f"  Pooled ({len(SEEDS)} seeds): ΔMAE={d:+.4f}  p={p:.4f}"
          + ("  *" if p < 0.05 else ""))

print(f"\nAll results saved to {SEED_DIRS[0]}/")
