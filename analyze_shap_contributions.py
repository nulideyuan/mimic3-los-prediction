"""
analyze_shap_contributions.py

SHAP analysis on M5's Expert A (bulk) and Expert B (tail).

Both experts take [struct | lstm] as input and predict the residual
(LOS - pred_struct).  Their SHAP values show how each feature contributes
to M5's residual correction — i.e., what text adds on top of structured.

Key design:
  Group-level (top panel):
    Structured block : mean_patients( |Σ_k SHAP_k| )   — net effect in days
    Text block       : mean_patients( |Σ_j Σ_d SHAP_jd| ) — net effect in days
    Both are signed sums over the full block, then absolute value, so sign
    cancellations within each block are preserved.  Result is in days-residual
    and directly comparable between groups.

  Feature-level (detail panels):
    Structured features : mean |SHAP_k| over in-scope patients
    Text sections       : mean |Σ_d SHAP_d| over covered patients
    Each group ranked within itself — no cross-group axis mixing.

Outputs saved to <out>/shap_*:
  shap_combined_expA.png/pdf   — Expert A (bulk patients)
  shap_combined_expB.png/pdf   — Expert B (tail patients)
  shap_section_rank.csv        — per-section net SHAP, both experts
  shap_feature_rank.csv        — full feature-level SHAP table

Usage:
  python mimic3_dataset/analyze_shap_contributions.py \
      --out outputs_mimic3_lstm_full_gated_residual
"""

import os
import json
import pickle
import argparse
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib

from utils import SECTIONS_TO_USE

parser = argparse.ArgumentParser()
parser.add_argument("--out", default="outputs_mimic3_lstm_full_gated_residual")
parser.add_argument("--top-struct", type=int, default=20,
                    help="Top-N structured features to show in detail panel")
args = parser.parse_args()
OUT = args.out

# ── Load artifacts ────────────────────────────────────────────────────────────

print("Loading artifacts...")

with open(f"{OUT}/model_config.json") as f:
    cfg = json.load(f)

hidden_dim = cfg["hidden_dim"]
n_sections = cfg["n_sections"]

expert_A = joblib.load(f"{OUT}/expert_A.pkl")
expert_B = joblib.load(f"{OUT}/expert_B.pkl")

with open(f"{OUT}/feat_cols.json") as f:
    feat_cols = json.load(f)

Xte_struct = np.load(f"{OUT}/test_struct.npy")
Xte_lstm   = np.load(f"{OUT}/test_lstm.npy")
Xte_full   = np.hstack([Xte_struct, Xte_lstm])
y_te       = np.load(f"{OUT}/test_y.npy")
y_tr       = np.load(f"{OUT}/train_y.npy")
test_hadm  = np.load(f"{OUT}/test_hadm.npy").tolist()

n_struct = len(feat_cols)
print(f"  Feature matrix: {Xte_full.shape}  "
      f"(struct={n_struct}, lstm={Xte_lstm.shape[1]})")

# ── Patient split: bulk vs tail ───────────────────────────────────────────────

p90_thresh = float(np.percentile(y_tr, 90))
print(f"  P90 threshold (train): {p90_thresh:.2f} days")

bulk_mask = y_te < p90_thresh
tail_mask = y_te >= p90_thresh

Xte_bulk = Xte_full[bulk_mask]
Xte_tail = Xte_full[tail_mask]
test_hadm_bulk = [h for h, m in zip(test_hadm, bulk_mask) if m]
test_hadm_tail = [h for h, m in zip(test_hadm, tail_mask) if m]

print(f"  Bulk (< P90): {bulk_mask.sum()}   Tail (≥ P90): {tail_mask.sum()}")

for label, model, X in [("Expert A", expert_A, Xte_bulk),
                         ("Expert B", expert_B, Xte_tail)]:
    model_n = getattr(model, "n_features_in_", None)
    if model_n is not None and model_n != X.shape[1]:
        raise ValueError(f"{label} expects {model_n} features but input has {X.shape[1]}.")

# ── Section coverage ──────────────────────────────────────────────────────────

print("Loading seqs for coverage...")
with open(f"{OUT}/seqs.pkl", "rb") as f:
    seqs = pickle.load(f)

def compute_coverage(hadm_ids):
    return np.array([
        [seqs[SECTIONS_TO_USE[j]].get(int(hid)) is not None for hid in hadm_ids]
        for j in range(n_sections)
    ], dtype=bool)   # (n_sections, n_patients)

covered_bulk = compute_coverage(test_hadm_bulk)
covered_tail = compute_coverage(test_hadm_tail)

# ── Feature names ─────────────────────────────────────────────────────────────

lstm_feat_names = [f"{sec}__h{d}"
                   for sec in SECTIONS_TO_USE
                   for d in range(hidden_dim)]
all_feat_names = feat_cols + lstm_feat_names

# ── SHAP + ranking ────────────────────────────────────────────────────────────

def compute_shap(model, X, label):
    print(f"Computing SHAP for {label} (n={len(X)})...")
    sv = shap.TreeExplainer(model).shap_values(X)
    print(f"  Done. shape: {sv.shape}")
    return sv

def build_rank_table(sv, covered):
    """Per-feature ranking DataFrame + group-level totals."""
    coverage_pct = covered.mean(axis=1) * 100
    rows = []

    for k, col in enumerate(feat_cols):
        rows.append({
            "feature":       col,
            "type":          "structured",
            "section":       col,
            "mean_abs_shap": float(np.abs(sv[:, k]).mean()),
            "n_patients":    int(sv.shape[0]),
            "coverage_pct":  100.0,
        })

    for j, sec in enumerate(SECTIONS_TO_USE):
        dim_s = n_struct + j * hidden_dim
        dim_e = n_struct + (j + 1) * hidden_dim
        net_per_patient = np.abs(sv[:, dim_s:dim_e].sum(axis=1))
        in_scope = covered[j]
        if in_scope.sum() == 0:
            cov_shap, n_cov = 0.0, 0
        else:
            cov_shap = float(net_per_patient[in_scope].mean())
            n_cov    = int(in_scope.sum())
        rows.append({
            "feature":       sec,
            "type":          "text",
            "section":       sec,
            "mean_abs_shap": cov_shap,
            "n_patients":    n_cov,
            "coverage_pct":  float(coverage_pct[j]),
        })

    df = (pd.DataFrame(rows)
          .sort_values("mean_abs_shap", ascending=False)
          .reset_index(drop=True))

    # Group-level: net effect of each block per patient, then mean |·|
    struct_group = float(np.abs(sv[:, :n_struct].sum(axis=1)).mean())
    text_group   = float(np.abs(sv[:, n_struct:].sum(axis=1)).mean())

    return df, struct_group, text_group

sv_A = compute_shap(expert_A, Xte_bulk, "Expert A (bulk)")
sv_B = compute_shap(expert_B, Xte_tail, "Expert B (tail)")

df_rank_A, struct_group_A, text_group_A = build_rank_table(sv_A, covered_bulk)
df_rank_B, struct_group_B, text_group_B = build_rank_table(sv_B, covered_tail)

# ── Save tables ───────────────────────────────────────────────────────────────

mean_abs_A = np.abs(sv_A).mean(axis=0)
mean_abs_B = np.abs(sv_B).mean(axis=0)

feat_df = pd.DataFrame({
    "feature":         all_feat_names,
    "mean_abs_shap_A": mean_abs_A,
    "mean_abs_shap_B": mean_abs_B,
})
feat_df.to_csv(f"{OUT}/shap_feature_rank.csv", index=False)

sec_cols = ["section", "type", "mean_abs_shap", "n_patients", "coverage_pct"]
sec_A = df_rank_A[sec_cols].rename(columns={
    "mean_abs_shap": "shap_A", "n_patients": "n_A", "coverage_pct": "cov_A"})
sec_B = df_rank_B[sec_cols].rename(columns={
    "mean_abs_shap": "shap_B", "n_patients": "n_B", "coverage_pct": "cov_B"})
section_df = sec_A.merge(sec_B[["section", "shap_B", "n_B", "cov_B"]],
                         on="section", how="outer")
section_df.to_csv(f"{OUT}/shap_section_rank.csv", index=False)

for label, df, sg, tg in [
    ("Expert A (bulk)", df_rank_A, struct_group_A, text_group_A),
    ("Expert B (tail)", df_rank_B, struct_group_B, text_group_B),
]:
    print(f"\n{label}: struct_group={sg:.4f}d  text_group={tg:.4f}d")
    print(df[df["type"] == "text"]
          .sort_values("mean_abs_shap", ascending=False)
          [["section", "coverage_pct", "mean_abs_shap", "n_patients"]]
          .to_string(index=False, float_format="%.4f"))

# ── Plot ──────────────────────────────────────────────────────────────────────

COLORS = {"structured": "#2166AC", "text": "#D6604D"}


def make_plot(sv, df_rank, struct_group, text_group,
              expert_label, scope_label, n_patients, out_stem):
    """
    3-panel figure:
      Top   : group-level bar (Structured block vs Text block)
      Bot-L : top structured features ranked by mean |SHAP|
      Bot-R : text sections ranked by mean |net SHAP| (within-group only)
    """
    top_struct = (df_rank[df_rank["type"] == "structured"]
                  .head(args.top_struct)
                  .sort_values("mean_abs_shap", ascending=True)
                  .reset_index(drop=True))
    text_rows  = (df_rank[df_rank["type"] == "text"]
                  .sort_values("mean_abs_shap", ascending=True)
                  .reset_index(drop=True))

    n_s = len(top_struct)
    n_t = len(text_rows)
    detail_h = max(n_s, n_t) * 0.38 + 1.2

    fig = plt.figure(figsize=(14, detail_h + 2.8))
    gs  = fig.add_gridspec(2, 2,
                           height_ratios=[1.6, detail_h],
                           hspace=0.45, wspace=0.5)

    ax_top    = fig.add_subplot(gs[0, :])
    ax_struct = fig.add_subplot(gs[1, 0])
    ax_text   = fig.add_subplot(gs[1, 1])

    # ── Top: group comparison ──
    g_vals   = [struct_group, text_group]
    g_labels = ["Structured\nfeatures", "Text sections\n(LSTM)"]
    g_colors = [COLORS["structured"], COLORS["text"]]
    bars_g   = ax_top.barh([0, 1], g_vals, color=g_colors,
                            edgecolor="white", linewidth=0.5, height=0.45)
    ax_top.set_yticks([0, 1])
    ax_top.set_yticklabels(g_labels, fontsize=10)
    max_g = max(g_vals)
    for bar, val in zip(bars_g, g_vals):
        ax_top.text(bar.get_width() + 0.005 * max_g,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f} d", va="center", fontsize=9.5, fontweight="bold")
    ax_top.set_xlabel("Mean |net SHAP| (days-residual)\n"
                      "[signed sum over entire block per patient, then |·|]", fontsize=8.5)
    ax_top.set_title(
        f"M5 {expert_label}  [{scope_label},  n={n_patients:,}]  —  block-level contribution",
        fontsize=11, fontweight="bold")
    ax_top.spines[["top", "right"]].set_visible(False)

    # ── Bottom-left: structured features ──
    bars_s = ax_struct.barh(range(n_s), top_struct["mean_abs_shap"],
                             color=COLORS["structured"], edgecolor="white", linewidth=0.5)
    ax_struct.set_yticks(range(n_s))
    ax_struct.set_yticklabels(top_struct["feature"].tolist(), fontsize=7.5)
    max_s = top_struct["mean_abs_shap"].max()
    for bar, val in zip(bars_s, top_struct["mean_abs_shap"]):
        ax_struct.text(bar.get_width() + 0.005 * max_s,
                       bar.get_y() + bar.get_height() / 2,
                       f"{val:.3f}", va="center", fontsize=6.5)
    ax_struct.set_xlabel("Mean |SHAP| (days-residual)", fontsize=8)
    ax_struct.set_title(f"Top {n_s} structured features", fontsize=9, fontweight="bold")
    ax_struct.spines[["top", "right"]].set_visible(False)

    # ── Bottom-right: text sections ──
    ylabels_t = [f"{row['section']}  [cov={row['coverage_pct']:.0f}%]"
                 for _, row in text_rows.iterrows()]
    bars_t = ax_text.barh(range(n_t), text_rows["mean_abs_shap"],
                           color=COLORS["text"], edgecolor="white", linewidth=0.5)
    ax_text.set_yticks(range(n_t))
    ax_text.set_yticklabels(ylabels_t, fontsize=7.5)
    max_t = text_rows["mean_abs_shap"].max() if text_rows["mean_abs_shap"].max() > 0 else 1.0
    for bar, val in zip(bars_t, text_rows["mean_abs_shap"]):
        ax_text.text(bar.get_width() + 0.005 * max_t,
                     bar.get_y() + bar.get_height() / 2,
                     f"{val:.3f}", va="center", fontsize=6.5)
    ax_text.set_xlabel("Mean |net SHAP| per covered patient (days-residual)\n"
                       "[|Σ hidden dims| per patient, then mean over covered patients]",
                       fontsize=8)
    ax_text.set_title("Text sections (LSTM)", fontsize=9, fontweight="bold")
    ax_text.spines[["top", "right"]].set_visible(False)

    plt.savefig(f"{OUT}/{out_stem}.png", dpi=150, bbox_inches="tight")
    plt.savefig(f"{OUT}/{out_stem}.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUT}/{out_stem}.png/pdf")


make_plot(sv_A, df_rank_A, struct_group_A, text_group_A,
          expert_label="Expert A",
          scope_label=f"bulk, LOS < P90={p90_thresh:.1f}d",
          n_patients=int(bulk_mask.sum()),
          out_stem="shap_combined_expA")

make_plot(sv_B, df_rank_B, struct_group_B, text_group_B,
          expert_label="Expert B",
          scope_label=f"tail, LOS ≥ P90={p90_thresh:.1f}d",
          n_patients=int(tail_mask.sum()),
          out_stem="shap_combined_expB")

print(f"\nSaved: {OUT}/shap_section_rank.csv")
print(f"       {OUT}/shap_feature_rank.csv")
