"""
run_interpretability.py  —  Clean interpretability analysis for EMNLP paper.

Design principles:
  - Structured features: TreeSHAP on M1 (structured-only LightGBM).
    One SHAP value per feature, no aggregation ambiguity.
  - Text sections: Permutation importance on Expert A/B.
    Model-agnostic; no hidden-dim interpretation required.
  - Temporal attention: Nursing sections only (median ≥4 notes/patient).
    Only positions where n_patients ≥ min_pos_n. No single-note imputation.
  - Recency: Only patients with ≥1 section having ≥2 notes (no 0.5 imputation).
    Bulk vs tail split (not LOS quartiles).

Figures  →  <interp-out>/
  fig1_struct_shap     TreeSHAP on M1 — top structured features
  fig2_text_perm       Permutation importance Expert A (bulk) vs B (tail)
  fig3_attn_nursing    Temporal attention by note position (nursing only)
  fig4_recency         Recency index violin: bulk vs tail
  fig5_error_bucket    MAE by LOS bucket M1 vs M5
  fig6_case_study      Short-stay vs long-stay attention case study

CSVs  →  <interp-out>/
  struct_shap_rank.csv
  text_perm_importance.csv
  attn_by_position.csv
  recency_by_patient.csv

Usage:
  python mimic3_dataset/run_interpretability.py \\
      --out      outputs_mimic3_lstm_full_gated_residual \\
      --interp-out outputs_mimic3_interpretability
"""

import os, sys, json, pickle, argparse, warnings
import numpy as np
import pandas as pd
import torch
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import mean_absolute_error

sys.path.insert(0, os.path.dirname(__file__))
from utils import (LSTMWithAttention, pad_seq, SECTIONS_TO_USE,
                   NURSING_SECTIONS, RADIOLOGY_SECTIONS, NURSING_OTHER_SECTIONS)

warnings.filterwarnings("ignore")

# ── Args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--out",         default="outputs_mimic3_lstm_full_gated_residual")
parser.add_argument("--interp-out",  default="outputs_mimic3_interpretability")
parser.add_argument("--top-struct",  type=int, default=25)
parser.add_argument("--min-pos-n",   type=int, default=50)
parser.add_argument("--perm-repeats",type=int, default=5)
parser.add_argument("--skip-shap",   action="store_true")
args = parser.parse_args()

OUT  = args.out
IOUT = args.interp_out
os.makedirs(IOUT, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.dpi": 150, "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 9,  "ytick.labelsize": 9,
    "legend.fontsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
})
BLUE = "#2166AC"; ORANGE = "#D6604D"; GRAY = "#888888"

SEC_ABBR = {
    "nursing_assessment":                "Nurs-Assess",
    "nursing_action":                    "Nurs-Action",
    "nursing_response":                  "Nurs-Response",
    "nursing_plan":                      "Nurs-Plan",
    "radiology_wet_read":                "Rad-WetRead",
    "radiology_indication":              "Rad-Indic",
    "radiology_technique":               "Rad-Tech",
    "radiology_impression":              "Rad-Impr",
    "nursing_other_other_nursing_other": "NursOth-Other",
    "nursing_other_respiratory_care":    "NursOth-Resp",
}
SEC_COLOR = {**{s: "#4CAF50" for s in NURSING_SECTIONS},
             **{s: "#FF9800" for s in RADIOLOGY_SECTIONS},
             **{s: "#E91E63" for s in NURSING_OTHER_SECTIONS}}

# ── Load artifacts ─────────────────────────────────────────────────────────────

print("Loading artifacts...")
with open(f"{OUT}/model_config.json") as f: cfg = json.load(f)
HIDDEN    = cfg["hidden_dim"]
SEC_EMB   = cfg.get("sec_emb_dim", 32)
MAX_NOTES = cfg.get("max_notes", 8)
N_SEC     = len(SECTIONS_TO_USE)

with open(f"{OUT}/feat_cols.json") as f: feat_cols = json.load(f)
n_struct = len(feat_cols)

y_te       = np.load(f"{OUT}/test_y.npy")
y_tr       = np.load(f"{OUT}/train_y.npy")
Xte_struct = np.load(f"{OUT}/test_struct.npy")
Xte_lstm   = np.load(f"{OUT}/test_lstm.npy")
Xte_full   = np.hstack([Xte_struct, Xte_lstm])
pred_m1    = np.load(f"{OUT}/pred_m1.npy")
pred_m5    = np.load(f"{OUT}/pred_m5.npy")
test_hadm  = np.load(f"{OUT}/test_hadm.npy").tolist()

P90     = float(np.percentile(y_tr, 90))
tail_te = y_te >= P90
bulk_te = ~tail_te

m1_model = joblib.load(f"{OUT}/lgbm_structured_only.pkl")
expert_A = joblib.load(f"{OUT}/expert_A.pkl")
expert_B = joblib.load(f"{OUT}/expert_B.pkl")

print("  Loading seqs.pkl...")
with open(f"{OUT}/seqs.pkl", "rb") as f: seqs = pickle.load(f)

print("  Loading LSTM model...")
lstm_model = LSTMWithAttention(input_dim=768, n_sections=N_SEC,
                                sec_emb_dim=SEC_EMB, hidden_dim=HIDDEN)
lstm_model.load_state_dict(torch.load(f"{OUT}/lstm_model.pt", map_location="cpu"))
lstm_model.eval()

sec_covered = np.array([
    [seqs[SECTIONS_TO_USE[j]].get(int(h)) is not None for h in test_hadm]
    for j in range(N_SEC)
], dtype=bool)
coverage_pct = sec_covered.mean(axis=1) * 100
print(f"  n_test={len(y_te):,}  P90={P90:.1f}d  "
      f"bulk={bulk_te.sum():,}  tail={tail_te.sum():,}")

# ── Attention helper ──────────────────────────────────────────────────────────

@torch.no_grad()
def attn_weights_for(hid, sec_idx):
    sec = SECTIONS_TO_USE[sec_idx]
    arr = seqs[sec].get(int(hid))
    if arr is None: return None, 0
    n_real = min(len(arr), MAX_NOTES)
    x  = torch.tensor(pad_seq(arr, MAX_NOTES)[None], dtype=torch.float32)
    si = torch.tensor([sec_idx], dtype=torch.long)
    sec_e = lstm_model.sec_emb(si).unsqueeze(1).expand(-1, x.size(1), -1)
    out, _ = lstm_model.lstm(torch.cat([x, sec_e], dim=-1))
    w = torch.softmax(lstm_model.attn(out), dim=1).squeeze().cpu().numpy()
    w[n_real:] = 0.0
    s = w[:n_real].sum()
    if s > 1e-8: w[:n_real] /= s
    return w[:n_real], n_real

# ══════════════════════════════════════════════════════════════════════════════
# FIG 1 — TreeSHAP on M1 (structured features only)
# ══════════════════════════════════════════════════════════════════════════════

if not args.skip_shap:
    print("\n[Fig 1] TreeSHAP on M1 (structured only)...")
    explainer = shap.TreeExplainer(m1_model)
    sv_m1     = explainer.shap_values(Xte_struct)       # (n_test, n_struct)
    mean_abs  = np.abs(sv_m1).mean(axis=0)              # (n_struct,)

    shap_df = (pd.DataFrame({"feature": feat_cols, "mean_abs_shap": mean_abs})
               .sort_values("mean_abs_shap", ascending=False)
               .reset_index(drop=True))
    shap_df.to_csv(f"{IOUT}/struct_shap_rank.csv", index=False)

    top = shap_df.head(args.top_struct).sort_values("mean_abs_shap", ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(5, len(top)*0.38)))
    bars = ax.barh(range(len(top)), top["mean_abs_shap"],
                   color=BLUE, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"].tolist(), fontsize=8)
    mx = top["mean_abs_shap"].max()
    for bar, val in zip(bars, top["mean_abs_shap"]):
        ax.text(bar.get_width()+0.003*mx, bar.get_y()+bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=7)
    ax.set_xlabel("Mean |SHAP| (days LOS)", fontsize=10)
    ax.set_title(f"Structured Feature Importance (TreeSHAP on M1)\n"
                 f"n={len(y_te):,} test patients", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{IOUT}/fig1_struct_shap.png", dpi=150, bbox_inches="tight")
    plt.savefig(f"{IOUT}/fig1_struct_shap.pdf", bbox_inches="tight")
    plt.close()
    print(f"  Saved: {IOUT}/fig1_struct_shap.png/pdf")
    print(f"  Top 5: {shap_df.head(5)[['feature','mean_abs_shap']].to_string(index=False)}")
else:
    print("\n[Fig 1] Skipping SHAP (--skip-shap)")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 1b — Section recency: bulk vs tail per section (all sections, multi-note)
# ══════════════════════════════════════════════════════════════════════════════

print("\n[Fig 1b] Section recency (bulk vs tail per section)...")

sec_recency = {j: {"bulk": [], "tail": []} for j in range(N_SEC)}

for i, hid in enumerate(test_hadm):
    is_tail = bool(tail_te[i])
    for j in range(N_SEC):
        w, n_real = attn_weights_for(hid, j)
        if w is None or n_real < 2:
            continue
        pos = np.arange(n_real) / (n_real - 1)
        rec = float(np.dot(pos, w))
        grp = "tail" if is_tail else "bulk"
        sec_recency[j][grp].append(rec)

# Build summary: mean recency per section per group
rec_rows = []
for j, sec in enumerate(SECTIONS_TO_USE):
    b = sec_recency[j]["bulk"]; t = sec_recency[j]["tail"]
    rec_rows.append({
        "section":        sec,
        "abbr":           SEC_ABBR[sec],
        "bulk_n":         len(b),
        "tail_n":         len(t),
        "bulk_mean_rec":  float(np.mean(b)) if b else np.nan,
        "tail_mean_rec":  float(np.mean(t)) if t else np.nan,
        "bulk_std_rec":   float(np.std(b))  if b else np.nan,
        "tail_std_rec":   float(np.std(t))  if t else np.nan,
        "diff_tail_bulk": float(np.mean(t) - np.mean(b)) if (b and t) else np.nan,
    })
df_secr = pd.DataFrame(rec_rows)
df_secr.to_csv(f"{IOUT}/section_recency.csv", index=False)
print(df_secr[["abbr","bulk_n","tail_n","bulk_mean_rec","tail_mean_rec","diff_tail_bulk"]]
      .to_string(index=False, float_format="%.3f"))

# Plot: grouped bars, one per section, bulk vs tail, sorted by diff
valid = df_secr.dropna(subset=["bulk_mean_rec","tail_mean_rec"])
valid = valid.sort_values("diff_tail_bulk", ascending=False).reset_index(drop=True)

x  = np.arange(len(valid)); bw = 0.35
fig, ax = plt.subplots(figsize=(13, 5))
ax.bar(x-bw/2, valid["bulk_mean_rec"], bw,
       yerr=valid["bulk_std_rec"]/np.sqrt(valid["bulk_n"].clip(1)),
       color=BLUE, alpha=0.85, label="Bulk (LOS<P90)",
       error_kw=dict(elinewidth=1, capsize=3))
ax.bar(x+bw/2, valid["tail_mean_rec"], bw,
       yerr=valid["tail_std_rec"]/np.sqrt(valid["tail_n"].clip(1)),
       color=ORANGE, alpha=0.85, label="Tail (LOS≥P90)",
       error_kw=dict(elinewidth=1, capsize=3))

for xi, row in valid.iterrows():
    d = row["diff_tail_bulk"]
    col = ORANGE if d > 0 else BLUE
    ax.text(xi, max(row["bulk_mean_rec"], row["tail_mean_rec"]) + 0.015,
            f"{d:+.2f}", ha="center", fontsize=7.5, color=col, fontweight="bold")

ax.axhline(0.5, color=GRAY, linestyle="--", linewidth=1, label="Uniform (0.5)")
ax.set_xticks(x)
ax.set_xticklabels(
    [f"{r['abbr']}\nnB={r['bulk_n']} nT={r['tail_n']}" for _, r in valid.iterrows()],
    fontsize=7.5)
ax.set_ylabel("Mean Recency Index\n(0=earliest note, 1=latest)", fontsize=10)
ax.set_title("Attention Recency by Section: Bulk vs Tail Patients\n"
             "(multi-note patients only; ±SEM error bars; sorted by tail−bulk diff)",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9); ax.set_ylim(0.3, 0.85)
ax.grid(axis="y", linewidth=0.4, alpha=0.4)
plt.tight_layout()
plt.savefig(f"{IOUT}/fig1b_section_recency.png", dpi=150, bbox_inches="tight")
plt.savefig(f"{IOUT}/fig1b_section_recency.pdf", bbox_inches="tight")
plt.close()
print(f"  Saved: {IOUT}/fig1b_section_recency.png/pdf")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 2 — Text section permutation importance (Expert A vs B)
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n[Fig 2] Text section permutation importance ({args.perm_repeats} repeats)...")

y_resid = y_te - pred_m1
rng     = np.random.default_rng(42)
imp     = {"A": np.zeros((N_SEC, args.perm_repeats)),
           "B": np.zeros((N_SEC, args.perm_repeats))}
n_eval  = {"A": np.zeros(N_SEC, dtype=int),
           "B": np.zeros(N_SEC, dtype=int)}

for j in range(N_SEC):
    ds, de = n_struct + j*HIDDEN, n_struct + (j+1)*HIDDEN
    for key, expert, mask in [("A", expert_A, bulk_te), ("B", expert_B, tail_te)]:
        m = sec_covered[j] & mask
        if m.sum() < 10: continue
        Xj, yj = Xte_full[m], y_resid[m]
        base = mean_absolute_error(yj, expert.predict(Xj))
        for r in range(args.perm_repeats):
            perm = rng.permutation(len(Xj))
            Xp = Xj.copy(); Xp[:, ds:de] = Xp[perm, ds:de]
            imp[key][j, r] = mean_absolute_error(yj, expert.predict(Xp)) - base
        n_eval[key][j] = int(m.sum())

imp_A_m = imp["A"].mean(axis=1); imp_A_s = imp["A"].std(axis=1)
imp_B_m = imp["B"].mean(axis=1); imp_B_s = imp["B"].std(axis=1)

# Save CSV
perm_rows = []
for j, sec in enumerate(SECTIONS_TO_USE):
    perm_rows.append({
        "section":      sec,
        "coverage_pct": round(coverage_pct[j], 1),
        "imp_A_mean":   round(imp_A_m[j], 4),
        "imp_A_std":    round(imp_A_s[j], 4),
        "n_A":          n_eval["A"][j],
        "imp_B_mean":   round(imp_B_m[j], 4),
        "imp_B_std":    round(imp_B_s[j], 4),
        "n_B":          n_eval["B"][j],
    })
pd.DataFrame(perm_rows).to_csv(f"{IOUT}/text_perm_importance.csv", index=False)

order = np.argsort(imp_B_m)[::-1]
x = np.arange(N_SEC); w = 0.38
fig, ax = plt.subplots(figsize=(12, 5.5))
ax.bar(x-w/2, imp_A_m[order], w, yerr=imp_A_s[order], color=BLUE, alpha=0.85,
       label=f"Expert A — bulk (LOS<P90, n={bulk_te.sum():,})",
       error_kw=dict(elinewidth=1.2, capsize=3))
ax.bar(x+w/2, imp_B_m[order], w, yerr=imp_B_s[order], color=ORANGE, alpha=0.85,
       label=f"Expert B — tail (LOS≥P90, n={tail_te.sum():,})",
       error_kw=dict(elinewidth=1.2, capsize=3))

xlbls = []
for i in order:
    sec = SECTIONS_TO_USE[i]
    nA  = n_eval["A"][i]; nB = n_eval["B"][i]
    xlbls.append(f"{SEC_ABBR[sec]}\ncov={coverage_pct[i]:.0f}%\nnA={nA} nB={nB}")
ax.set_xticks(x)
ax.set_xticklabels(xlbls, fontsize=7.5)
ax.set_ylabel("MAE increase when section permuted (days)", fontsize=10)
ax.set_title("Text Section Permutation Importance\n"
             "Expert A (bulk patients) vs Expert B (tail patients)\n"
             "(evaluated only on patients covered by each section; sorted by Expert B)",
             fontsize=11, fontweight="bold")
ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{IOUT}/fig2_text_perm.png", dpi=150, bbox_inches="tight")
plt.savefig(f"{IOUT}/fig2_text_perm.pdf", bbox_inches="tight")
plt.close()
print(f"  Saved: {IOUT}/fig2_text_perm.png/pdf")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 3 — Temporal attention (nursing sections, multi-note patients only)
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n[Fig 3] Temporal attention (min_pos_n={args.min_pos_n})...")

NURS_IDX = [SECTIONS_TO_USE.index(s) for s in NURSING_SECTIONS]

attn_data = {j: {"bulk_sum": np.zeros(MAX_NOTES), "tail_sum": np.zeros(MAX_NOTES),
                  "bulk_n":  np.zeros(MAX_NOTES, dtype=int),
                  "tail_n":  np.zeros(MAX_NOTES, dtype=int)} for j in NURS_IDX}

for i, hid in enumerate(test_hadm):
    is_tail = bool(tail_te[i])
    for j in NURS_IDX:
        w, n_real = attn_weights_for(hid, j)
        if w is None or n_real < 2:   # skip single-note patients
            continue
        grp = "tail" if is_tail else "bulk"
        for pos in range(n_real):
            attn_data[j][f"{grp}_sum"][pos] += w[pos]
            attn_data[j][f"{grp}_n"][pos]   += 1

# CSV
csv_rows = []
for j in NURS_IDX:
    sec = SECTIONS_TO_USE[j]
    for pos in range(MAX_NOTES):
        bn = attn_data[j]["bulk_n"][pos]; tn = attn_data[j]["tail_n"][pos]
        csv_rows.append({
            "section": sec, "position": pos+1,
            "bulk_n":  bn, "tail_n": tn,
            "bulk_mean_attn": attn_data[j]["bulk_sum"][pos]/bn if bn>0 else np.nan,
            "tail_mean_attn": attn_data[j]["tail_sum"][pos]/tn if tn>0 else np.nan,
        })
pd.DataFrame(csv_rows).to_csv(f"{IOUT}/attn_by_position.csv", index=False)

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
axes = axes.flatten()
MIN_N = args.min_pos_n

for ax, j in zip(axes, NURS_IDX):
    sec = SECTIONS_TO_USE[j]
    d   = attn_data[j]

    # positions where BOTH bulk AND tail have enough patients
    pos_bulk = [p for p in range(MAX_NOTES) if d["bulk_n"][p] >= MIN_N]
    pos_tail = [p for p in range(MAX_NOTES) if d["tail_n"][p] >= MIN_N]

    if pos_bulk:
        bulk_attn = [d["bulk_sum"][p]/d["bulk_n"][p] for p in pos_bulk]
        ax.plot([p+1 for p in pos_bulk], bulk_attn, "o-", color=BLUE,
                linewidth=2, markersize=5,
                label=f"Bulk (n≥{MIN_N} at pos)")
    if pos_tail:
        tail_attn = [d["tail_sum"][p]/d["tail_n"][p] for p in pos_tail]
        ax.plot([p+1 for p in pos_tail], tail_attn, "s--", color=ORANGE,
                linewidth=2, markersize=5,
                label=f"Tail (n≥{MIN_N} at pos)")

    ax.axhline(1/MAX_NOTES, color=GRAY, linestyle=":", linewidth=1,
               label=f"Uniform (1/{MAX_NOTES})")

    # n_patients annotation
    for pos in range(MAX_NOTES):
        bn = d["bulk_n"][pos]; tn = d["tail_n"][pos]
        if bn >= MIN_N or tn >= MIN_N:
            ax.annotate(f"b:{bn}\nt:{tn}",
                        xy=(pos+1, 0), xycoords=("data","axes fraction"),
                        xytext=(0, -32), textcoords="offset points",
                        fontsize=6, ha="center", color=GRAY,
                        annotation_clip=False)

    ax.set_title(SEC_ABBR[sec], fontsize=10, fontweight="bold")
    ax.set_xlabel("Note position (chronological, 1=earliest)", fontsize=8.5)
    ax.set_ylabel("Mean attention weight", fontsize=8.5)
    ax.set_xticks(range(1, MAX_NOTES+1))
    if ax is axes[0]: ax.legend(fontsize=8, loc="upper right")

fig.suptitle(
    f"Temporal Attention by Note Position — Nursing Sections Only\n"
    f"(multi-note patients only, n_real≥2; positions shown if n≥{MIN_N};\n"
    f" b=bulk count, t=tail count at each position)",
    fontsize=10, fontweight="bold")
plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.savefig(f"{IOUT}/fig3_attn_nursing.png", dpi=150, bbox_inches="tight")
plt.savefig(f"{IOUT}/fig3_attn_nursing.pdf", bbox_inches="tight")
plt.close()
print(f"  Saved: {IOUT}/fig3_attn_nursing.png/pdf")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 4 — Recency violin: bulk vs tail (multi-note patients only)
# ══════════════════════════════════════════════════════════════════════════════

print("\n[Fig 4] Recency violin (multi-note patients only)...")

recs = []
for i, hid in enumerate(test_hadm):
    pos_all, w_all = [], []
    for j in range(N_SEC):
        w, n_real = attn_weights_for(hid, j)
        if w is None or n_real < 2:   # skip single-note: no ordering signal
            continue
        pos = np.arange(n_real) / (n_real - 1)   # 0=earliest, 1=latest
        pos_all.extend(pos.tolist())
        w_all.extend(w.tolist())
    if not w_all:
        continue
    tot = sum(w_all)
    if tot < 1e-8: continue
    recs.append({
        "hadm_id":  hid,
        "los_days": float(y_te[i]),
        "is_tail":  bool(tail_te[i]),
        "recency":  sum(p*wt for p,wt in zip(pos_all,w_all)) / tot,
        "n_notes":  len(w_all),
    })

df_rec = pd.DataFrame(recs)
df_rec.to_csv(f"{IOUT}/recency_by_patient.csv", index=False)

bulk_rec = df_rec.loc[~df_rec["is_tail"], "recency"].values
tail_rec = df_rec.loc[ df_rec["is_tail"], "recency"].values

fig, ax = plt.subplots(figsize=(7, 5))
parts = ax.violinplot([bulk_rec, tail_rec], positions=[0, 1],
                       widths=0.55, showmedians=False, showextrema=False)
for pc, col in zip(parts["bodies"], [BLUE, ORANGE]):
    pc.set_facecolor(col); pc.set_alpha(0.70)
    pc.set_edgecolor("black"); pc.set_linewidth(0.8)

for k, (vals, col) in enumerate([(bulk_rec, BLUE), (tail_rec, ORANGE)]):
    q25, med, q75 = np.percentile(vals, [25, 50, 75])
    ax.plot([k,k], [q25,q75], color="black", linewidth=2.5, solid_capstyle="round")
    ax.plot(k, med, "_", color="black", markersize=14, markeredgewidth=2.5)
    ax.text(k, ax.get_ylim()[1] if ax.get_ylim()[1] < 1 else 0.95,
            f"n={len(vals):,}\nmed={med:.3f}", ha="center", fontsize=9, va="bottom")

ax.axhline(0.5, color=GRAY, linestyle="--", linewidth=1.2, label="Uniform (0.5)")
ax.set_xticks([0, 1])
ax.set_xticklabels([f"Bulk (LOS<P90={P90:.0f}d)\nn={len(bulk_rec):,}",
                    f"Tail (LOS≥P90={P90:.0f}d)\nn={len(tail_rec):,}"], fontsize=10)
ax.set_ylabel("Recency Index\n(0=attention on earliest note, 1=latest note)", fontsize=10)
ax.set_title("Attention Recency Index: Bulk vs Tail Patients\n"
             "(multi-note patients only — single-note sections excluded)", fontsize=11,
             fontweight="bold")
ax.legend(loc="upper left")
ax.set_ylim(0.1, 1.0)
ax.grid(axis="y", linewidth=0.4, alpha=0.4)
plt.tight_layout()
plt.savefig(f"{IOUT}/fig4_recency.png", dpi=150, bbox_inches="tight")
plt.savefig(f"{IOUT}/fig4_recency.pdf", bbox_inches="tight")
plt.close()
print(f"  Saved: {IOUT}/fig4_recency.png/pdf")
print(f"  Bulk:  n={len(bulk_rec):,}  median={np.median(bulk_rec):.3f}")
print(f"  Tail:  n={len(tail_rec):,}  median={np.median(tail_rec):.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 5 — MAE per LOS bucket
# ══════════════════════════════════════════════════════════════════════════════

print("\n[Fig 5] MAE per LOS bucket...")
bins   = [0, 3, 7, 14, 30, np.inf]
labels = ["0–3d","3–7d","7–14d","14–30d","30+d"]
bucket = np.clip(np.digitize(y_te, bins)-1, 0, len(labels)-1)

fig, ax = plt.subplots(figsize=(9, 4.5))
x, bw = np.arange(len(labels)), 0.35
for i, (name, pred, col) in enumerate([("Struct only (M1)", pred_m1, GRAY),
                                         ("MoE residual (M5)", pred_m5, ORANGE)]):
    maes = [mean_absolute_error(y_te[bucket==b], pred[bucket==b])
            if (bucket==b).sum()>5 else np.nan for b in range(len(labels))]
    bars = ax.bar(x+(i-0.5)*bw, maes, bw, color=col, alpha=0.85, label=name)
    for bar, mae in zip(bars, maes):
        if not np.isnan(mae):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
                    f"{mae:.1f}", ha="center", fontsize=8)

bn = [(bucket==b).sum() for b in range(len(labels))]
ax.set_xticks(x)
ax.set_xticklabels([f"{l}\n(n={n:,})" for l,n in zip(labels,bn)])
ax.set_ylabel("MAE (days)"); ax.legend()
ax.set_title("MAE by LOS Bucket: M5 Gains Concentrated in Long-Stay Patients",
             fontweight="bold")
plt.tight_layout()
plt.savefig(f"{IOUT}/fig5_error_bucket.png", dpi=150, bbox_inches="tight")
plt.savefig(f"{IOUT}/fig5_error_bucket.pdf", bbox_inches="tight")
plt.close()
print(f"  Saved: {IOUT}/fig5_error_bucket.png/pdf")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 6 — Case study
# ══════════════════════════════════════════════════════════════════════════════

print("\n[Fig 6] Case study...")
NURS_P  = "processed/section_trajectory_long/nursing_sections_24h_long.parquet"
RAD_P   = "processed/section_trajectory_long/radiology_24h_long.parquet"
NURSO_P = "processed/section_trajectory_long/nursing_other_sections_24h_long.parquet"

def load_notes(hid):
    rows = []
    for path, allowed in [(NURS_P,NURSING_SECTIONS),(RAD_P,RADIOLOGY_SECTIONS),
                           (NURSO_P,NURSING_OTHER_SECTIONS)]:
        if not os.path.exists(path): continue
        df_p = pd.read_parquet(path)
        df_p.columns = df_p.columns.str.lower()
        df_p["hadm_id"] = df_p["hadm_id"].astype(int)
        df_p = df_p[(df_p["hadm_id"]==int(hid)) & df_p["section_name"].isin(allowed)
                    & (df_p["text"].fillna("").str.strip().str.len()>5)].copy()
        if "hours_from_admit" in df_p.columns:
            df_p = df_p.sort_values("hours_from_admit")
        rows.append(df_p[["section_name","hours_from_admit","text"]])
    if not rows:
        return pd.DataFrame(columns=["section_name","hours_from_admit","text"])
    return pd.concat(rows).sort_values("hours_from_admit").reset_index(drop=True)

def pick_patient(kind="short"):
    lo = np.percentile(y_te, 10 if kind=="short" else 90)
    hi = np.percentile(y_te, 25 if kind=="short" else 95)
    cands = [(test_hadm[i], y_te[i]) for i in range(len(test_hadm)) if lo<=y_te[i]<=hi]
    cands.sort(key=lambda x: x[1])
    for hid, los in cands:
        n_s = sum(1 for j in range(N_SEC) if seqs[SECTIONS_TO_USE[j]].get(int(hid)) is not None)
        if n_s >= 3: return hid, float(los)
    return None, None

def get_attentions(hid, notes_df):
    sec_idx = {s: 0 for s in SECTIONS_TO_USE}
    attns = []
    for _, row in notes_df.iterrows():
        sec = row["section_name"]
        if sec not in SECTIONS_TO_USE:
            attns.append(0.0); continue
        j = SECTIONS_TO_USE.index(sec)
        w, _ = attn_weights_for(hid, j)
        if w is None: attns.append(0.0); continue
        pos = sec_idx[sec]; sec_idx[sec] += 1
        attns.append(float(w[pos]) if pos < len(w) else 0.0)
    return attns

def fmt_h(h):
    h=max(0.0,float(h)); return f"{int(h):02d}:{int(round((h-int(h))*60)):02d}"

def plot_timeline(ax, notes_df, attns, title):
    if notes_df.empty: ax.set_title(f"{title}\n(no notes)"); return
    n=len(notes_df); ys=np.arange(n)[::-1]; mx=max(attns) if max(attns)>0 else 1.0
    ylbls=[]
    for idx,(_,row) in enumerate(notes_df.iterrows()):
        y=ys[idx]; a=attns[idx]; sec=row["section_name"]
        ax.barh(y, a/mx, height=0.65, color=SEC_COLOR.get(sec,GRAY),
                alpha=0.80, edgecolor="white", linewidth=0.4)
        ax.text(a/mx+0.015, y, f"{a:.3f}", va="center", fontsize=7, color="#444")
        short=(sec.replace("nursing_other_","nurs_oth/")
                  .replace("nursing_","nurs/").replace("radiology_","rad/"))
        ylbls.append(f"{fmt_h(row.get('hours_from_admit',0))}  {short}")
    ax.set_xlim(0,1.25); ax.set_ylim(-0.8,n-0.2)
    ax.set_yticks(ys); ax.set_yticklabels(ylbls, fontsize=8)
    ax.set_xlabel("Normalised attention weight", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.axvline(1.0,color=GRAY,linewidth=0.6,linestyle="--",alpha=0.5)
    ax.tick_params(axis="y",length=0)

short_hid, short_los = pick_patient("short")
long_hid,  long_los  = pick_patient("long")
print(f"  Short-stay: hadm_id={short_hid}  LOS={short_los:.1f}d")
print(f"  Long-stay:  hadm_id={long_hid}   LOS={long_los:.1f}d")

fig, axes = plt.subplots(1, 2, figsize=(15, 10))
for ax, hid, los, kind in [(axes[0],short_hid,short_los,"Short-stay"),
                             (axes[1],long_hid, long_los, "Long-stay")]:
    if hid is None: ax.set_title(f"{kind} — no suitable patient"); continue
    ndf = (load_notes(hid).groupby("section_name",group_keys=False)
           .head(MAX_NOTES).sort_values("hours_from_admit").reset_index(drop=True))
    plot_timeline(ax, ndf, get_attentions(hid, ndf),
                  f"{kind} (LOS={los:.1f}d, n={len(ndf)} notes)")

fig.legend(handles=[mpatches.Patch(color="#4CAF50",label="Nursing"),
                    mpatches.Patch(color="#FF9800",label="Radiology"),
                    mpatches.Patch(color="#E91E63",label="Nursing other")],
           loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5,-0.01))
fig.suptitle("24h Note Trajectory with Learned Attention Weights\n"
             "(bar = normalised attention weight; HH:MM = hours from admission;\n"
             " illustrative examples — not population-level evidence)",
             fontsize=10, y=1.01)
plt.tight_layout()
plt.savefig(f"{IOUT}/fig6_case_study.png", dpi=150, bbox_inches="tight")
plt.savefig(f"{IOUT}/fig6_case_study.pdf", bbox_inches="tight")
plt.close()
print(f"  Saved: {IOUT}/fig6_case_study.png/pdf")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"All outputs → {IOUT}/")
print(f"  fig1_struct_shap.png/pdf   TreeSHAP on M1 (structured)")
print(f"  fig2_text_perm.png/pdf     Permutation importance Expert A vs B")
print(f"  fig3_attn_nursing.png/pdf  Temporal attention (nursing, multi-note)")
print(f"  fig4_recency.png/pdf       Recency violin bulk vs tail")
print(f"  fig5_error_bucket.png/pdf  MAE by LOS bucket")
print(f"  fig6_case_study.png/pdf    Case study")
print(f"  struct_shap_rank.csv  |  text_perm_importance.csv")
print(f"  attn_by_position.csv  |  recency_by_patient.csv")
